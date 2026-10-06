import os
import re
import time
import uuid
import shutil
import argparse
import subprocess
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import gradio as gr
from gradio_image_annotation import image_annotator

from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device


WEIGHTS = os.getenv("HF_HUB", "weights")


def safe_filename(path):
    name = Path(path).stem
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def get_path(file_obj):
    if file_obj is None:
        return ""
    if isinstance(file_obj, (str, Path)):
        return str(file_obj)
    if hasattr(file_obj, "path") and file_obj.path:
        return str(file_obj.path)
    if hasattr(file_obj, "name") and file_obj.name:
        return str(file_obj.name)
    raise TypeError(f"Unsupported file type: {type(file_obj)}")


def normalize_files(files):
    if files is None:
        return []
    if not isinstance(files, list):
        files = [files]
    paths = []
    for f in files:
        p = get_path(f)
        if p:
            paths.append(p)
    return paths


def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found in PATH. Install FFmpeg first."
        )


def read_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read first frame: {video_path}")
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS: {video_path}")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid resolution: {video_path}")

    return {
        "first_frame": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        "fps": fps,
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "duration": n_frames / fps
    }


def get_bbox(annotation):
    if not isinstance(annotation, dict):
        return None

    boxes = annotation.get("boxes", [])
    if not boxes:
        return None

    box = boxes[0]
    return (
        int(round(float(box["xmin"]))),
        int(round(float(box["ymin"]))),
        int(round(float(box["xmax"]))),
        int(round(float(box["ymax"])))
    )


def clamp_bbox(bbox, width, height):
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, min(int(xmin), width - 1))
    ymin = max(0, min(int(ymin), height - 1))
    xmax = max(1, min(int(xmax), width))
    ymax = max(1, min(int(ymax), height))

    if xmax <= xmin or ymax <= ymin:
        raise ValueError(
            f"Invalid bounding box {bbox} for {width}x{height} video."
        )
    return xmin, ymin, xmax, ymax


def bbox_text(bbox):
    if bbox is None:
        return "No bounding box selected."
    xmin, ymin, xmax, ymax = bbox
    return (
        f"xmin = {xmin}\n"
        f"ymin = {ymin}\n"
        f"xmax = {xmax}\n"
        f"ymax = {ymax}\n"
        f"width = {xmax - xmin}\n"
        f"height = {ymax - ymin}"
    )


def make_annotation(video_path, bbox=None):
    info = read_video_info(video_path)
    data = {
        "image": info["first_frame"],
        "boxes": []
    }

    if bbox is not None:
        bbox = clamp_bbox(
            bbox,
            info["width"],
            info["height"]
        )
        xmin, ymin, xmax, ymax = bbox
        data["boxes"] = [{
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        }]

    return data


def make_preview(video_path, bbox):
    if not video_path or bbox is None:
        return None

    info = read_video_info(video_path)
    bbox = clamp_bbox(
        bbox,
        info["width"],
        info["height"]
    )

    frame = info["first_frame"].copy()
    xmin, ymin, xmax, ymax = bbox

    overlay = frame.copy()
    overlay[ymin:ymax, xmin:xmax] = [255, 0, 0]
    preview = cv2.addWeighted(
        frame,
        0.55,
        overlay,
        0.45,
        0
    )

    cv2.rectangle(
        preview,
        (xmin, ymin),
        (xmax - 1, ymax - 1),
        (255, 255, 0),
        3
    )

    return preview


def trim_video(
    input_video,
    start_time,
    end_time,
    output_path
):
    require_ffmpeg()

    info = read_video_info(input_video)
    duration = info["duration"]

    start_time = max(0.0, float(start_time))
    end_time = min(float(end_time), duration)

    if end_time <= start_time:
        raise ValueError(
            f"End time ({end_time:.3f}) must be greater than "
            f"start time ({start_time:.3f})."
        )

    # If the whole video is selected, no need to create another file.
    if (
        abs(start_time) < 1e-6
        and abs(end_time - duration) < 1.0 / max(info["fps"], 1.0)
    ):
        shutil.copy2(input_video, output_path)
        return output_path

    clip_duration = end_time - start_time

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_time:.6f}",
        "-i", input_video,
        "-t", f"{clip_duration:.6f}",
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed while cutting the video:\n\n"
            + result.stderr[-4000:]
        )

    if not os.path.exists(output_path):
        raise RuntimeError(
            f"FFmpeg completed but output was not created: {output_path}"
        )

    return output_path


def create_mask_video(input_video, bbox, output_mask):
    info = read_video_info(input_video)
    width = info["width"]
    height = info["height"]
    fps = info["fps"]

    bbox = clamp_bbox(
        bbox,
        width,
        height
    )

    xmin, ymin, xmax, ymax = bbox

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    # Lossless mask because read_mask() uses mask > 0.
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer = cv2.VideoWriter(
        str(output_mask),
        fourcc,
        fps,
        (width, height),
        True
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not create FFV1 mask video. "
            "Your FFmpeg/OpenCV build may not support FFV1."
        )

    frames = 0

    while True:
        ret, _ = cap.read()
        if not ret:
            break

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )
        mask[ymin:ymax, xmin:xmax] = 255

        mask_bgr = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR
        )
        writer.write(mask_bgr)
        frames += 1

    cap.release()
    writer.release()

    if frames == 0:
        raise RuntimeError(
            f"No mask frames generated: {output_mask}"
        )

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frames
    }


def verify_mask(mask_path, video_info):
    cap = cv2.VideoCapture(mask_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open generated mask: {mask_path}"
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        n_frames += 1

    cap.release()

    if width != video_info["width"]:
        raise RuntimeError(
            f"Mask width mismatch: {width} vs {video_info['width']}"
        )

    if height != video_info["height"]:
        raise RuntimeError(
            f"Mask height mismatch: {height} vs {video_info['height']}"
        )

    if abs(fps - video_info["fps"]) > 1e-3:
        raise RuntimeError(
            f"Mask FPS mismatch: {fps} vs {video_info['fps']}"
        )

    if n_frames != video_info["n_frames"]:
        raise RuntimeError(
            f"Mask frame count mismatch: {n_frames} vs "
            f"{video_info['n_frames']}"
        )


def ensure_output_resolution(
    output_video,
    target_width,
    target_height
):
    """
    DiffuEraser can internally resize according to max_img_size.
    This function makes the final file spatially match the
    trimmed input resolution.
    """
    require_ffmpeg()

    info = read_video_info(output_video)

    if (
        info["width"] == target_width
        and info["height"] == target_height
    ):
        return output_video

    temp_output = (
        str(Path(output_video).with_suffix("")) +
        "_resized.mp4"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", output_video,
        "-vf", f"scale={target_width}:{target_height}:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        temp_output
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed while restoring output resolution:\n\n"
            + result.stderr[-4000:]
        )

    shutil.move(
        temp_output,
        output_video
    )

    return output_video


class VideoRemovalEngine:
    def __init__(self, args):
        self.args = args
        self.device = get_device()

        print("=" * 70)
        print("Initializing ProPainter + DiffuEraser")
        print(f"Device: {self.device}")
        print("=" * 70)

        self.video_inpainting_sd = DiffuEraser(
            self.device,
            args.base_model_path,
            args.vae_path,
            args.diffueraser_path,
            ckpt=args.ckpt
        )

        self.propainter = Propainter(
            args.propainter_model_dir,
            device=self.device
        )

        print("Models initialized.")

    def process_one(
        self,
        input_video,
        input_mask,
        output_dir
    ):
        args = self.args
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        name = safe_filename(input_video)
        priori_path = output_dir / f"{name}_priori.mp4"
        output_path = output_dir / f"{name}_diffueraser_result.mp4"

        input_info = read_video_info(input_video)

        print("\n" + "=" * 70)
        print(f"Processing: {input_video}")
        print(f"Mask: {input_mask}")
        print(f"Input: {input_info['width']}x{input_info['height']}")
        print(f"FPS: {input_info['fps']}")
        print(f"Frames: {input_info['n_frames']}")
        print("=" * 70)

        start_time = time.time()

        print("Running ProPainter...")
        self.propainter.forward(
            input_video,
            input_mask,
            str(priori_path),
            video_length=args.video_length,
            ref_stride=args.ref_stride,
            neighbor_length=args.neighbor_length,
            subvideo_length=args.subvideo_length,
            mask_dilation=args.mask_dilation_iter
        )

        if not priori_path.exists():
            raise RuntimeError(
                f"ProPainter did not create: {priori_path}"
            )

        print("Running DiffuEraser...")
        self.video_inpainting_sd.forward(
            input_video,
            input_mask,
            str(priori_path),
            str(output_path),
            max_img_size=args.max_img_size,
            video_length=args.video_length,
            mask_dilation_iter=args.mask_dilation_iter,
            guidance_scale=None
        )

        if not output_path.exists():
            raise RuntimeError(
                f"DiffuEraser did not create: {output_path}"
            )

        # Restore the spatial resolution if DiffuEraser resized it.
        ensure_output_resolution(
            str(output_path),
            input_info["width"],
            input_info["height"]
        )

        elapsed = time.time() - start_time

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Finished in {elapsed:.4f} s")

        return {
            "input_video": input_video,
            "input_mask": input_mask,
            "priori_path": str(priori_path),
            "output_path": str(output_path),
            "inference_time": elapsed
        }

    def process_batch(
        self,
        input_videos,
        boxes,
        save_path
    ):
        if not input_videos:
            raise ValueError("No videos were uploaded.")

        if len(boxes) != len(input_videos):
            raise ValueError(
                "The number of bounding boxes must equal "
                "the number of videos."
            )

        session_id = uuid.uuid4().hex[:10]
        session_dir = Path(save_path) / f"session_{session_id}"
        session_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        results = []
        total_start = time.time()

        for i, input_video in enumerate(input_videos):
            print(
                f"\n[{i + 1}/{len(input_videos)}] "
                f"Processing video"
            )

            try:
                bbox = boxes[i]
                if bbox is None:
                    raise ValueError(
                        f"No bounding box selected for "
                        f"{Path(input_video).name}"
                    )

                info = read_video_info(input_video)
                bbox = clamp_bbox(
                    bbox,
                    info["width"],
                    info["height"]
                )

                video_name = safe_filename(input_video)
                output_dir = session_dir / video_name
                output_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                mask_path = (
                    output_dir /
                    f"{video_name}_bbox_mask.avi"
                )

                create_mask_video(
                    input_video,
                    bbox,
                    str(mask_path)
                )

                verify_mask(
                    str(mask_path),
                    info
                )

                result = self.process_one(
                    input_video=input_video,
                    input_mask=str(mask_path),
                    output_dir=str(output_dir)
                )

                result["bbox"] = bbox
                result["mask_path"] = str(mask_path)
                results.append(result)

            except Exception as e:
                traceback.print_exc()
                results.append({
                    "input_video": input_video,
                    "error": str(e)
                })

        total_time = time.time() - total_start

        print("\n" + "=" * 70)
        print(
            f"Batch finished: {len(input_videos)} videos "
            f"in {total_time:.2f} seconds"
        )
        print("=" * 70)

        return results


def create_initial_state(files):
    paths = normalize_files(files)

    if not paths:
        return {
            "originals": [],
            "current": [],
            "boxes": [],
            "trims": []
        }

    trims = []
    for path in paths:
        info = read_video_info(path)
        trims.append((0.0, info["duration"]))

    return {
        "originals": paths,
        "current": paths.copy(),
        "boxes": [None] * len(paths),
        "trims": trims
    }


def upload_videos(files):
    state = create_initial_state(files)
    paths = state["current"]

    if not paths:
        return (
            state,
            gr.update(choices=[], value=None),
            None,
            None,
            None,
            None,
            gr.update(minimum=0, maximum=1, value=0),
            gr.update(minimum=1, maximum=1, value=1),
            "No video selected.",
            "No bounding box selected."
        )

    choices = [
        (f"{i + 1}. {Path(p).name}", i)
        for i, p in enumerate(paths)
    ]

    path = paths[0]
    info = read_video_info(path)
    annotation = make_annotation(path)
    trim_end = info["duration"]

    video_info = (
        f"Video 1/{len(paths)}\n"
        f"Name: {Path(path).name}\n"
        f"Resolution: {info['width']} × {info['height']}\n"
        f"FPS: {info['fps']:.6f}\n"
        f"Frames: {info['n_frames']}\n"
        f"Duration: {info['duration']:.2f} s"
    )

    return (
        state,
        gr.update(
            choices=choices,
            value=0
        ),
        path,
        annotation,
        video_info,
        "No bounding box selected.",
        gr.update(
            minimum=0,
            maximum=max(info["duration"], 0.01),
            value=0
        ),
        gr.update(
            minimum=0,
            maximum=max(info["duration"], 0.01),
            value=trim_end
        ),
        f"Loaded {len(paths)} video(s).",
        bbox_text(None)
    )


def select_video(
    video_index,
    state
):
    if video_index is None or not state["current"]:
        return (
            None,
            None,
            "No video selected.",
            "No bounding box selected.",
            gr.update(minimum=0, maximum=1, value=0),
            gr.update(minimum=1, maximum=1, value=1)
        )

    i = int(video_index)
    path = state["current"][i]
    info = read_video_info(path)
    bbox = state["boxes"][i]
    start_time, end_time = state["trims"][i]

    annotation = make_annotation(
        path,
        bbox
    )

    video_info = (
        f"Video {i + 1}/{len(state['current'])}\n"
        f"Name: {Path(state['originals'][i]).name}\n"
        f"Working clip: {Path(path).name}\n"
        f"Resolution: {info['width']} × {info['height']}\n"
        f"FPS: {info['fps']:.6f}\n"
        f"Frames: {info['n_frames']}\n"
        f"Duration: {info['duration']:.2f} s\n"
        f"Selected range: {start_time:.2f} → {end_time:.2f} s"
    )

    return (
        path,
        annotation,
        video_info,
        bbox_text(bbox),
        gr.update(
            minimum=0,
            maximum=max(info["duration"], 0.01),
            value=start_time
        ),
        gr.update(
            minimum=0,
            maximum=max(info["duration"], 0.01),
            value=end_time
        )
    )


def save_bbox(
    video_index,
    annotation,
    state
):
    if video_index is None:
        return state, "No video selected."

    i = int(video_index)
    bbox = get_bbox(annotation)

    state["boxes"][i] = bbox

    return state, bbox_text(bbox)


def update_mask_preview(
    video_index,
    annotation,
    state
):
    if video_index is None or not state["current"]:
        return None

    bbox = get_bbox(annotation)
    if bbox is None:
        return None

    path = state["current"][int(video_index)]

    try:
        return make_preview(
            path,
            bbox
        )
    except Exception:
        return None


def apply_cut(
    video_index,
    start_time,
    end_time,
    state,
    save_path
):
    if video_index is None:
        raise gr.Error("Select a video first.")

    i = int(video_index)
    original = state["originals"][i]

    info = read_video_info(original)

    start_time = float(start_time)
    end_time = float(end_time)

    if start_time < 0:
        raise gr.Error("Start time cannot be negative.")

    if end_time > info["duration"]:
        end_time = info["duration"]

    if end_time <= start_time:
        raise gr.Error(
            "End time must be greater than start time."
        )

    session_id = uuid.uuid4().hex[:10]
    cut_dir = (
        Path(save_path) /
        f"session_{session_id}" /
        "clips"
    )
    cut_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    name = safe_filename(original)

    clipped_path = (
        cut_dir /
        f"{name}_{start_time:.2f}_{end_time:.2f}.mp4"
    )

    try:
        trim_video(
            original,
            start_time,
            end_time,
            str(clipped_path)
        )
    except Exception as e:
        raise gr.Error(str(e))

    state["current"][i] = str(clipped_path)
    state["trims"][i] = (
        start_time,
        end_time
    )

    # The first frame changed, so the old box is invalid.
    state["boxes"][i] = None

    clipped_info = read_video_info(
        str(clipped_path)
    )

    annotation = make_annotation(
        str(clipped_path)
    )

    video_info = (
        f"Video {i + 1}/{len(state['current'])}\n"
        f"Original: {Path(original).name}\n"
        f"Working clip: {Path(clipped_path).name}\n"
        f"Resolution: {clipped_info['width']} × "
        f"{clipped_info['height']}\n"
        f"FPS: {clipped_info['fps']:.6f}\n"
        f"Frames: {clipped_info['n_frames']}\n"
        f"Duration: {clipped_info['duration']:.2f} s\n"
        f"Selected range: {start_time:.2f} → "
        f"{end_time:.2f} s"
    )

    return (
        state,
        str(clipped_path),
        annotation,
        video_info,
        bbox_text(None),
        None,
        "Cut applied. Draw a new bounding box for this clip."
    )


def reset_cut(
    video_index,
    state
):
    if video_index is None:
        raise gr.Error("Select a video first.")

    i = int(video_index)
    original = state["originals"][i]
    info = read_video_info(original)

    state["current"][i] = original
    state["trims"][i] = (
        0.0,
        info["duration"]
    )
    state["boxes"][i] = None

    annotation = make_annotation(original)

    video_info = (
        f"Video {i + 1}/{len(state['current'])}\n"
        f"Name: {Path(original).name}\n"
        f"Resolution: {info['width']} × {info['height']}\n"
        f"FPS: {info['fps']:.6f}\n"
        f"Frames: {info['n_frames']}\n"
        f"Duration: {info['duration']:.2f} s\n"
        f"Selected range: 0.00 → {info['duration']:.2f} s"
    )

    return (
        state,
        original,
        annotation,
        video_info,
        bbox_text(None),
        None,
        gr.update(
            minimum=0,
            maximum=max(info["duration"], 0.01),
            value=0
        ),
        gr.update(
            minimum=0,
            maximum=max(info["duration"], 0.01),
            value=info["duration"]
        ),
        "Cut reset. Original video restored."
    )


def run_gui(
    state,
    engine,
    save_path
):
    if not state or not state["current"]:
        raise gr.Error(
            "Upload at least one video."
        )

    missing = []

    for i, bbox in enumerate(state["boxes"]):
        if bbox is None:
            missing.append(
                Path(state["originals"][i]).name
            )

    if missing:
        raise gr.Error(
            "Please draw a bounding box for every video:\n"
            + "\n".join(missing)
        )

    results = engine.process_batch(
        input_videos=state["current"],
        boxes=state["boxes"],
        save_path=save_path
    )

    successful = [
        r for r in results
        if r.get("output_path")
        and os.path.exists(r["output_path"])
    ]

    files = [
        r["output_path"]
        for r in successful
    ]

    first_result = (
        files[0]
        if files
        else None
    )

    lines = [
        "PROCESSING COMPLETE",
        "",
        f"Videos: {len(state['current'])}",
        f"Successful: {len(successful)}",
        f"Failed: {len(results) - len(successful)}",
        ""
    ]

    for r in results:
        name = Path(
            r["input_video"]
        ).name

        if r.get("output_path"):
            lines.extend([
                f"[OK] {name}",
                f"     bbox: {r['bbox']}",
                f"     mask: {r['mask_path']}",
                f"     output: {r['output_path']}",
                ""
            ])
        else:
            lines.extend([
                f"[FAILED] {name}",
                f"     {r.get('error', 'Unknown error')}",
                ""
            ])

    return (
        files,
        first_result,
        "\n".join(lines)
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="GUI for ProPainter + DiffuEraser video object removal."
    )

    parser.add_argument(
        "--video_length",
        type=int,
        default=10
    )
    parser.add_argument(
        "--mask_dilation_iter",
        type=int,
        default=8
    )
    parser.add_argument(
        "--max_img_size",
        type=int,
        default=960
    )
    parser.add_argument(
        "--ref_stride",
        type=int,
        default=10
    )
    parser.add_argument(
        "--neighbor_length",
        type=int,
        default=10
    )
    parser.add_argument(
        "--subvideo_length",
        type=int,
        default=50
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="2-Step"
    )

    parser.add_argument(
        "--base_model_path",
        type=str,
        default=f"{WEIGHTS}/stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=f"{WEIGHTS}/sd-vae-ft-mse"
    )
    parser.add_argument(
        "--diffueraser_path",
        type=str,
        default=f"{WEIGHTS}/diffuEraser"
    )
    parser.add_argument(
        "--propainter_model_dir",
        type=str,
        default=f"{WEIGHTS}/propainter"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./outputs"
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=8000
    )
    parser.add_argument(
        "--no_share",
        action="store_false"
    )

    return parser


def build_demo(engine, save_path):
    css = """
    .gradio-container {
        max-width: 1350px !important;
        margin: 0 auto !important;
        background: #f6f8fb !important;
    }
    .hero {
        text-align: center;
        padding: 28px 20px 20px 20px;
        margin-bottom: 18px;
        border-radius: 18px;
        background: linear-gradient(135deg,#111827,#263449);
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,.12);
    }
    .hero h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 750;
    }
    .hero p {
        margin: 10px 0 0 0;
        color: #cbd5e1;
        font-size: 16px;
    }
    .section {
        border-radius: 16px;
        padding: 18px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(0,0,0,.05);
        margin-bottom: 16px;
    }
    .step {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    #remove-btn {
        min-height: 58px !important;
        font-size: 19px !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
    }
    .mono textarea {
        font-family: monospace !important;
    }
    footer {
        display: none !important;
    }
    """

    with gr.Blocks(
        title="Video Object Remover",
        theme=gr.themes.Soft(),
        css=css
    ) as demo:
        session_state = gr.State({
            "originals": [],
            "current": [],
            "boxes": [],
            "trims": []
        })

        gr.HTML("""
        <div class="hero">
            <h1>Video Object & Text Remover</h1>
            <p>Trim your video, mark the object, and remove it with ProPainter + DiffuEraser.</p>
        </div>
        """)

        with gr.Column(elem_classes="section"):
            gr.Markdown(
                '<div class="step">1. Select Video Files</div>'
            )
            videos = gr.File(
                label="Upload one or more videos",
                file_count="multiple",
                file_types=[
                    ".mp4",
                    ".mov",
                    ".avi",
                    ".mkv",
                    ".webm"
                ],
                type="filepath"
            )
            video_selector = gr.Dropdown(
                label="Current Video",
                choices=[],
                value=None
            )

        with gr.Column(elem_classes="section"):
            gr.Markdown(
                '<div class="step">2. Trim the Current Video</div>'
            )

            current_video = gr.Video(
                label="Current Video",
                interactive=False,
                height=420
            )

            video_info = gr.Textbox(
                label="Video Information",
                lines=7,
                interactive=False,
                elem_classes="mono"
            )

            with gr.Row():
                start_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.01,
                    label="Start Time (seconds)"
                )
                end_slider = gr.Slider(
                    minimum=1,
                    maximum=1,
                    value=1,
                    step=0.01,
                    label="End Time (seconds)"
                )

            with gr.Row():
                cut_button = gr.Button(
                    "✂ Apply Cut",
                    variant="primary"
                )
                reset_cut_button = gr.Button(
                    "↩ Reset Cut"
                )

            trim_status = gr.Textbox(
                label="Trim Status",
                interactive=False
            )

        with gr.Column(elem_classes="section"):
            gr.Markdown(
                '<div class="step">3. Mark the Object / Text to Remove</div>'
            )

            gr.Markdown(
                "Draw **one box** around the object or text. "
                "Every video has its own independent box."
            )

            annotation = image_annotator(
                value=None,
                label="First Frame of Current Video",
                single_box=True,
                disable_edit_boxes=True,
                box_min_size=5,
                box_thickness=3,
                box_selected_thickness=4,
                height=600,
                width=1000,
                show_clear_button=True,
                show_remove_button=True
            )

            bbox_coordinates = gr.Textbox(
                label="Bounding Box",
                lines=6,
                interactive=False,
                elem_classes="mono"
            )

            mask_preview = gr.Image(
                label="Mask Preview — Red Area Will Be Removed",
                interactive=False,
                height=420
            )

        with gr.Column(elem_classes="section"):
            gr.Markdown(
                '<div class="step">4. Run Video Removal</div>'
            )

            gr.Markdown(
                "Make sure every uploaded video has a bounding box "
                "before starting inference."
            )

            remove_button = gr.Button(
                "🚀 Remove Objects",
                variant="primary",
                elem_id="remove-btn"
            )

            status = gr.Textbox(
                label="Processing Status",
                lines=16,
                interactive=False,
                elem_classes="mono"
            )

        with gr.Column(elem_classes="section"):
            gr.Markdown(
                '<div class="step">5. Results</div>'
            )

            result_video = gr.Video(
                label="First Result",
                height=420
            )

            result_files = gr.Files(
                label="All Result Videos"
            )

        videos.change(
            fn=upload_videos,
            inputs=videos,
            outputs=[
                session_state,
                video_selector,
                current_video,
                annotation,
                video_info,
                bbox_coordinates,
                start_slider,
                end_slider,
                trim_status,
                bbox_coordinates
            ]
        )

        video_selector.change(
            fn=select_video,
            inputs=[
                video_selector,
                session_state
            ],
            outputs=[
                current_video,
                annotation,
                video_info,
                bbox_coordinates,
                start_slider,
                end_slider
            ]
        )

        annotation.change(
            fn=save_bbox,
            inputs=[
                video_selector,
                annotation,
                session_state
            ],
            outputs=[
                session_state,
                bbox_coordinates
            ]
        )

        annotation.change(
            fn=update_mask_preview,
            inputs=[
                video_selector,
                annotation,
                session_state
            ],
            outputs=mask_preview
        )

        cut_button.click(
            fn=lambda idx, start, end, state: apply_cut(
                idx,
                start,
                end,
                state,
                save_path
            ),
            inputs=[
                video_selector,
                start_slider,
                end_slider,
                session_state
            ],
            outputs=[
                session_state,
                current_video,
                annotation,
                video_info,
                bbox_coordinates,
                mask_preview,
                trim_status
            ]
        )

        reset_cut_button.click(
            fn=reset_cut,
            inputs=[
                video_selector,
                session_state
            ],
            outputs=[
                session_state,
                current_video,
                annotation,
                video_info,
                bbox_coordinates,
                mask_preview,
                start_slider,
                end_slider,
                trim_status
            ]
        )

        remove_button.click(
            fn=lambda state: run_gui(
                state,
                engine,
                save_path
            ),
            inputs=session_state,
            outputs=[
                result_files,
                result_video,
                status
            ]
        )

    return demo


def main():
    parser = build_parser()
    args = parser.parse_args()

    Path(args.save_path).mkdir(
        parents=True,
        exist_ok=True
    )

    require_ffmpeg()

    engine = VideoRemovalEngine(args)

    demo = build_demo(
        engine=engine,
        save_path=args.save_path
    )

    demo.queue()

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.no_share,
        show_error=True
    )


if __name__ == "__main__":
    main()