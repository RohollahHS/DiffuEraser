import os
import re
import time
import uuid
import shutil
import argparse
import subprocess
import traceback
import json
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch
import gradio as gr
from gradio_image_annotation import image_annotator

from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device

WEIGHTS = os.getenv("HF_HUB", "weights")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}

def safe_filename(path):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(path).stem)

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
    return [p for f in files if (p := get_path(f))]

def require_ffmpeg():
    missing = [x for x in ("ffmpeg", "ffprobe") if shutil.which(x) is None]
    if missing:
        raise RuntimeError("Missing required executable(s): " + ", ".join(missing) + ". Install FFmpeg and make sure they are in PATH.")

def run_command(cmd, label, timeout):
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{label} timed out after {timeout} seconds.")
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n\n{(result.stderr or '')[-5000:]}")
    return result

def read_video_info(video_path):
    require_ffmpeg()
    video_path = str(video_path)
    if not os.path.isfile(video_path):
        raise RuntimeError(f"Video does not exist: {video_path}")
    cmd = [
        "ffprobe", "-hide_banner", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", video_path
    ]
    result = run_command(cmd, "ffprobe", 20)
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Could not parse ffprobe output for: {video_path}") from e
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    try:
        fps = float(Fraction(stream.get("r_frame_rate") or "0/1"))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    try:
        duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    try:
        n_frames = int(stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        n_frames = 0
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS: {video_path}")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid resolution: {video_path}")
    if duration <= 0:
        raise RuntimeError(f"Invalid duration: {video_path}")
    frame_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", "0", "-i", video_path, "-frames:v", "1",
        "-f", "image2pipe", "-vcodec", "png", "pipe:1"
    ]
    try:
        frame_result = subprocess.run(frame_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timed out while extracting the first frame: {video_path}")
    if frame_result.returncode != 0:
        stderr = frame_result.stderr.decode("utf-8", errors="replace") if isinstance(frame_result.stderr, bytes) else (frame_result.stderr or "")
        raise RuntimeError(f"Could not read first frame:\n{stderr[-5000:]}")
    frame = cv2.imdecode(np.frombuffer(frame_result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode first frame: {video_path}")
    return {
        "first_frame": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        "fps": fps,
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "duration": duration
    }

def get_bbox(annotation):
    if not isinstance(annotation, dict):
        return None
    boxes = annotation.get("boxes") or []
    if not boxes:
        return None
    box = boxes[0]
    try:
        return (
            int(round(float(box["xmin"]))),
            int(round(float(box["ymin"]))),
            int(round(float(box["xmax"]))),
            int(round(float(box["ymax"])))
        )
    except (KeyError, TypeError, ValueError):
        return None

def clamp_bbox(bbox, width, height):
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, min(int(xmin), width - 1))
    ymin = max(0, min(int(ymin), height - 1))
    xmax = max(1, min(int(xmax), width))
    ymax = max(1, min(int(ymax), height))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid bounding box {bbox} for {width}x{height} video.")
    return xmin, ymin, xmax, ymax

def bbox_text(bbox):
    if bbox is None:
        return "No bounding box selected."
    xmin, ymin, xmax, ymax = bbox
    return f"xmin = {xmin}\nymin = {ymin}\nxmax = {xmax}\nymax = {ymax}\nwidth = {xmax - xmin}\nheight = {ymax - ymin}"

def make_annotation(video_path, bbox=None):
    info = read_video_info(video_path)
    data = {"image": info["first_frame"], "boxes": []}
    if bbox is not None:
        xmin, ymin, xmax, ymax = clamp_bbox(bbox, info["width"], info["height"])
        data["boxes"] = [{"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}]
    return data

def make_preview(video_path, bbox):
    if not video_path or bbox is None:
        return None
    info = read_video_info(video_path)
    xmin, ymin, xmax, ymax = clamp_bbox(bbox, info["width"], info["height"])
    frame = info["first_frame"].copy()
    overlay = frame.copy()
    overlay[ymin:ymax, xmin:xmax] = [255, 0, 0]
    preview = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)
    cv2.rectangle(preview, (xmin, ymin), (xmax - 1, ymax - 1), (255, 255, 0), 3)
    return preview

def trim_video(input_video, start_time, end_time, output_path):
    require_ffmpeg()
    info = read_video_info(input_video)
    duration = info["duration"]
    start_time = max(0.0, float(start_time))
    end_time = min(float(end_time), duration)
    if end_time <= start_time:
        raise ValueError(f"End time ({end_time:.3f}) must be greater than start time ({start_time:.3f}).")
    if abs(start_time) < 1e-6 and abs(end_time - duration) < 1.0 / max(info["fps"], 1.0):
        shutil.copy2(input_video, output_path)
        return output_path
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{start_time:.6f}", "-i", input_video,
        "-t", f"{end_time - start_time:.6f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
        output_path
    ]
    run_command(cmd, "FFmpeg video trimming", 1800)
    if not os.path.isfile(output_path):
        raise RuntimeError(f"FFmpeg completed but output was not created: {output_path}")
    return output_path

def create_mask_video(input_video, bbox, output_mask):
    info = read_video_info(input_video)
    width, height, fps = info["width"], info["height"], info["fps"]
    xmin, ymin, xmax, ymax = clamp_bbox(bbox, width, height)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")
    writer = cv2.VideoWriter(str(output_mask), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height), True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create FFV1 mask video. Your OpenCV/FFmpeg build may not support FFV1.")
    frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[ymin:ymax, xmin:xmax] = 255
        writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        frames += 1
    cap.release()
    writer.release()
    if frames == 0:
        raise RuntimeError(f"No mask frames generated: {output_mask}")
    return {"fps": fps, "width": width, "height": height, "frames": frames}

def verify_mask(mask_path, video_info):
    cap = cv2.VideoCapture(mask_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open generated mask: {mask_path}")
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
        raise RuntimeError(f"Mask width mismatch: {width} vs {video_info['width']}")
    if height != video_info["height"]:
        raise RuntimeError(f"Mask height mismatch: {height} vs {video_info['height']}")
    if abs(fps - video_info["fps"]) > 1e-3:
        raise RuntimeError(f"Mask FPS mismatch: {fps} vs {video_info['fps']}")
    if video_info["n_frames"] > 0 and n_frames != video_info["n_frames"]:
        raise RuntimeError(f"Mask frame count mismatch: {n_frames} vs {video_info['n_frames']}")

def ensure_output_resolution(output_video, target_width, target_height):
    require_ffmpeg()
    info = read_video_info(output_video)
    if info["width"] == target_width and info["height"] == target_height:
        return output_video
    temp_output = str(Path(output_video).with_suffix("")) + "_resized.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", output_video, "-map", "0:v:0", "-map", "0:a?",
        "-vf", f"scale={target_width}:{target_height}:flags=lanczos",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
        temp_output
    ]
    run_command(cmd, "FFmpeg output resizing", 1800)
    if not os.path.isfile(temp_output):
        raise RuntimeError(f"Resized output was not created: {temp_output}")
    os.replace(temp_output, output_video)
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

    def process_one(self, input_video, input_mask, output_dir):
        args = self.args
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
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
            raise RuntimeError(f"ProPainter did not create: {priori_path}")
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
            raise RuntimeError(f"DiffuEraser did not create: {output_path}")
        ensure_output_resolution(str(output_path), input_info["width"], input_info["height"])
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

    def process_batch(self, input_videos, boxes, save_path):
        if not input_videos:
            raise ValueError("No videos were uploaded.")
        if len(boxes) != len(input_videos):
            raise ValueError("The number of bounding boxes must equal the number of videos.")
        session_dir = Path(save_path) / f"session_{uuid.uuid4().hex[:10]}"
        session_dir.mkdir(parents=True, exist_ok=True)
        results = []
        total_start = time.time()
        for i, input_video in enumerate(input_videos):
            print(f"\n[{i + 1}/{len(input_videos)}] Processing video")
            try:
                bbox = boxes[i]
                if bbox is None:
                    raise ValueError(f"No bounding box selected for {Path(input_video).name}")
                info = read_video_info(input_video)
                bbox = clamp_bbox(bbox, info["width"], info["height"])
                video_name = safe_filename(input_video)
                output_dir = session_dir / video_name
                output_dir.mkdir(parents=True, exist_ok=True)
                mask_path = output_dir / f"{video_name}_bbox_mask.avi"
                create_mask_video(input_video, bbox, str(mask_path))
                verify_mask(str(mask_path), info)
                result = self.process_one(input_video, str(mask_path), str(output_dir))
                result["bbox"] = bbox
                result["mask_path"] = str(mask_path)
                results.append(result)
            except Exception as e:
                traceback.print_exc()
                results.append({"input_video": input_video, "error": str(e)})
        total_time = time.time() - total_start
        print("\n" + "=" * 70)
        print(f"Batch finished: {len(input_videos)} videos in {total_time:.2f} seconds")
        print("=" * 70)
        return results

def empty_state():
    return {
        "originals": [],
        "current": [],
        "boxes": [],
        "trims": [],
        "current_index": None
    }

def create_initial_state(files):
    paths = normalize_files(files)
    return {
        "originals": paths,
        "current": paths.copy(),
        "boxes": [None] * len(paths),
        "trims": [None] * len(paths),
        "current_index": 0 if paths else None
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
            "No bounding box selected.",
            gr.update(minimum=0, maximum=1, value=0),
            gr.update(minimum=1, maximum=1, value=1),
            None
        )
    choices = [(f"{i + 1}. {Path(p).name}", i) for i, p in enumerate(paths)]
    path = paths[0]
    try:
        info = read_video_info(path)
    except Exception as e:
        raise gr.Error(str(e))
    state["trims"][0] = (0.0, info["duration"])
    annotation = make_annotation(path)
    video_info = (
        f"Video 1/{len(paths)}\n"
        f"Name: {Path(path).name}\n"
        f"Resolution: {info['width']} × {info['height']}\n"
        f"FPS: {info['fps']:.6f}\n"
        f"Frames: {info['n_frames'] or 'unknown'}\n"
        f"Duration: {info['duration']:.2f} s"
    )
    return (
        state,
        gr.update(choices=choices, value=0),
        path,
        annotation,
        video_info,
        bbox_text(None),
        gr.update(minimum=0, maximum=max(info["duration"], 0.01), value=0),
        gr.update(minimum=0, maximum=max(info["duration"], 0.01), value=info["duration"]),
        "Video loaded. Draw a bounding box."
    )

def select_video(video_index, state):
    if video_index is None or not state["current"]:
        return state, None, None, "No video selected.", "No bounding box selected.", gr.update(minimum=0, maximum=1, value=0), gr.update(minimum=1, maximum=1, value=1)
    i = int(video_index)
    path = state["current"][i]
    try:
        info = read_video_info(path)
    except Exception as e:
        raise gr.Error(str(e))
    state = dict(state)
    state["current_index"] = i
    state["trims"] = state["trims"].copy()
    if state["trims"][i] is None:
        state["trims"][i] = (0.0, info["duration"])
    bbox = state["boxes"][i]
    start_time, end_time = state["trims"][i]
    annotation = make_annotation(path, bbox)
    video_info = (
        f"Video {i + 1}/{len(state['current'])}\n"
        f"Name: {Path(state['originals'][i]).name}\n"
        f"Working clip: {Path(path).name}\n"
        f"Resolution: {info['width']} × {info['height']}\n"
        f"FPS: {info['fps']:.6f}\n"
        f"Frames: {info['n_frames'] or 'unknown'}\n"
        f"Duration: {info['duration']:.2f} s\n"
        f"Selected range: {start_time:.2f} → {end_time:.2f} s"
    )
    preview = make_preview(path, bbox) if bbox else None
    return (
        state,
        path,
        annotation,
        video_info,
        bbox_text(bbox),
        gr.update(minimum=0, maximum=max(info["duration"], 0.01), value=start_time),
        gr.update(minimum=0, maximum=max(info["duration"], 0.01), value=end_time)
    )

def annotation_changed(annotation, state):
    if not state or not state.get("current"):
        return state, "No video selected.", None
    i = state.get("current_index")
    if i is None:
        return state, "No video selected.", None
    path = state["current"][i]
    bbox = get_bbox(annotation)
    state = dict(state)
    state["boxes"] = state["boxes"].copy()
    state["boxes"][i] = bbox
    if bbox is None:
        return state, bbox_text(None), None
    try:
        info = read_video_info(path)
        bbox = clamp_bbox(bbox, info["width"], info["height"])
        state["boxes"][i] = bbox
        preview = make_preview(path, bbox)
        return state, bbox_text(bbox), preview
    except Exception as e:
        return state, f"Invalid bounding box: {e}", None

def apply_cut(video_index, start_time, end_time, state, save_path):
    if video_index is None:
        raise gr.Error("Select a video first.")
    i = int(video_index)
    original = state["originals"][i]
    try:
        info = read_video_info(original)
    except Exception as e:
        raise gr.Error(str(e))
    start_time = max(0.0, float(start_time))
    end_time = min(float(end_time), info["duration"])
    if end_time <= start_time:
        raise gr.Error("End time must be greater than start time.")
    cut_dir = Path(save_path) / f"session_{uuid.uuid4().hex[:10]}" / "clips"
    cut_dir.mkdir(parents=True, exist_ok=True)
    name = safe_filename(original)
    clipped_path = cut_dir / f"{name}_{start_time:.2f}_{end_time:.2f}.mp4"
    try:
        trim_video(original, start_time, end_time, str(clipped_path))
        clipped_info = read_video_info(str(clipped_path))
    except Exception as e:
        raise gr.Error(str(e))
    state = dict(state)
    state["current"] = state["current"].copy()
    state["trims"] = state["trims"].copy()
    state["boxes"] = state["boxes"].copy()
    state["current"][i] = str(clipped_path)
    state["trims"][i] = (start_time, end_time)
    state["boxes"][i] = None
    state["current_index"] = i
    annotation = make_annotation(str(clipped_path))
    video_info = (
        f"Video {i + 1}/{len(state['current'])}\n"
        f"Original: {Path(original).name}\n"
        f"Working clip: {Path(clipped_path).name}\n"
        f"Resolution: {clipped_info['width']} × {clipped_info['height']}\n"
        f"FPS: {clipped_info['fps']:.6f}\n"
        f"Frames: {clipped_info['n_frames'] or 'unknown'}\n"
        f"Duration: {clipped_info['duration']:.2f} s\n"
        f"Selected range: {start_time:.2f} → {end_time:.2f} s"
    )
    return (
        state,
        str(clipped_path),
        annotation,
        video_info,
        bbox_text(None),
        None,
        "Cut applied. Draw a new bounding box."
    )

def reset_cut(video_index, state):
    if video_index is None:
        raise gr.Error("Select a video first.")
    i = int(video_index)
    original = state["originals"][i]
    try:
        info = read_video_info(original)
    except Exception as e:
        raise gr.Error(str(e))
    state = dict(state)
    state["current"] = state["current"].copy()
    state["trims"] = state["trims"].copy()
    state["boxes"] = state["boxes"].copy()
    state["current"][i] = original
    state["trims"][i] = (0.0, info["duration"])
    state["boxes"][i] = None
    state["current_index"] = i
    annotation = make_annotation(original)
    video_info = (
        f"Video {i + 1}/{len(state['current'])}\n"
        f"Name: {Path(original).name}\n"
        f"Resolution: {info['width']} × {info['height']}\n"
        f"FPS: {info['fps']:.6f}\n"
        f"Frames: {info['n_frames'] or 'unknown'}\n"
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
        gr.update(minimum=0, maximum=max(info["duration"], 0.01), value=0),
        gr.update(minimum=0, maximum=max(info["duration"], 0.01), value=info["duration"]),
        "Cut reset. Original video restored."
    )

def load_selected_video(video_path, state):
    if isinstance(video_path, list):
        video_path = video_path[0] if video_path else None
    if not video_path:
        raise gr.Error("Please select a video first.")
    video_path = str(video_path)
    path_obj = Path(video_path)
    if not path_obj.is_file():
        raise gr.Error(f"Video does not exist:\n{video_path}")
    if path_obj.suffix.lower() not in VIDEO_EXTENSIONS:
        raise gr.Error(f"Unsupported video file: {path_obj.name}")
    try:
        info = read_video_info(video_path)
    except Exception as e:
        raise gr.Error(str(e))
    state = {
        "originals": [video_path],
        "current": [video_path],
        "boxes": [None],
        "trims": [(0.0, info["duration"])],
        "current_index": 0
    }
    annotation = make_annotation(video_path)
    video_info = (
        f"Name: {path_obj.name}\n"
        f"Resolution: {info['width']} × {info['height']}\n"
        f"FPS: {info['fps']:.6f}\n"
        f"Frames: {info['n_frames'] or 'unknown'}\n"
        f"Duration: {info['duration']:.2f} s"
    )
    return (
        state,
        gr.update(choices=[(f"1. {path_obj.name}", 0)], value=0),
        video_path,
        annotation,
        video_info,
        bbox_text(None),
        gr.update(minimum=0, maximum=max(info["duration"], 0.01), value=0),
        gr.update(minimum=0, maximum=max(info["duration"], 0.01), value=info["duration"]),
        None,
        "Video loaded successfully. Draw a bounding box."
    )

def run_gui(state, engine, save_path):
    if not state or not state.get("current"):
        raise gr.Error("Upload at least one video.")
    missing = []
    for i, bbox in enumerate(state["boxes"]):
        if bbox is None:
            missing.append(Path(state["originals"][i]).name)
    if missing:
        raise gr.Error("Please draw a bounding box for every video:\n" + "\n".join(missing))
    results = engine.process_batch(state["current"], state["boxes"], save_path)
    successful = [r for r in results if r.get("output_path") and os.path.exists(r["output_path"])]
    files = [r["output_path"] for r in successful]
    first_result = files[0] if files else None
    lines = [
        "PROCESSING COMPLETE",
        "",
        f"Videos: {len(state['current'])}",
        f"Successful: {len(successful)}",
        f"Failed: {len(results) - len(successful)}",
        ""
    ]
    for r in results:
        name = Path(r["input_video"]).name
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
    return files, first_result, "\n".join(lines)

def build_parser():
    parser = argparse.ArgumentParser(description="GUI for ProPainter + DiffuEraser video object removal.")
    parser.add_argument("--video_length", type=int, default=10)
    parser.add_argument("--mask_dilation_iter", type=int, default=8)
    parser.add_argument("--max_img_size", type=int, default=960)
    parser.add_argument("--ref_stride", type=int, default=10)
    parser.add_argument("--neighbor_length", type=int, default=10)
    parser.add_argument("--subvideo_length", type=int, default=50)
    parser.add_argument("--ckpt", type=str, default="2-Step")
    parser.add_argument("--base_model_path", type=str, default=f"{WEIGHTS}/stable-diffusion-v1-5")
    parser.add_argument("--vae_path", type=str, default=f"{WEIGHTS}/sd-vae-ft-mse")
    parser.add_argument("--diffueraser_path", type=str, default=f"{WEIGHTS}/diffuEraser")
    parser.add_argument("--propainter_model_dir", type=str, default=f"{WEIGHTS}/propainter")
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--browse_path", type=str, default="/scratch/rohhs/downloads/yt-dlp")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--share", action="store_true")
    return parser

def build_demo(engine, save_path, browse_path):
    css = """
.gradio-container{max-width:1350px!important;margin:0 auto!important;background:#f6f8fb!important}
.hero{text-align:center;padding:28px 20px 20px;margin-bottom:18px;border-radius:18px;background:linear-gradient(135deg,#111827,#263449);color:white;box-shadow:0 10px 30px rgba(0,0,0,.12)}
.hero h1{margin:0;font-size:34px;font-weight:750}
.hero p{margin:10px 0 0;color:#cbd5e1;font-size:16px}
.section{border-radius:16px;padding:18px;background:white;border:1px solid #e5e7eb;box-shadow:0 5px 18px rgba(0,0,0,.05);margin-bottom:16px}
.step{font-size:18px;font-weight:700;margin-bottom:10px}
#remove-btn{min-height:58px!important;font-size:19px!important;font-weight:700!important;border-radius:12px!important}
.mono textarea{font-family:monospace!important}
footer{display:none!important}
"""

    with gr.Blocks(title="Video Object Remover", theme=gr.themes.Soft(), css=css) as demo:
        session_state = gr.State(empty_state())

        gr.HTML(
            '<div class="hero"><h1>Video Object & Text Remover</h1><p>Trim your video, mark the object, and remove it with ProPainter + DiffuEraser.</p></div>'
        )

        with gr.Column(elem_classes="section"):
            gr.Markdown('<div class="step">1. Select Video Files</div>')

            videos = gr.File(
                label="Upload one or more videos from your computer",
                file_count="multiple",
                file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"],
                type="filepath"
            )

            video_selector = gr.Dropdown(
                label="Uploaded Video",
                choices=[],
                value=None,
                interactive=True
            )

            gr.Markdown("Or browse files on the machine running this Gradio app:")

            server_video = gr.FileExplorer(
                label="Host Filesystem",
                root_dir=str(Path(browse_path).resolve()),
                glob="**/*",
                file_count="single",
                interactive=True
            )

            load_video_button = gr.Button("📂 Load Host File", variant="primary")

        with gr.Column(elem_classes="section"):
            gr.Markdown('<div class="step">2. Trim the Current Video</div>')

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
                cut_button = gr.Button("✂ Apply Cut", variant="primary")
                reset_cut_button = gr.Button("↩ Reset Cut")

            trim_status = gr.Textbox(
                label="Trim Status",
                interactive=False
            )

        with gr.Column(elem_classes="section"):
            gr.Markdown('<div class="step">3. Mark the Object / Text to Remove</div>')
            gr.Markdown("Draw **one box** around the object or text. The box is saved automatically.")

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
            gr.Markdown('<div class="step">4. Run Video Removal</div>')
            gr.Markdown("After the box and mask preview appear, click **🚀 Remove Objects**.")

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
            gr.Markdown('<div class="step">5. Results</div>')

            result_video = gr.Video(
                label="First Result",
                height=420
            )

            result_files = gr.Files(
                label="All Result Videos"
            )

        videos.upload(
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
                trim_status
            ],
            queue=False,
            show_progress="minimal"
        )

        video_selector.change(
            fn=select_video,
            inputs=[video_selector, session_state],
            outputs=[
                session_state,
                current_video,
                annotation,
                video_info,
                bbox_coordinates,
                start_slider,
                end_slider
            ],
            queue=False
        )

        annotation.change(
            fn=annotation_changed,
            inputs=[annotation, session_state],
            outputs=[
                session_state,
                bbox_coordinates,
                mask_preview
            ],
            queue=False,
            show_progress="minimal"
        )

        cut_button.click(
            fn=lambda idx, start, end, state: apply_cut(idx, start, end, state, save_path),
            inputs=[video_selector, start_slider, end_slider, session_state],
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

        load_video_button.click(
            fn=load_selected_video,
            inputs=[server_video, session_state],
            outputs=[
                session_state,
                video_selector,
                current_video,
                annotation,
                video_info,
                bbox_coordinates,
                start_slider,
                end_slider,
                mask_preview,
                trim_status
            ],
            queue=False,
            show_progress="minimal"
        )

        reset_cut_button.click(
            fn=reset_cut,
            inputs=[video_selector, session_state],
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
            fn=lambda state: run_gui(state, engine, save_path),
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
    save_path = Path(args.save_path).expanduser().resolve()
    browse_path = Path(args.browse_path).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    if not browse_path.is_dir():
        raise RuntimeError(f"Browse path does not exist or is not a directory: {browse_path}")
    require_ffmpeg()
    engine = VideoRemovalEngine(args)
    demo = build_demo(
        engine=engine,
        save_path=str(save_path),
        browse_path=str(browse_path)
    )
    demo.queue()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
        allowed_paths=[str(save_path), str(browse_path)]
    )

if __name__ == "__main__":
    main()