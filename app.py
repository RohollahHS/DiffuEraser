import os
import re
import time
import argparse
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import gradio as gr
from gradio_image_annotation import image_annotator

from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device


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
    return [get_path(f) for f in files if get_path(f)]


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
    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))
    xmax = max(1, min(xmax, width))
    ymax = max(1, min(ymax, height))

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid bounding box.")

    return xmin, ymin, xmax, ymax


def format_bbox(bbox):
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
    value = {
        "image": info["first_frame"],
        "boxes": []
    }

    if bbox is not None:
        bbox = clamp_bbox(bbox, info["width"], info["height"])
        xmin, ymin, xmax, ymax = bbox
        value["boxes"] = [{
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        }]

    return value


def make_preview(video_path, bbox):
    if not video_path or bbox is None:
        return None

    info = read_video_info(video_path)
    bbox = clamp_bbox(bbox, info["width"], info["height"])

    frame = info["first_frame"].copy()
    xmin, ymin, xmax, ymax = bbox

    overlay = frame.copy()
    overlay[ymin:ymax, xmin:xmax] = [255, 0, 0]
    preview = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)

    cv2.rectangle(
        preview,
        (xmin, ymin),
        (xmax - 1, ymax - 1),
        (255, 255, 0),
        3
    )

    return preview


def create_mask_video(input_video, bbox, output_mask):
    info = read_video_info(input_video)
    width = info["width"]
    height = info["height"]
    fps = info["fps"]
    bbox = clamp_bbox(bbox, width, height)
    xmin, ymin, xmax, ymax = bbox

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

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
            "Install/use an OpenCV build with FFmpeg + FFV1 support."
        )

    frame_count = 0

    while True:
        ret, _ = cap.read()
        if not ret:
            break

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[ymin:ymax, xmin:xmax] = 255
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        writer.write(mask_bgr)
        frame_count += 1

    cap.release()
    writer.release()

    if frame_count == 0:
        raise RuntimeError(f"Empty mask generated: {output_mask}")

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frame_count
    }


def verify_mask(mask_path, video_info):
    cap = cv2.VideoCapture(mask_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open generated mask: {mask_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    actual_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        actual_frames += 1

    cap.release()

    if width != video_info["width"] or height != video_info["height"]:
        raise RuntimeError(
            f"Mask resolution mismatch: "
            f"{width}x{height} vs "
            f"{video_info['width']}x{video_info['height']}"
        )

    if abs(fps - video_info["fps"]) > 1e-3:
        raise RuntimeError(
            f"Mask FPS mismatch: {fps} vs {video_info['fps']}"
        )

    if actual_frames != video_info["n_frames"]:
        raise RuntimeError(
            f"Mask frame count mismatch: "
            f"{actual_frames} vs {video_info['n_frames']}"
        )


class VideoRemovalEngine:
    def __init__(self, args):
        self.args = args
        self.device = get_device()

        print("=" * 70)
        print("Initializing models")
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

        video_name = safe_filename(input_video)
        priori_path = output_dir / f"{video_name}_priori.mp4"
        output_path = output_dir / f"{video_name}_diffueraser_result.mp4"

        print("\n" + "=" * 70)
        print(f"Processing: {input_video}")
        print(f"Mask     : {input_mask}")
        print(f"Prior    : {priori_path}")
        print(f"Output   : {output_path}")
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

        inference_time = time.time() - start_time

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Finished in {inference_time:.4f} s")

        return {
            "input_video": input_video,
            "input_mask": input_mask,
            "priori_path": str(priori_path),
            "output_path": str(output_path),
            "inference_time": inference_time
        }

    def process_batch(self, input_videos, boxes, save_path):
        if not input_videos:
            raise ValueError("No input videos.")

        if len(boxes) != len(input_videos):
            raise ValueError(
                f"Number of boxes ({len(boxes)}) must equal "
                f"number of videos ({len(input_videos)})."
            )

        results = []
        total_start = time.time()

        for i, input_video in enumerate(input_videos):
            bbox = boxes[i]
            if bbox is None:
                results.append({
                    "input_video": input_video,
                    "error": "No bounding box was selected."
                })
                continue

            try:
                info = read_video_info(input_video)
                bbox = clamp_bbox(
                    bbox,
                    info["width"],
                    info["height"]
                )

                video_name = safe_filename(input_video)
                output_dir = Path(save_path) / video_name
                output_dir.mkdir(parents=True, exist_ok=True)

                mask_path = output_dir / f"{video_name}_bbox_mask.avi"

                print(
                    f"\n[{i + 1}/{len(input_videos)}] "
                    f"Generating mask for {video_name}"
                )
                print(f"Bounding box: {bbox}")

                create_mask_video(
                    input_video,
                    bbox,
                    mask_path
                )

                verify_mask(
                    str(mask_path),
                    info
                )

                result = self.process_one(
                    input_video=input_video,
                    input_mask=str(mask_path),
                    output_dir=output_dir
                )

                result["bbox"] = bbox
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
            f"Batch completed: {len(input_videos)} videos "
            f"in {total_time:.4f} s"
        )
        print("=" * 70)

        return results


def update_video_list(files):
    paths = normalize_files(files)

    if not paths:
        return [], [], None, "", "", {}

    choices = [
        (f"{i + 1}. {Path(path).name}", i)
        for i, path in enumerate(paths)
    ]

    boxes = {str(i): None for i in range(len(paths))}

    first_info = read_video_info(paths[0])

    info_text = (
        f"Selected videos: {len(paths)}\n"
        f"Current video: {Path(paths[0]).name}\n"
        f"Resolution: {first_info['width']} × {first_info['height']}\n"
        f"FPS: {first_info['fps']:.6f}\n"
        f"Frames: {first_info['n_frames']}\n"
        f"Duration: {first_info['duration']:.2f} s"
    )

    annotation = make_annotation(paths[0])

    return (
        paths,
        choices,
        0,
        info_text,
        "No bounding box selected.",
        boxes,
        annotation
    )


def select_video(video_index, video_paths, boxes):
    if video_index is None or not video_paths:
        return None, "", "No video selected.", boxes

    video_index = int(video_index)
    video_path = video_paths[video_index]
    bbox = boxes.get(str(video_index))

    info = read_video_info(video_path)

    info_text = (
        f"Video {video_index + 1}/{len(video_paths)}\n"
        f"Name: {Path(video_path).name}\n"
        f"Resolution: {info['width']} × {info['height']}\n"
        f"FPS: {info['fps']:.6f}\n"
        f"Frames: {info['n_frames']}\n"
        f"Duration: {info['duration']:.2f} s"
    )

    annotation = make_annotation(
        video_path,
        bbox
    )

    return (
        annotation,
        info_text,
        format_bbox(bbox),
        boxes
    )


def save_current_bbox(video_index, annotation, boxes):
    if video_index is None:
        return boxes, "No video selected."

    bbox = get_bbox(annotation)
    if bbox is None:
        boxes[str(video_index)] = None
        return boxes, "No bounding box selected."

    boxes[str(video_index)] = bbox
    return boxes, format_bbox(bbox)


def update_preview(video_index, annotation, video_paths):
    if video_index is None or not video_paths:
        return None

    bbox = get_bbox(annotation)
    if bbox is None:
        return None

    return make_preview(
        video_paths[int(video_index)],
        bbox
    )


def run_gui(input_videos, boxes, engine, save_path):
    if not input_videos:
        raise gr.Error("Upload at least one video.")

    if not boxes:
        raise gr.Error("No bounding boxes have been created.")

    bbox_list = []

    for i in range(len(input_videos)):
        bbox = boxes.get(str(i))
        if bbox is None:
            raise gr.Error(
                f"Please draw a bounding box for video "
                f"{i + 1}: {Path(input_videos[i]).name}"
            )
        bbox_list.append(tuple(bbox))

    results = engine.process_batch(
        input_videos=input_videos,
        boxes=bbox_list,
        save_path=save_path
    )

    successful = [
        r for r in results
        if r.get("output_path") and Path(r["output_path"]).exists()
    ]

    failed = [
        r for r in results
        if r.get("error")
    ]

    result_files = [
        r["output_path"]
        for r in successful
    ]

    first_result = (
        result_files[0]
        if result_files
        else None
    )

    lines = [
        "Processing complete.",
        f"Videos: {len(input_videos)}",
        f"Successful: {len(successful)}",
        f"Failed: {len(failed)}",
        ""
    ]

    for result in results:
        name = Path(result["input_video"]).name

        if result.get("output_path"):
            lines.append(f"[OK] {name}")
            lines.append(
                f"     bbox = {result['bbox']}"
            )
            lines.append(
                f"     output = {result['output_path']}"
            )
        else:
            lines.append(
                f"[FAILED] {name}: {result.get('error')}"
            )

    return (
        result_files,
        first_result,
        "\n".join(lines)
    )


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_video",
        type=str,
        nargs="+",
        default=None,
        help="Optional initial video list."
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
        default=f"weights/stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=f"weights/sd-vae-ft-mse"
    )
    parser.add_argument(
        "--diffueraser_path",
        type=str,
        default=f"weights/diffuEraser"
    )
    parser.add_argument(
        "--propainter_model_dir",
        type=str,
        default=f"weights/propainter"
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
        "--share",
        action="store_true"
    )

    return parser


def build_demo(engine, save_path):
    title = """
    <div style="text-align:center;font-size:34px;font-family:Arial,sans-serif;font-weight:bold;">
        Video Object / Text Remover
    </div>
    <div style="text-align:center;font-size:16px;color:#666;margin:10px 0 20px;">
        Upload multiple videos and draw a separate removal box for each video.
    </div>
    """

    instructions = """
    ### Workflow
    **1. Upload videos** → **2. Select a video** → **3. Draw its box** → **4. Select the next video** → **5. Draw its box** → **6. Remove Objects**

    Each video has its own bounding box and its own automatically generated mask.
    """

    css = """
    #main {max-width:1200px;margin:auto;}
    #remove_btn {width:60%;margin:15px auto;display:block;font-size:20px;}
    .mono textarea {font-family:monospace !important;}
    footer {display:none !important;}
    """

    with gr.Blocks(
        title="Video Object Remover",
        theme=gr.themes.Soft(),
        css=css
    ) as demo:
        video_state = gr.State([])
        boxes_state = gr.State({})

        gr.HTML(title)
        gr.Markdown(instructions)

        videos = gr.File(
            label="1. Upload Video(s)",
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
            label="2. Select Video to Annotate",
            choices=[],
            type="value",
            value=None
        )

        video_info = gr.Textbox(
            label="Video Information",
            lines=6,
            interactive=False,
            elem_classes="mono"
        )

        annotation = image_annotator(
            value=None,
            label="3. Draw Bounding Box",
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

        bbox_text = gr.Textbox(
            label="Current Bounding Box",
            lines=6,
            interactive=False,
            elem_classes="mono"
        )

        mask_preview = gr.Image(
            label="Mask Preview — Red Area Will Be Removed",
            interactive=False,
            height=450
        )

        gr.Markdown(
            "Draw a box for **every video** before clicking Remove Objects."
        )

        remove_btn = gr.Button(
            "Remove Objects",
            variant="primary",
            size="lg",
            elem_id="remove_btn"
        )

        status = gr.Textbox(
            label="Processing Status",
            lines=15,
            interactive=False,
            elem_classes="mono"
        )

        result_video = gr.Video(
            label="First Result"
        )

        result_files = gr.Files(
            label="All Results"
        )

        videos.change(
            fn=update_video_list,
            inputs=videos,
            outputs=[
                video_state,
                video_selector,
                video_selector,
                video_info,
                bbox_text,
                boxes_state,
                annotation
            ]
        )

        video_selector.change(
            fn=select_video,
            inputs=[
                video_selector,
                video_state,
                boxes_state
            ],
            outputs=[
                annotation,
                video_info,
                bbox_text,
                boxes_state
            ]
        )

        annotation.change(
            fn=save_current_bbox,
            inputs=[
                video_selector,
                annotation,
                boxes_state
            ],
            outputs=[
                boxes_state,
                bbox_text
            ]
        )

        annotation.change(
            fn=update_preview,
            inputs=[
                video_selector,
                annotation,
                video_state
            ],
            outputs=mask_preview
        )

        remove_btn.click(
            fn=lambda input_videos, boxes: run_gui(
                input_videos,
                boxes,
                engine,
                save_path
            ),
            inputs=[
                video_state,
                boxes_state
            ],
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

    os.makedirs(
        args.save_path,
        exist_ok=True
    )

    engine = VideoRemovalEngine(args)

    demo = build_demo(
        engine=engine,
        save_path=args.save_path
    )

    demo.queue()

    initial_files = args.input_video

    if initial_files:
        # These files are only used as the initial GUI video list.
        # The masks still MUST be created through the GUI.
        initial_files = [
            str(Path(x).resolve())
            for x in initial_files
        ]

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()