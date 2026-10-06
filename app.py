import os
import re
import time
import uuid
import shutil
import argparse
import subprocess
import threading
import traceback
import json
from fractions import Fraction
from pathlib import Path

import cv2
import torch
import gradio as gr

from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device

ENGINE = None
ENGINE_LOCK = threading.Lock()

GUI_CSS = ".gradio-container{max-width:1350px!important;margin:0 auto!important}.hero{text-align:center;padding:24px 20px;margin-bottom:16px;border-radius:18px;background:linear-gradient(135deg,#111827,#263449);color:white}.hero h1{margin:0;font-size:32px}.hero p{margin:8px 0 0;color:#cbd5e1}.section{border-radius:16px;padding:18px;background:white;border:1px solid #e5e7eb;box-shadow:0 5px 18px rgba(0,0,0,.05);margin-bottom:16px}.step{font-size:18px;font-weight:700;margin-bottom:10px}#remove-btn{min-height:58px!important;font-size:19px!important;font-weight:700!important;border-radius:12px!important}.mono textarea{font-family:monospace!important}footer{display:none!important}"


def safe_filename(path):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(path).stem)


def normalize_files(files):
    if files is None:
        return []
    if not isinstance(files, list):
        files = [files]
    paths = []
    for item in files:
        if item is None:
            continue
        if isinstance(item, (str, Path)):
            path = str(item)
        elif hasattr(item, "path") and item.path:
            path = str(item.path)
        elif hasattr(item, "name") and item.name:
            path = str(item.name)
        else:
            raise TypeError(f"Unsupported file type: {type(item)}")
        if path:
            paths.append(path)
    return paths


def require_tools():
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError("Missing required executables: " + ", ".join(missing) + ". Install FFmpeg and make sure it is in PATH.")


def run_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}):\n{' '.join(map(str, cmd))}\n\n{result.stderr[-5000:]}")
    return result


def probe_video(video_path):
    video_path = str(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    result = run_cmd(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", video_path])
    data = json.loads(result.stdout)
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise RuntimeError(f"No video stream found: {video_path}")
    rate_text = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(rate_text))
    except Exception:
        fps = 0.0
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS reported by ffprobe: {video_path}")
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    duration_text = video_stream.get("duration") or data.get("format", {}).get("duration") or "0"
    duration = float(duration_text)
    n_frames_text = video_stream.get("nb_frames")
    try:
        n_frames = int(n_frames_text) if n_frames_text not in (None, "N/A", "") else max(1, int(round(duration * fps)))
    except Exception:
        n_frames = max(1, int(round(duration * fps)))
    if width <= 0 or height <= 0 or duration <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return {"fps": fps, "n_frames": n_frames, "width": width, "height": height, "duration": duration}


def extract_first_frame(video_path, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(output_path)])
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg did not create first-frame image: {output_path}")
    return str(output_path)


def load_frame_image(frame_path):
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read frame image: {frame_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def clamp_bbox(bbox, width, height):
    xmin, ymin, xmax, ymax = [int(round(v)) for v in bbox]
    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))
    xmax = max(1, min(xmax, width))
    ymax = max(1, min(ymax, height))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid bounding box {bbox} for {width}x{height} video.")
    return xmin, ymin, xmax, ymax


def get_bbox_from_points(points, width, height):
    if len(points) < 2:
        return None
    (x1, y1), (x2, y2) = points[-2], points[-1]
    xmin, xmax = sorted((int(round(x1)), int(round(x2))))
    ymin, ymax = sorted((int(round(y1)), int(round(y2))))
    return clamp_bbox((xmin, ymin, xmax, ymax), width, height)


def bbox_text(bbox):
    if bbox is None:
        return "No bounding box selected."
    xmin, ymin, xmax, ymax = bbox
    return f"xmin = {xmin}\nymin = {ymin}\nxmax = {xmax}\nymax = {ymax}\nwidth = {xmax - xmin}\nheight = {ymax - ymin}"


def draw_bbox(frame, bbox=None, point=None):
    output = frame.copy()
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        overlay = output.copy()
        overlay[ymin:ymax, xmin:xmax] = (255, 0, 0)
        output = cv2.addWeighted(output, 0.55, overlay, 0.45, 0)
        cv2.rectangle(output, (xmin, ymin), (xmax - 1, ymax - 1), (255, 255, 0), 3)
    elif point is not None:
        x, y = point
        cv2.circle(output, (x, y), 8, (255, 255, 0), -1)
        cv2.circle(output, (x, y), 12, (0, 0, 0), 2)
    return output


def make_preview(frame_path, bbox):
    if not frame_path or bbox is None:
        return None
    return draw_bbox(load_frame_image(frame_path), bbox)


def trim_video(input_video, start_time, end_time, output_path):
    info = probe_video(input_video)
    start_time = max(0.0, float(start_time))
    end_time = min(float(end_time), info["duration"])
    if end_time <= start_time:
        raise ValueError(f"End time ({end_time:.3f}) must be greater than start time ({start_time:.3f}).")
    if start_time <= 1e-6 and abs(end_time - info["duration"]) < 1.0 / max(info["fps"], 1.0):
        shutil.copy2(input_video, output_path)
        return output_path
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clip_duration = end_time - start_time
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start_time:.6f}", "-i", input_video, "-t", f"{clip_duration:.6f}", "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", output_path]
    run_cmd(cmd)
    if not os.path.exists(output_path):
        raise RuntimeError(f"FFmpeg completed but output was not created: {output_path}")
    return output_path


def create_mask_video(input_video, bbox, output_mask):
    info = probe_video(input_video)
    bbox = clamp_bbox(bbox, info["width"], info["height"])
    xmin, ymin, xmax, ymax = bbox
    fps_text = f"{info['fps']:.12g}"
    output_mask = str(output_mask)
    Path(output_mask).parent.mkdir(parents=True, exist_ok=True)
    drawbox = f"drawbox=x={xmin}:y={ymin}:w={xmax - xmin}:h={ymax - ymin}:color=white:t=fill"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s={info['width']}x{info['height']}:r={fps_text}", "-frames:v", str(info["n_frames"]), "-vf", drawbox, "-an", "-c:v", "ffv1", "-pix_fmt", "gray", output_mask]
    run_cmd(cmd)
    if not os.path.exists(output_mask):
        raise RuntimeError(f"Mask was not created: {output_mask}")
    return {"fps": info["fps"], "width": info["width"], "height": info["height"], "frames": info["n_frames"]}


def ensure_output_resolution(output_video, target_width, target_height):
    info = probe_video(output_video)
    if info["width"] == target_width and info["height"] == target_height:
        return output_video
    temp_output = str(Path(output_video).with_name(Path(output_video).stem + "_resized.mp4"))
    run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", output_video, "-vf", f"scale={target_width}:{target_height}:flags=lanczos", "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", temp_output])
    os.replace(temp_output, output_video)
    return output_video


class VideoRemovalEngine:
    def __init__(self, args):
        self.args = args
        self.device = get_device()
        self._validate_paths()
        print("=" * 70)
        print("Initializing ProPainter + DiffuEraser")
        print(f"Device: {self.device}")
        print("=" * 70)
        self.video_inpainting_sd = DiffuEraser(self.device, args.base_model_path, args.vae_path, args.diffueraser_path, ckpt=args.ckpt)
        print("DiffuEraser loaded.")
        self.propainter = Propainter(args.propainter_model_dir, device=self.device)
        print("ProPainter loaded.")
        print("Models initialized.")

    def _validate_paths(self):
        paths = {
            "base_model_path": self.args.base_model_path,
            "vae_path": self.args.vae_path,
            "diffueraser_path": self.args.diffueraser_path,
            "propainter_model_dir": self.args.propainter_model_dir,
        }
        missing = [f"{name}={path}" for name, path in paths.items() if not Path(path).exists()]
        if missing:
            raise FileNotFoundError("Model path(s) not found:\n" + "\n".join(missing))

    def process_one(self, input_video, input_mask, output_dir, progress=None, index=0, total=1):
        args = self.args
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = safe_filename(input_video)
        priori_path = output_dir / f"{name}_priori.mp4"
        output_path = output_dir / f"{name}_diffueraser_result.mp4"
        input_info = probe_video(input_video)
        start_time = time.time()
        if progress:
            progress(index / total, desc=f"ProPainter: {Path(input_video).name}")
        self.propainter.forward(input_video, input_mask, str(priori_path), video_length=args.video_length, ref_stride=args.ref_stride, neighbor_length=args.neighbor_length, subvideo_length=args.subvideo_length, mask_dilation=args.mask_dilation_iter)
        if not priori_path.exists():
            raise RuntimeError(f"ProPainter did not create: {priori_path}")
        if progress:
            progress(min((index + 0.5) / total, 0.99), desc=f"DiffuEraser: {Path(input_video).name}")
        self.video_inpainting_sd.forward(input_video, input_mask, str(priori_path), str(output_path), max_img_size=args.max_img_size, video_length=args.video_length, mask_dilation_iter=args.mask_dilation_iter, guidance_scale=None)
        if not output_path.exists():
            raise RuntimeError(f"DiffuEraser did not create: {output_path}")
        ensure_output_resolution(str(output_path), input_info["width"], input_info["height"])
        elapsed = time.time() - start_time
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"input_video": input_video, "input_mask": input_mask, "priori_path": str(priori_path), "output_path": str(output_path), "inference_time": elapsed}

    def process_batch(self, input_videos, boxes, save_path, progress=None):
        if not input_videos:
            raise ValueError("No videos were uploaded.")
        if len(boxes) != len(input_videos):
            raise ValueError("The number of bounding boxes must equal the number of videos.")
        session_id = uuid.uuid4().hex[:10]
        session_dir = Path(save_path) / f"session_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        results = []
        total = len(input_videos)
        for i, input_video in enumerate(input_videos):
            try:
                bbox = boxes[i]
                if bbox is None:
                    raise ValueError(f"No bounding box selected for {Path(input_video).name}")
                info = probe_video(input_video)
                bbox = clamp_bbox(bbox, info["width"], info["height"])
                video_name = safe_filename(input_video)
                output_dir = session_dir / video_name
                output_dir.mkdir(parents=True, exist_ok=True)
                mask_path = output_dir / f"{video_name}_bbox_mask.avi"
                if progress:
                    progress(i / total, desc=f"Creating mask: {Path(input_video).name}")
                create_mask_video(input_video, bbox, mask_path)
                result = self.process_one(input_video, str(mask_path), str(output_dir), progress=progress, index=i, total=total)
                result["bbox"] = bbox
                result["mask_path"] = str(mask_path)
                results.append(result)
            except Exception as exc:
                traceback.print_exc()
                results.append({"input_video": input_video, "error": str(exc)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if progress:
            progress(1.0, desc="Processing complete")
        return results


def empty_state():
    return {"session_id": None, "originals": [], "current": [], "boxes": [], "trims": [], "points": [], "meta": [], "frames": [], "selected": None}


def format_video_info(state, index):
    path = state["current"][index]
    info = state["meta"][index]
    original = Path(state["originals"][index]).name
    current = Path(path).name
    start_time, end_time = state["trims"][index]
    return f"Video {index + 1}/{len(state['current'])}\nOriginal: {original}\nWorking clip: {current}\nResolution: {info['width']} × {info['height']}\nFPS: {info['fps']:.6f}\nFrames: {info['n_frames']}\nDuration: {info['duration']:.2f} s\nSelected range: {start_time:.2f} → {end_time:.2f} s"


def slider_update(duration, value):
    return gr.Slider(minimum=0, maximum=max(duration, 0.01), value=min(max(float(value), 0), duration), step=0.01)


def prepare_selected(state, index, save_path):
    index = int(index)
    path = state["current"][index]
    info = probe_video(path)
    cache_dir = Path(save_path) / "_gui_cache" / state["session_id"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_path = cache_dir / f"{safe_filename(path)}_frame.png"
    if not frame_path.exists():
        extract_first_frame(path, frame_path)
    state["selected"] = index
    state["meta"][index] = info
    state["frames"][index] = str(frame_path)
    state["points"][index] = []
    if state["trims"][index] is None:
        state["trims"][index] = (0.0, info["duration"])
    frame = load_frame_image(frame_path)
    bbox = state["boxes"][index]
    coords = bbox if bbox is not None else (None, None, None, None)
    return state, path, frame, format_video_info(state, index), bbox_text(bbox), make_preview(str(frame_path), bbox), slider_update(info["duration"], state["trims"][index][0]), slider_update(info["duration"], state["trims"][index][1]), "Video loaded. Click the top-left corner of the box, then the bottom-right corner.", *coords


def upload_videos(files, save_path):
    paths = normalize_files(files)
    if not paths:
        state = empty_state()
        return state, gr.Dropdown(choices=[], value=None, type="index"), None, None, "No video selected.", "No bounding box selected.", None, gr.Slider(minimum=0, maximum=1, value=0, step=0.01), gr.Slider(minimum=0, maximum=1, value=1, step=0.01), "No video uploaded.", None, None, None, None
    state = {"session_id": uuid.uuid4().hex[:12], "originals": paths.copy(), "current": paths.copy(), "boxes": [None] * len(paths), "trims": [None] * len(paths), "points": [[] for _ in paths], "meta": [None] * len(paths), "frames": [None] * len(paths), "selected": 0}
    choices = [(f"{i + 1}. {Path(path).name}", i) for i, path in enumerate(paths)]
    state, current_video, frame, info_text, bbox_info, preview, start_slider, end_slider, status, *coords = prepare_selected(state, 0, save_path)
    return state, gr.Dropdown(choices=choices, value=0, type="index"), current_video, frame, info_text, bbox_info, preview, start_slider, end_slider, status, *coords


def select_video(video_index, state, save_path):
    if video_index is None or not state["current"]:
        return state, None, None, "No video selected.", "No bounding box selected.", None, gr.Slider(minimum=0, maximum=1, value=0, step=0.01), gr.Slider(minimum=0, maximum=1, value=1, step=0.01), "No video selected.", None, None, None, None
    try:
        return prepare_selected(state, int(video_index), save_path)
    except Exception as exc:
        raise gr.Error(str(exc))


def handle_frame_click(state, evt: gr.SelectData):
    if not state or state.get("selected") is None:
        return state, None, "No bounding box selected.", None, "No video selected.", None, None, None, None
    index = int(state["selected"])
    frame_path = state["frames"][index]
    info = state["meta"][index] or probe_video(state["current"][index])
    idx = evt.index
    if not isinstance(idx, (tuple, list)) or len(idx) < 2:
        bbox = state["boxes"][index]
        return state, load_frame_image(frame_path), bbox_text(bbox), make_preview(frame_path, bbox), "Could not read click coordinates.", *(bbox if bbox else (None, None, None, None))
    x = max(0, min(int(round(idx[0])), info["width"] - 1))
    y = max(0, min(int(round(idx[1])), info["height"] - 1))
    points = state["points"][index]
    if len(points) >= 2:
        points = []
        state["boxes"][index] = None
    points.append((x, y))
    state["points"][index] = points
    frame = load_frame_image(frame_path)
    if len(points) == 1:
        return state, draw_bbox(frame, point=(x, y)), "First point set. Click the bottom-right corner.", None, f"Top-left point: ({x}, {y})", x, y, None, None
    bbox = get_bbox_from_points(points, info["width"], info["height"])
    state["boxes"][index] = bbox
    return state, draw_bbox(frame, bbox), bbox_text(bbox), make_preview(frame_path, bbox), "Bounding box set. Click again to replace it.", *bbox


def apply_bbox_coordinates(xmin, ymin, xmax, ymax, state):
    if not state or state.get("selected") is None:
        raise gr.Error("Select a video first.")
    index = int(state["selected"])
    info = state["meta"][index] or probe_video(state["current"][index])
    try:
        bbox = clamp_bbox((xmin, ymin, xmax, ymax), info["width"], info["height"])
    except Exception as exc:
        raise gr.Error(str(exc))
    state["boxes"][index] = bbox
    state["points"][index] = []
    frame = load_frame_image(state["frames"][index])
    return state, draw_bbox(frame, bbox), bbox_text(bbox), make_preview(state["frames"][index], bbox), "Bounding box applied.", *bbox


def clear_bbox(state):
    if not state or state.get("selected") is None:
        return state, None, "No bounding box selected.", None, "No video selected.", None, None, None, None
    index = int(state["selected"])
    state["boxes"][index] = None
    state["points"][index] = []
    frame = load_frame_image(state["frames"][index])
    return state, frame, bbox_text(None), None, "Bounding box cleared. Click two points to create a new one.", None, None, None, None


def apply_cut(video_index, start_time, end_time, state, save_path):
    if video_index is None:
        raise gr.Error("Select a video first.")
    index = int(video_index)
    original = state["originals"][index]
    info = probe_video(original)
    start_time = float(start_time)
    end_time = min(float(end_time), info["duration"])
    if start_time < 0 or end_time <= start_time:
        raise gr.Error("End time must be greater than start time and start time cannot be negative.")
    session_dir = Path(save_path) / "_gui_cache" / state["session_id"] / "clips"
    session_dir.mkdir(parents=True, exist_ok=True)
    name = safe_filename(original)
    clipped_path = session_dir / f"{name}_{start_time:.2f}_{end_time:.2f}.mp4"
    try:
        trim_video(original, start_time, end_time, clipped_path)
        clipped_info = probe_video(clipped_path)
        frame_path = session_dir / f"{name}_{start_time:.2f}_{end_time:.2f}_frame.png"
        extract_first_frame(clipped_path, frame_path)
    except Exception as exc:
        raise gr.Error(str(exc))
    state["current"][index] = str(clipped_path)
    state["trims"][index] = (start_time, end_time)
    state["boxes"][index] = None
    state["points"][index] = []
    state["meta"][index] = clipped_info
    state["frames"][index] = str(frame_path)
    frame = load_frame_image(frame_path)
    return state, str(clipped_path), frame, format_video_info(state, index), bbox_text(None), None, slider_update(clipped_info["duration"], 0), slider_update(clipped_info["duration"], clipped_info["duration"]), "Cut applied. Draw a new bounding box on the new first frame.", None, None, None, None


def reset_cut(video_index, state, save_path):
    if video_index is None:
        raise gr.Error("Select a video first.")
    index = int(video_index)
    original = state["originals"][index]
    info = probe_video(original)
    state["current"][index] = original
    state["trims"][index] = (0.0, info["duration"])
    state["boxes"][index] = None
    state["points"][index] = []
    cache_dir = Path(save_path) / "_gui_cache" / state["session_id"]
    frame_path = cache_dir / f"video_{index}_frame.png"
    extract_first_frame(original, frame_path)
    state["meta"][index] = info
    state["frames"][index] = str(frame_path)
    frame = load_frame_image(frame_path)
    return state, original, frame, format_video_info(state, index), bbox_text(None), None, slider_update(info["duration"], 0), slider_update(info["duration"], info["duration"]), "Cut reset. Original video restored.", None, None, None, None


def load_models_ui(args):
    global ENGINE
    if ENGINE is not None:
        yield True, f"Models already loaded on {ENGINE.device}."
        return
    yield False, "Loading models. The UI will stay responsive; errors will appear here instead of crashing the app."
    try:
        with ENGINE_LOCK:
            if ENGINE is None:
                start = time.time()
                ENGINE = VideoRemovalEngine(args)
                elapsed = time.time() - start
        yield True, f"Models loaded successfully on {ENGINE.device} in {elapsed:.1f} s."
    except Exception as exc:
        traceback.print_exc()
        ENGINE = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield False, "MODEL LOAD FAILED:\n" + str(exc) + "\n\nFull traceback is printed in the server terminal."


def run_gui(state, models_loaded, save_path, progress=gr.Progress(track_tqdm=False)):
    global ENGINE
    if not state or not state.get("current"):
        raise gr.Error("Upload at least one video.")
    if not models_loaded or ENGINE is None:
        raise gr.Error("Load the models first.")
    missing = [Path(state["originals"][i]).name for i, box in enumerate(state["boxes"]) if box is None]
    if missing:
        raise gr.Error("Draw a bounding box for every video:\n" + "\n".join(missing))
    results = ENGINE.process_batch(state["current"], state["boxes"], save_path, progress=progress)
    successful = [r for r in results if r.get("output_path") and os.path.exists(r["output_path"])]
    files = [r["output_path"] for r in successful]
    lines = ["PROCESSING COMPLETE", "", f"Videos: {len(state['current'])}", f"Successful: {len(successful)}", f"Failed: {len(results) - len(successful)}", ""]
    for result in results:
        name = Path(result["input_video"]).name
        if result.get("output_path"):
            lines += [f"[OK] {name}", f"     bbox: {result['bbox']}", f"     mask: {result['mask_path']}", f"     output: {result['output_path']}", f"     time: {result['inference_time']:.2f} s", ""]
        else:
            lines += [f"[FAILED] {name}", f"     {result.get('error', 'Unknown error')}", ""]
    return files, files[0] if files else None, "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description="Gradio GUI for ProPainter + DiffuEraser video object removal.")
    parser.add_argument("--video_length", type=int, default=10)
    parser.add_argument("--mask_dilation_iter", type=int, default=8)
    parser.add_argument("--max_img_size", type=int, default=960)
    parser.add_argument("--ref_stride", type=int, default=10)
    parser.add_argument("--neighbor_length", type=int, default=10)
    parser.add_argument("--subvideo_length", type=int, default=50)
    parser.add_argument("--ckpt", type=str, default="2-Step")
    parser.add_argument("--base_model_path", type=str, default="weights/stable-diffusion-v1-5")
    parser.add_argument("--vae_path", type=str, default="weights/sd-vae-ft-mse")
    parser.add_argument("--diffueraser_path", type=str, default="weights/diffuEraser")
    parser.add_argument("--propainter_model_dir", type=str, default="weights/propainter")
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--share", action="store_true")
    return parser


def build_demo(args):
    save_path = args.save_path
    def run_gui_event(state, models_loaded, progress=gr.Progress(track_tqdm=False)):
        return run_gui(state, models_loaded, save_path, progress)
    initial_state = empty_state()
    block_kwargs = {}
    if "theme" in __import__("inspect").signature(gr.Blocks).parameters:
        block_kwargs["theme"] = gr.themes.Soft()
    if "css" in __import__("inspect").signature(gr.Blocks).parameters:
        block_kwargs["css"] = GUI_CSS
    block_kwargs["title"] = "Video Object Remover"
    with gr.Blocks(**block_kwargs) as demo:
        session_state = gr.State(initial_state, time_to_live=86400)
        models_loaded = gr.State(False)
        gr.HTML('<div class="hero"><h1>Video Object & Text Remover</h1><p>Upload, trim, click two corners of the object, then run ProPainter + DiffuEraser.</p></div>')
        with gr.Column(elem_classes="section"):
            gr.Markdown('<div class="step">1. Upload Videos</div>')
            videos = gr.File(label="Upload one or more videos", file_count="multiple", file_types=["video"], type="filepath")
            video_selector = gr.Dropdown(label="Current Video", choices=[], value=None, type="index", interactive=True)
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
        with gr.Column(elem_classes="section"):
            gr.Markdown('<div class="step">2. Trim Current Video</div>')
            current_video = gr.Video(label="Current Video", interactive=False, height=420)
            video_info = gr.Textbox(label="Video Information", lines=8, interactive=False, elem_classes="mono")
            with gr.Row():
                start_slider = gr.Slider(minimum=0, maximum=1, value=0, step=0.01, label="Start Time (seconds)")
                end_slider = gr.Slider(minimum=0, maximum=1, value=1, step=0.01, label="End Time (seconds)")
            with gr.Row():
                cut_button = gr.Button("✂ Apply Cut", variant="primary")
                reset_cut_button = gr.Button("↩ Reset Cut")
            trim_status = gr.Textbox(label="Trim Status", interactive=False)
        with gr.Column(elem_classes="section"):
            gr.Markdown('<div class="step">3. Mark Object / Text</div>')
            gr.Markdown("Click the **top-left** corner of the box, then click the **bottom-right** corner. Click again to replace the box. You can also edit the coordinates manually.")
            frame_image = gr.Image(label="First Frame — click two corners", type="numpy", interactive=True, height=600, width=1000)
            bbox_coordinates = gr.Textbox(label="Bounding Box", lines=6, interactive=False, elem_classes="mono")
            with gr.Row():
                xmin = gr.Number(label="xmin", precision=0)
                ymin = gr.Number(label="ymin", precision=0)
                xmax = gr.Number(label="xmax", precision=0)
                ymax = gr.Number(label="ymax", precision=0)
            with gr.Row():
                apply_bbox_button = gr.Button("Apply Coordinates", variant="primary")
                clear_bbox_button = gr.Button("Clear Box")
            bbox_status = gr.Textbox(label="Box Status", interactive=False)
            mask_preview = gr.Image(label="Mask Preview — highlighted area will be removed", interactive=False, height=420)
        with gr.Column(elem_classes="section"):
            gr.Markdown('<div class="step">4. Load Models</div>')
            load_models_button = gr.Button("🧠 Load ProPainter + DiffuEraser", variant="primary")
            model_status = gr.Textbox(label="Model Status", lines=4, interactive=False, elem_classes="mono")
        with gr.Column(elem_classes="section"):
            gr.Markdown('<div class="step">5. Run Removal</div>')
            remove_button = gr.Button("🚀 Remove Objects", variant="primary", elem_id="remove-btn")
            processing_status = gr.Textbox(label="Processing Status", lines=18, interactive=False, elem_classes="mono")
        with gr.Column(elem_classes="section"):
            gr.Markdown('<div class="step">6. Results</div>')
            result_video = gr.Video(label="First Result", height=420)
            result_files = gr.Files(label="All Result Videos")
        upload_outputs = [session_state, video_selector, current_video, frame_image, video_info, bbox_coordinates, mask_preview, start_slider, end_slider, upload_status, xmin, ymin, xmax, ymax]
        videos.upload(lambda files: upload_videos(files, save_path), inputs=videos, outputs=upload_outputs, show_progress="minimal")
        videos.clear(lambda: (empty_state(), gr.Dropdown(choices=[], value=None, type="index"), None, None, "No video selected.", "No bounding box selected.", None, gr.Slider(minimum=0, maximum=1, value=0, step=0.01), gr.Slider(minimum=0, maximum=1, value=1, step=0.01), "Videos cleared.", None, None, None, None), outputs=upload_outputs, show_progress="hidden")
        video_selector.input(lambda idx, state: select_video(idx, state, save_path), inputs=[video_selector, session_state], outputs=[session_state, current_video, frame_image, video_info, bbox_coordinates, mask_preview, start_slider, end_slider, trim_status, xmin, ymin, xmax, ymax], show_progress="minimal")
        frame_image.select(handle_frame_click, inputs=[session_state], outputs=[session_state, frame_image, bbox_coordinates, mask_preview, bbox_status, xmin, ymin, xmax, ymax], show_progress="hidden")
        apply_bbox_button.click(apply_bbox_coordinates, inputs=[xmin, ymin, xmax, ymax, session_state], outputs=[session_state, frame_image, bbox_coordinates, mask_preview, bbox_status, xmin, ymin, xmax, ymax], show_progress="minimal")
        clear_bbox_button.click(clear_bbox, inputs=[session_state], outputs=[session_state, frame_image, bbox_coordinates, mask_preview, bbox_status, xmin, ymin, xmax, ymax], show_progress="hidden")
        cut_button.click(lambda idx, start, end, state: apply_cut(idx, start, end, state, save_path), inputs=[video_selector, start_slider, end_slider, session_state], outputs=[session_state, current_video, frame_image, video_info, bbox_coordinates, mask_preview, start_slider, end_slider, trim_status, xmin, ymin, xmax, ymax], show_progress="full")
        reset_cut_button.click(lambda idx, state: reset_cut(idx, state, save_path), inputs=[video_selector, session_state], outputs=[session_state, current_video, frame_image, video_info, bbox_coordinates, mask_preview, start_slider, end_slider, trim_status, xmin, ymin, xmax, ymax], show_progress="minimal")
        load_models_button.click(lambda: load_models_ui(args), inputs=None, outputs=[models_loaded, model_status], concurrency_limit=1, concurrency_id="model", show_progress="full")
        remove_button.click(run_gui_event, inputs=[session_state, models_loaded], outputs=[result_files, result_video, processing_status], concurrency_limit=1, concurrency_id="model", show_progress="full")
    return demo


def main():
    args = build_parser().parse_args()
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    require_tools()
    print(f"Gradio version: {gr.__version__}")
    print("Gradio app starting. Models are NOT loaded at startup.")
    demo = build_demo(args)
    launch_kwargs = {
        "server_name": args.server_name,
        "server_port": args.server_port,
        "share": args.share,
        "show_error": True
    }
    import inspect
    launch_params = inspect.signature(demo.launch).parameters
    if "theme" in launch_params:
        launch_kwargs["theme"] = gr.themes.Soft()
    if "css" in launch_params:
        launch_kwargs["css"] = GUI_CSS
    demo.queue(max_size=16, default_concurrency_limit=8).launch(**launch_kwargs)


if __name__ == "__main__":
    main()