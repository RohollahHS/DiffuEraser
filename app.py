```python
# Remove: import glob
# Add:
import json
from fractions import Fraction

# Keep your other imports exactly as they are.

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}

def require_ffmpeg():
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError("Missing required executable(s): " + ", ".join(missing) + ". Install FFmpeg and make sure ffmpeg/ffprobe are in PATH.")

def _command_stderr(result):
    if isinstance(result.stderr, bytes):
        return result.stderr.decode("utf-8", errors="replace")
    return result.stderr or ""

def read_video_info(video_path):
    require_ffmpeg()
    video_path = str(video_path)
    if not os.path.isfile(video_path):
        raise RuntimeError(f"Video does not exist: {video_path}")

    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration:format=duration",
        "-of", "json",
        video_path
    ]
    try:
        probe = subprocess.run(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe timed out while reading: {video_path}")

    if probe.returncode != 0:
        raise RuntimeError(f"Could not probe video:\n{_command_stderr(probe)}")

    try:
        data = json.loads(probe.stdout)
        stream = data["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Invalid ffprobe output for: {video_path}") from e

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)

    rate = stream.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    duration_value = stream.get("duration") or data.get("format", {}).get("duration") or 0
    try:
        duration = float(duration_value)
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
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-ss", "0",
        "-i", video_path,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "png",
        "pipe:1"
    ]
    try:
        frame_result = subprocess.run(
            frame_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timed out while extracting the first frame: {video_path}")

    if frame_result.returncode != 0:
        raise RuntimeError(f"Could not read first frame:\n{_command_stderr(frame_result)}")

    frame_array = np.frombuffer(frame_result.stdout, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode first frame: {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return {
        "first_frame": frame,
        "fps": fps,
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "duration": duration
    }

def _run_ffmpeg(cmd, label, timeout=1800):
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{label} timed out.")
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n\n{result.stderr[-4000:]}")
    return result

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

    clip_duration = end_time - start_time
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
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
    _run_ffmpeg(cmd, "FFmpeg video trimming")
    if not os.path.isfile(output_path):
        raise RuntimeError(f"FFmpeg completed but output was not created: {output_path}")
    return output_path

def ensure_output_resolution(output_video, target_width, target_height):
    require_ffmpeg()
    info = read_video_info(output_video)
    if info["width"] == target_width and info["height"] == target_height:
        return output_video

    temp_output = str(Path(output_video).with_suffix("")) + "_resized.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", output_video,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-vf", f"scale={target_width}:{target_height}:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        temp_output
    ]
    _run_ffmpeg(cmd, "FFmpeg output resizing")
    if not os.path.isfile(temp_output):
        raise RuntimeError(f"Resized output was not created: {temp_output}")
    os.replace(temp_output, output_video)
    return output_video

def create_initial_state(files):
    paths = normalize_files(files)
    return {
        "originals": paths,
        "current": paths.copy(),
        "boxes": [None] * len(paths),
        "trims": [None] * len(paths)
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
            "No video selected."
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
        f"Loaded {len(paths)} video(s)."
    )

def select_video(video_index, state):
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

    try:
        info = read_video_info(path)
    except Exception as e:
        raise gr.Error(str(e))

    if state["trims"][i] is None:
        state["trims"] = state["trims"].copy()
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

    return (
        path,
        annotation,
        video_info,
        bbox_text(bbox),
        gr.update(minimum=0, maximum=max(info["duration"], 0.01), value=start_time),
        gr.update(minimum=0, maximum=max(info["duration"], 0.01), value=end_time)
    )

def save_bbox(video_index, annotation, state):
    if video_index is None:
        return state, "No video selected."

    i = int(video_index)
    bbox = get_bbox(annotation)
    state = dict(state)
    state["boxes"] = state["boxes"].copy()
    state["boxes"][i] = bbox
    return state, bbox_text(bbox)

def update_mask_preview(video_index, annotation, state):
    if video_index is None or not state["current"]:
        return None

    bbox = get_bbox(annotation)
    if bbox is None:
        return None

    path = state["current"][int(video_index)]
    try:
        return make_preview(path, bbox)
    except Exception:
        return None

def load_selected_video(video_path, state):
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
        "trims": [(0.0, info["duration"])]
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
        video_path,
        annotation,
        video_info,
        bbox_text(None),
        gr.update(minimum=0, maximum=max(info["duration"], 0.01), value=0),
        gr.update(minimum=0, maximum=max(info["duration"], 0.01), value=info["duration"]),
        None,
        "Video loaded successfully. Draw a bounding box."
    )

def build_demo(engine, save_path, browse_path):
    css = """
.gradio-container {
    max-width: 1350px !important;
    margin: 0 auto !important;
    background: #f6f8fb !important;
}
.hero {
    text-align: center;
    padding: 28px 20px 20px;
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
    margin: 10px 0 0;
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

    with gr.Blocks(title="Video Object Remover", theme=gr.themes.Soft(), css=css) as demo:
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

            gr.Markdown(
                "Or browse files on the machine running this Gradio app:"
            )

            server_video = gr.FileExplorer(
                label="Host Filesystem",
                root_dir=str(Path(browse_path).resolve()),
                glob="**/*",
                file_count="single",
                interactive=True
            )

            load_video_button = gr.Button(
                "📂 Load Host File",
                variant="primary"
            )

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

            gr.Markdown(
                "Draw **one box** around the object or text. Every video has its own independent box."
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
            gr.Markdown('<div class="step">4. Run Video Removal</div>')

            gr.Markdown(
                "Make sure every uploaded video has a bounding box before starting inference."
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
            gr.Markdown('<div class="step">5. Results</div>')

            result_video = gr.Video(
                label="First Result",
                height=420
            )

            result_files = gr.Files(
                label="All Result Videos"
            )

        # IMPORTANT: use .upload(), not .change(), for uploaded files.
        # Also: bbox_coordinates appears only once in outputs.
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
            fn=save_bbox,
            inputs=[video_selector, annotation, session_state],
            outputs=[session_state, bbox_coordinates],
            queue=False
        )

        annotation.change(
            fn=update_mask_preview,
            inputs=[video_selector, annotation, session_state],
            outputs=mask_preview,
            queue=False
        )

        cut_button.click(
            fn=lambda idx, start, end, state: apply_cut(
                idx, start, end, state, save_path
            ),
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
            outputs=[result_files, result_video, status]
        )

    return demo

def build_parser():
    parser = argparse.ArgumentParser(
        description="GUI for ProPainter + DiffuEraser video object removal."
    )
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
    parser.add_argument("--browse_path", type=str, default=".")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--share", action="store_true")
    return parser

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
```