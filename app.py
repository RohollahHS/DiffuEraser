import os
import cv2
import uuid
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from gradio_image_annotation import image_annotator


# ============================================================
# YOUR MODEL IMPORTS
# ============================================================
# Replace these with the actual imports from your project.
#
# Example:
# from inference import propainter, video_inpainting_sd
#
# The important thing is that these two objects already expose:
#
# propainter.forward(...)
# video_inpainting_sd.forward(...)
#
# ----------------------------------------------------------------
# from your_module import propainter, video_inpainting_sd
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

# These are NOT shown to the user.
# Change these according to the parameters you normally use.

VIDEO_LENGTH = 81
REF_STRIDE = 10
NEIGHBOR_LENGTH = 10
SUBVIDEO_LENGTH = 50

MASK_DILATION_ITER = 6
MAX_IMG_SIZE = 960
GUIDANCE_SCALE = None

# Directory where final results are kept
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_video_info(video_path):
    """
    Open the input video and return:
        first_frame_rgb
        fps
        frame_count
        width
        height
        duration
    """
    if video_path is None:
        raise gr.Error("Please upload a video first.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise gr.Error(f"Could not open video:\n{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise gr.Error("Could not read the first frame of the video.")

    if fps <= 0:
        raise gr.Error("Could not determine the video FPS.")

    # OpenCV -> RGB
    first_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    duration = frame_count / fps if fps > 0 else 0

    info = (
        f"Resolution: {width} × {height}\n"
        f"FPS: {fps:.6f}\n"
        f"Frames: {frame_count}\n"
        f"Duration: {duration:.2f} sec"
    )

    return first_frame_rgb, fps, frame_count, width, height, info


def load_video(video_path):
    """
    Called after the user uploads a video.

    Returns an annotation object compatible with image_annotator.
    """
    (
        first_frame_rgb,
        fps,
        frame_count,
        width,
        height,
        info,
    ) = get_video_info(video_path)

    annotation = {
        "image": first_frame_rgb,
        "boxes": [],
    }

    return annotation, info


def extract_box(annotation):
    """
    Extract the single bounding box from the annotation component.

    Returns:
        (xmin, ymin, xmax, ymax)
    """
    if annotation is None:
        return None

    boxes = annotation.get("boxes", [])

    if not boxes:
        return None

    box = boxes[0]

    xmin = int(round(box["xmin"]))
    ymin = int(round(box["ymin"]))
    xmax = int(round(box["xmax"]))
    ymax = int(round(box["ymax"]))

    return xmin, ymin, xmax, ymax


def show_bbox(annotation):
    """
    Display the selected coordinates for debugging/confirmation.
    This is an output, not an additional user input.
    """
    bbox = extract_box(annotation)

    if bbox is None:
        return "No bounding box selected."

    xmin, ymin, xmax, ymax = bbox

    return (
        f"xmin = {xmin}\n"
        f"ymin = {ymin}\n"
        f"xmax = {xmax}\n"
        f"ymax = {ymax}\n"
        f"width  = {xmax - xmin}\n"
        f"height = {ymax - ymin}"
    )


# ============================================================
# MASK PREVIEW
# ============================================================

def make_mask_preview(video_path, annotation):
    """
    Creates a visualization of the mask over the first frame.

    White/bright region = area that will be removed.
    """
    if video_path is None or annotation is None:
        return None

    bbox = extract_box(annotation)

    if bbox is None:
        return None

    (
        first_frame_rgb,
        fps,
        frame_count,
        width,
        height,
        info,
    ) = get_video_info(video_path)

    xmin, ymin, xmax, ymax = bbox

    # Clamp to image dimensions
    xmin = max(0, min(xmin, width - 1))
    xmax = max(0, min(xmax, width))
    ymin = max(0, min(ymin, height - 1))
    ymax = max(0, min(ymax, height))

    if xmax <= xmin or ymax <= ymin:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[ymin:ymax, xmin:xmax] = 255

    # Create a visualization
    preview = first_frame_rgb.copy().astype(np.float32)

    # Red translucent overlay
    red = np.zeros_like(preview)
    red[:, :, 0] = 255

    alpha = 0.45

    region = mask > 0
    preview[region] = (
        preview[region] * (1.0 - alpha)
        + red[region] * alpha
    )

    preview = np.clip(preview, 0, 255).astype(np.uint8)

    # Draw rectangle border
    cv2.rectangle(
        preview,
        (xmin, ymin),
        (xmax - 1, ymax - 1),
        (255, 255, 0),
        thickness=3,
    )

    return preview


# ============================================================
# MASK VIDEO CREATION
# ============================================================

def create_mask_video(video_path, bbox, output_mask_path):
    """
    Create a mask video with EXACTLY:

        - same width
        - same height
        - same FPS
        - same number of frames

    as the source video.

    Every frame contains:
        black = keep
        white = remove

    FFV1 is used because the supplied read_mask() uses:

        m = np.array(mask > 0)

    A lossy codec can introduce small nonzero values into black
    areas, which is bad for a binary mask.
    """

    if bbox is None:
        raise ValueError("No bounding box was provided.")

    xmin, ymin, xmax, ymax = bbox

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video metadata.")

    # Clamp bounding box to original video coordinates.
    xmin = max(0, min(int(xmin), width - 1))
    xmax = max(0, min(int(xmax), width))
    ymin = max(0, min(int(ymin), height - 1))
    ymax = max(0, min(int(ymax), height))

    if xmax <= xmin or ymax <= ymin:
        cap.release()
        raise ValueError("The selected bounding box is invalid.")

    # --------------------------------------------------------
    # Use FFV1 lossless video for the mask.
    # AVI is generally more reliable for FFV1 than MP4.
    # --------------------------------------------------------
    output_mask_path = str(output_mask_path)

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")

    writer = cv2.VideoWriter(
        output_mask_path,
        fourcc,
        fps,
        (width, height),
        True,
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            "Could not create a lossless FFV1 mask video. "
            "Your OpenCV/FFmpeg build may not contain FFV1 support."
        )

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Binary mask
        mask = np.zeros((height, width), dtype=np.uint8)

        mask[ymin:ymax, xmin:xmax] = 255

        # VideoWriter expects 3-channel frame
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        writer.write(mask_bgr)

        frame_count += 1

    cap.release()
    writer.release()

    if frame_count == 0:
        raise RuntimeError("No frames were written to the mask video.")

    return output_mask_path, fps, frame_count, width, height


# ============================================================
# OPTIONAL: VERIFY THE GENERATED MASK
# ============================================================

def verify_mask_video(mask_path, input_fps, input_frames, width, height):
    """
    Verify that the generated mask really matches the source
    video dimensions, FPS and frame count.
    """

    cap = cv2.VideoCapture(mask_path)

    if not cap.isOpened():
        raise RuntimeError("Could not reopen generated mask video.")

    mask_fps = float(cap.get(cv2.CAP_PROP_FPS))
    mask_frames_metadata = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mask_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mask_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Count actual readable frames as an additional check.
    actual_frames = 0

    while True:
        ret, _ = cap.read()

        if not ret:
            break

        actual_frames += 1

    cap.release()

    if mask_width != width or mask_height != height:
        raise RuntimeError(
            f"Mask resolution mismatch: "
            f"mask={mask_width}x{mask_height}, "
            f"video={width}x{height}"
        )

    if abs(mask_fps - input_fps) > 1e-3:
        raise RuntimeError(
            f"Mask FPS mismatch: mask={mask_fps}, video={input_fps}"
        )

    if actual_frames != input_frames:
        raise RuntimeError(
            f"Mask frame count mismatch: "
            f"mask={actual_frames}, video={input_frames}"
        )

    return True


# ============================================================
# MODEL INFERENCE
# ============================================================

def run_removal(video_path, annotation):
    """
    Full inference pipeline:

        1. Read video properties
        2. Get bounding box
        3. Generate binary mask video
        4. ProPainter
        5. Diffuseraser
        6. Return resulting video
    """

    if video_path is None:
        raise gr.Error("Please upload a video.")

    bbox = extract_box(annotation)

    if bbox is None:
        raise gr.Error(
            "Please draw a bounding box around the object/text "
            "you want to remove."
        )

    try:
        (
            _first_frame,
            input_fps,
            input_frame_count,
            width,
            height,
            _info,
        ) = get_video_info(video_path)

        # ----------------------------------------------------
        # Create a private working directory
        # ----------------------------------------------------
        work_dir = Path(
            tempfile.mkdtemp(prefix="video_removal_")
        )

        mask_path = work_dir / "bbox_mask.avi"
        priori_path = work_dir / "priori.mp4"
        intermediate_output = work_dir / "diffuseraser_result.mp4"

        # ----------------------------------------------------
        # Create binary mask video
        # ----------------------------------------------------
        (
            mask_path,
            mask_fps,
            mask_frame_count,
            mask_width,
            mask_height,
        ) = create_mask_video(
            video_path,
            bbox,
            mask_path,
        )

        # ----------------------------------------------------
        # Verify mask compatibility
        # ----------------------------------------------------
        verify_mask_video(
            mask_path,
            input_fps,
            input_frame_count,
            width,
            height,
        )

        print("=" * 70)
        print("VIDEO")
        print(f"Path       : {video_path}")
        print(f"Resolution : {width} x {height}")
        print(f"FPS        : {input_fps}")
        print(f"Frames     : {input_frame_count}")

        print("\nBOUNDING BOX")
        print(f"xmin       : {bbox[0]}")
        print(f"ymin       : {bbox[1]}")
        print(f"xmax       : {bbox[2]}")
        print(f"ymax       : {bbox[3]}")

        print("\nMASK")
        print(f"Path       : {mask_path}")
        print(f"FPS        : {mask_fps}")
        print(f"Frames     : {mask_frame_count}")
        print(f"Resolution : {mask_width} x {mask_height}")

        print("=" * 70)

        # ----------------------------------------------------
        # STEP 1: PROPainter
        # ----------------------------------------------------
        print("\nRunning ProPainter...")

        propainter_result = propainter.forward(
            str(video_path),
            str(mask_path),
            str(priori_path),

            video_length=VIDEO_LENGTH,
            ref_stride=REF_STRIDE,
            neighbor_length=NEIGHBOR_LENGTH,
            subvideo_length=SUBVIDEO_LENGTH,
            mask_dilation=MASK_DILATION_ITER,
        )

        # Some implementations return the generated path.
        # Prefer the explicitly provided priori_path.
        if not priori_path.exists():

            if isinstance(propainter_result, (str, Path)):
                returned_path = Path(propainter_result)

                if returned_path.exists():
                    shutil.copy2(
                        returned_path,
                        priori_path,
                    )

            if not priori_path.exists():
                raise RuntimeError(
                    "ProPainter finished, but priori_path was not created:\n"
                    f"{priori_path}"
                )

        print("ProPainter finished.")

        # ----------------------------------------------------
        # STEP 2: DIFFUSERASER
        # ----------------------------------------------------
        print("\nRunning Diffuseraser...")

        diffuser_result = video_inpainting_sd.forward(
            str(video_path),
            str(mask_path),
            str(priori_path),
            str(intermediate_output),

            max_img_size=MAX_IMG_SIZE,
            video_length=VIDEO_LENGTH,
            mask_dilation_iter=MASK_DILATION_ITER,
            guidance_scale=GUIDANCE_SCALE,
        )

        # ----------------------------------------------------
        # Determine final result
        # ----------------------------------------------------
        if intermediate_output.exists():
            generated_video = intermediate_output

        elif isinstance(diffuser_result, (str, Path)):
            returned_path = Path(diffuser_result)

            if returned_path.exists():
                generated_video = returned_path
            else:
                raise RuntimeError(
                    "Diffuseraser returned a path that does not exist:\n"
                    f"{returned_path}"
                )

        else:
            raise RuntimeError(
                "Diffuseraser finished, but no output video was found."
            )

        print("Diffuseraser finished.")
        print(f"Generated video: {generated_video}")

        # ----------------------------------------------------
        # Copy final result into a persistent output directory.
        # Do this because work_dir may be deleted later.
        # ----------------------------------------------------
        output_filename = (
            f"removed_{uuid.uuid4().hex[:10]}.mp4"
        )

        final_output = OUTPUT_DIR / output_filename

        shutil.copy2(
            generated_video,
            final_output,
        )

        # Cleanup intermediate files
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass

        return str(final_output)

    except gr.Error:
        raise

    except Exception as e:
        print("\nERROR DURING INFERENCE:")
        import traceback
        traceback.print_exc()

        raise gr.Error(
            f"Video removal failed:\n\n{str(e)}"
        )


# ============================================================
# UI
# ============================================================

TITLE_HTML = """
<div style="
    text-align:center;
    font-size:36px;
    font-family:Arial, Helvetica, sans-serif;
    font-weight:700;
    margin-top:10px;
">
    Video Object / Text Remover
</div>

<div style="
    text-align:center;
    font-size:17px;
    font-family:Arial, Helvetica, sans-serif;
    color:#666;
    margin-top:8px;
    margin-bottom:20px;
">
    Upload a video, draw one box around the object or text,
    and run video inpainting.
</div>
"""


INSTRUCTIONS = """
### How to use

1. Upload a video.
2. The first frame will appear automatically.
3. Draw **one bounding box** around the object/text to remove.
4. Press **Remove Object**.
5. The same bounding box is used for every video frame.

The red preview shows the region that will become the removal mask.
"""


CSS = """
#main-container {
    max-width: 1200px;
    margin: auto;
}

#video-input,
#annotator,
#mask-preview,
#result-video {
    width: 100%;
}

#remove-button {
    width: 60%;
    margin: 15px auto;
    display: block;
    font-size: 20px;
}

.status-box textarea {
    font-family: monospace !important;
}

footer {
    display: none !important;
}
"""


with gr.Blocks(
    title="Video Object Remover",
    css=CSS,
    theme=gr.themes.Soft(),
) as demo:

    with gr.Column(elem_id="main-container"):

        gr.HTML(TITLE_HTML)

        gr.Markdown(INSTRUCTIONS)

        # ----------------------------------------------------
        # Video
        # ----------------------------------------------------
        video_input = gr.Video(
            label="1. Upload Video",
            sources=["upload"],
            format="mp4",
            elem_id="video-input",
        )

        video_info = gr.Textbox(
            label="Video Information",
            lines=4,
            interactive=False,
            elem_classes=["status-box"],
        )

        # ----------------------------------------------------
        # First frame + Bounding box
        # ----------------------------------------------------
        gr.Markdown(
            "### 2. Draw the removal box\n"
            "Drag a rectangle around the object or text you want removed."
        )

        bbox_editor = image_annotator(
            value=None,
            label="First Frame — Draw Bounding Box",
            single_box=True,
            disable_edit_boxes=True,
            show_remove_button=True,
            box_min_size=5,
            box_thickness=3,
            box_selected_thickness=4,
            height=600,
            width=1000,
        )

        bbox_coordinates = gr.Textbox(
            label="Selected Bounding Box",
            lines=6,
            interactive=False,
            placeholder="Draw a box above...",
            elem_classes=["status-box"],
        )

        # ----------------------------------------------------
        # Mask Preview
        # ----------------------------------------------------
        mask_preview = gr.Image(
            label="Mask Preview",
            type="numpy",
            interactive=False,
            height=500,
            elem_id="mask-preview",
        )

        # ----------------------------------------------------
        # Remove
        # ----------------------------------------------------
        remove_button = gr.Button(
            "Remove Object",
            variant="primary",
            elem_id="remove-button",
            size="lg",
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------
        result_video = gr.Video(
            label="Result",
            autoplay=False,
            elem_id="result-video",
        )

        # ----------------------------------------------------
        # EVENTS
        # ----------------------------------------------------

        # Automatically extract first frame after video upload
        video_input.change(
            fn=load_video,
            inputs=video_input,
            outputs=[
                bbox_editor,
                video_info,
            ],
        )

        # Update coordinate display
        bbox_editor.change(
            fn=show_bbox,
            inputs=bbox_editor,
            outputs=bbox_coordinates,
        )

        # Update mask visualization
        bbox_editor.change(
            fn=make_mask_preview,
            inputs=[
                video_input,
                bbox_editor,
            ],
            outputs=mask_preview,
        )

        # Run actual inference
        remove_button.click(
            fn=run_removal,
            inputs=[
                video_input,
                bbox_editor,
            ],
            outputs=result_video,
        )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.queue()

    demo.launch(
        server_name="0.0.0.0",
        server_port=8000,

        # Set True if you need a public Gradio share URL.
        share=True,

        show_error=True,
    )