import os
import json
import cv2
import numpy as np

from sam3.model_builder import build_sam3_video_predictor

DEVICE = "cuda"

VIDEO_PATH = "/scratch/rohhs/downloads/yt-dlp/deputy_saves_dog_from_car_on_fire_60.mp4"
OUTPUT_JSON = "text_boxes.json"

CHECKPOINT_PATH = "/scratch/rohhs/huggingface/hub/sam3/sam3.pt"
BPE_PATH = None

TEXT_PROMPT = "text"
OUTPUT_PROB_THRESHOLD = 0.25


print(f"Using device: {DEVICE}")
print("Loading SAM3...")

predictor = build_sam3_video_predictor(checkpoint_path=CHECKPOINT_PATH)

print("Starting video session...")

response = predictor.handle_request({"type": "start_session", "resource_path": VIDEO_PATH})
session_id = response["session_id"]

session = predictor._get_session(session_id)
inference_state = session["state"]

video_width = int(inference_state["orig_width"])
video_height = int(inference_state["orig_height"])
num_frames = int(inference_state["num_frames"])

print(f"Video resolution: {video_width}x{video_height}")
print(f"Number of frames: {num_frames}")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

if fps <= 0: raise RuntimeError("Could not determine video FPS.")

input_dir = os.path.dirname(VIDEO_PATH)
input_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
MASK_VIDEO_PATH = os.path.join(input_dir, f"{input_name}_mask.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
mask_writer = cv2.VideoWriter(MASK_VIDEO_PATH, fourcc, fps, (video_width, video_height), False)

if not mask_writer.isOpened(): raise RuntimeError(f"Could not open mask video for writing: {MASK_VIDEO_PATH}")

predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": TEXT_PROMPT, "output_prob_thresh": OUTPUT_PROB_THRESHOLD})

all_frames = []

for response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id, "propagation_direction": "forward", "start_frame_index": 0, "max_frame_num_to_track": None, "output_prob_thresh": OUTPUT_PROB_THRESHOLD}):
    frame_index = int(response["frame_index"])
    outputs = response["outputs"]
    boxes = outputs["out_boxes_xywh"]
    scores = outputs["out_probs"]
    object_ids = outputs["out_obj_ids"]
    frame_mask = np.zeros((video_height, video_width), dtype=np.uint8)
    frame_boxes = []
    for object_id, box, score in zip(object_ids, boxes, scores):
        x, y, w, h = map(float, box)
        x *= video_width
        y *= video_height
        w *= video_width
        h *= video_height
        x1 = max(0, min(video_width, int(round(x))))
        y1 = max(0, min(video_height, int(round(y))))
        x2 = max(0, min(video_width, int(round(x + w))))
        y2 = max(0, min(video_height, int(round(y + h))))
        if x2 <= x1 or y2 <= y1: continue
        frame_mask[y1:y2, x1:x2] = 255
        frame_boxes.append({"object_id": int(object_id), "bbox_xyxy": [x1, y1, x2, y2], "bbox_xywh": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    mask_writer.write(frame_mask)
    all_frames.append({"frame_index": frame_index, "boxes": frame_boxes})
    print(f"\rProcessing frame {frame_index + 1}/{num_frames}", end="")

mask_writer.release()

print("\nFinished propagation.")

predictor.handle_request({"type": "close_session", "session_id": session_id})

output = { "video": VIDEO_PATH, "mask_video": MASK_VIDEO_PATH, "prompt": TEXT_PROMPT, "width": video_width, "height": video_height, "fps": fps, "frames": all_frames, }

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"Bounding boxes saved to: {OUTPUT_JSON}")
print(f"Binary mask video saved to: {MASK_VIDEO_PATH}")