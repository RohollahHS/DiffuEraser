import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor, Sam3VideoConfig

# config = Sam3VideoConfig.from_pretrained("weights/sam3")
# config.image_size = 560
# model = Sam3VideoModel.from_pretrained("weights/sam3", config=config, device_map="auto")

model = Sam3VideoModel.from_pretrained("weights/sam3", device_map="auto")
processor = Sam3VideoProcessor.from_pretrained("weights/sam3")

# Load video frames
from transformers.video_utils import load_video
video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
video_frames, _ = load_video(video_url)
num_frames = len(video_frames)

# Initialize video inference session
inference_session = processor.init_video_session(video=video_frames, inference_device="cuda", processing_device="cpu", video_storage_device="cpu")

# Add text prompt to detect and track objects
text = "person"
inference_session = processor.add_text_prompt(inference_session=inference_session, text=text)

# Process all frames in the video
outputs_per_frame = {}
# Pass show_progress_bar=True to display a tqdm progress bar.
for model_outputs in model.propagate_in_video_iterator(inference_session=inference_session, max_frame_num_to_track=num_frames):
    processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
    outputs_per_frame[model_outputs.frame_idx] = processed_outputs

print(f"Processed {len(outputs_per_frame)} frames")

# Access results for a specific frame
frame_0_outputs = outputs_per_frame[0]
print(f"Detected {len(frame_0_outputs['object_ids'])} objects")
print(f"Object IDs: {frame_0_outputs['object_ids'].tolist()}")
print(f"Scores: {frame_0_outputs['scores'].tolist()}")
print(f"Boxes shape (XYXY format, absolute coordinates): {frame_0_outputs['boxes'].shape}")
print(f"Masks shape: {frame_0_outputs['masks'].shape}")