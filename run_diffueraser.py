
import os
import gc
import time
import argparse

import torch

from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device


def main():
    # Input parameters
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_video',
        type=str,
        nargs='+',
        default=["examples/example3/video.mp4"],
        help='One or more paths to input videos'
    )
    parser.add_argument(
        '--input_mask',
        type=str,
        nargs='+',
        default=["examples/example3/mask.mp4"],
        help='One or more paths to input masks, corresponding to the videos'
    )
    parser.add_argument(
        '--video_length',
        type=int,
        default=None,
        help='The maximum length of output video'
    )
    parser.add_argument(
        '--mask_dilation_iter',
        type=int,
        default=8,
        help='Adjust it to change the degree of mask expansion'
    )
    parser.add_argument(
        '--max_img_size',
        type=int,
        default=None,
        help='The maximum length of output width and height'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default="/scratch/rohhs/downloads/diffueraser",
        help='Path to the output directory'
    )
    parser.add_argument('--ref_stride', type=int, default=10,
                        help='Propainter params')
    parser.add_argument('--neighbor_length', type=int, default=10,
                        help='Propainter params')
    parser.add_argument('--subvideo_length', type=int, default=50,
                        help='Propainter params')
    parser.add_argument(
        '--base_model_path',
        type=str,
        default="weights/stable-diffusion-v1-5",
        help='Path to sd1.5 base model'
    )
    parser.add_argument(
        '--vae_path',
        type=str,
        default="weights/sd-vae-ft-mse",
        help='Path to vae'
    )
    parser.add_argument(
        '--diffueraser_path',
        type=str,
        default="weights/diffuEraser",
        help='Path to DiffuEraser'
    )
    parser.add_argument(
        '--propainter_model_dir',
        type=str,
        default="weights/propainter",
        help='Path to Propainter model'
    )

    args = parser.parse_args()

    # Make sure every video has a corresponding mask
    if len(args.input_video) != len(args.input_mask):
        parser.error(
            f"Number of videos ({len(args.input_video)}) must equal "
            f"number of masks ({len(args.input_mask)})."
        )

    os.makedirs(args.save_path, exist_ok=True)

    # Prepare output paths for each video-mask pair
    jobs = []

    for video_path, mask_path in zip(args.input_video, args.input_mask):
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        priori_path = os.path.join(
            args.save_path, video_name + "_priori.mp4"
        )
        output_path = os.path.join(
            args.save_path, video_name + "_diffueraser.mp4"
        )

        jobs.append({
            "video": video_path,
            "mask": mask_path,
            "priori": priori_path,
            "output": output_path,
        })

    # Initialize models once
    device = get_device()
    ckpt = "2-Step"

    video_inpainting_sd = DiffuEraser(
        device,
        args.base_model_path,
        args.vae_path,
        args.diffueraser_path,
        ckpt=ckpt
    )

    propainter = Propainter(
        args.propainter_model_dir,
        device=device
    )

    start_time = time.time()

    # Stage 1: Generate priori videos for all input pairs
    for index, job in enumerate(jobs, start=1):
        print(
            f"\n[{index}/{len(jobs)}] Running Propainter "
            f"for: {job['video']}"
        )

        propainter.forward(
            job["video"],
            job["mask"],
            job["priori"],
            video_length=args.video_length,
            ref_stride=args.ref_stride,
            neighbor_length=args.neighbor_length,
            subvideo_length=args.subvideo_length,
            mask_dilation=args.mask_dilation_iter
        )

    # Free Propainter memory before DiffuEraser inference
    del propainter
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Stage 2: Generate final inpainted videos
    guidance_scale = None  # Default guidance scale is 0

    for index, job in enumerate(jobs, start=1):
        print(
            f"\n[{index}/{len(jobs)}] Running DiffuEraser "
            f"for: {job['video']}"
        )

        video_inpainting_sd.forward(
            job["video"],
            job["mask"],
            job["priori"],
            job["output"],
            max_img_size=args.max_img_size,
            video_length=args.video_length,
            mask_dilation_iter=args.mask_dilation_iter,
            guidance_scale=guidance_scale
        )

        print(f"Output saved to: {job['output']}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    end_time = time.time()
    inference_time = end_time - start_time

    print(f"\nProcessed {len(jobs)} video(s).")
    print(f"Total DiffuEraser pipeline time: {inference_time:.4f} s")

    del video_inpainting_sd
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
