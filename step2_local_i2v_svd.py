# file: step2_local_i2v_svd.py
import os
from pathlib import Path
import math
import numpy as np
import torch
import imageio.v3 as iio
import cv2
from diffusers import StableVideoDiffusionPipeline
from tqdm import tqdm

# ---- Config (tune these for quality / speed / VRAM) ----
IMAGES_DIR = Path("1_extracted_images")
OUTPUT_DIR = Path("2_animated_clips")
MODEL_ID = "stabilityai/stable-video-diffusion-img2vid"  # HF repo
NUM_FRAMES = 25            # 14 or 25 are common presets
FPS = 12                   # 6~24; align with your Step 3 expectations
HEIGHT, WIDTH = 576, 1024  # 16:9 or 9:16; keep even numbers
GUIDANCE_SCALE = 1.2       # motion strength; 1.0~1.5 is a good start
DECODE_CHUNK_SIZE = 2      # low-memory trick
SEED = 1234                # set None for randomness

def load_image(path, target_h, target_w):
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return img

def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, variant="fp16" if dtype==torch.float16 else None
    )

    if device == "cuda":
        pipe.to(device)
        # Low-memory tips from diffusers docs:
        pipe.enable_model_cpu_offload()
        pipe.unet.enable_forward_chunking()
    else:
        print("Warning: CUDA not available; generation will be very slow.")

    generator = None
    if SEED is not None:
        generator = torch.Generator(device=device).manual_seed(SEED)

    images = sorted(IMAGES_DIR.glob("image_*.png"))
    if not images:
        print("No images found in 1_extracted_images/. Please put PNGs there.")
        return

    for img_path in tqdm(images, desc="I2V(SVD)"):
        out_name = OUTPUT_DIR / f"{img_path.stem}.mp4"
        if out_name.exists():
            continue

        image = load_image(str(img_path), HEIGHT, WIDTH)
        result = pipe(
            image=image,
            num_frames=NUM_FRAMES,
            decode_chunk_size=DECODE_CHUNK_SIZE,
            generator=generator,
            guidance_scale=GUIDANCE_SCALE,
            fps=FPS
        )
        frames = result.frames[0]  # list of PIL or np arrays
        frames_np = [np.asarray(f) for f in frames]

        # Write mp4 via imageio-ffmpeg (x264)
        iio.imwrite(
            out_name,
            frames_np,
            fps=FPS,
            codec="libx264",
            quality=8
        )

if __name__ == "__main__":
    main()
