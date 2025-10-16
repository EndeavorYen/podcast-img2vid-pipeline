import torch
import numpy as np
import imageio.v3 as iio
import cv2
from pathlib import Path
from diffusers import StableVideoDiffusionPipeline
from tqdm import tqdm

def load_image(path, target_h, target_w):
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return img

def generate_videos(
    model_id: str = "stabilityai/stable-video-diffusion-img2vid",
    height: int = 576,
    width: int = 1024,
    num_frames: int = 25,
    fps: int = 12,
    decode_chunk_size: int = 2,
    seed: int = 1234,
):
    """
    STEP 2: Generate video clips from images using Stable Video Diffusion.
    """
    print("\n--- STEP 2: Generating videos from images ---")

    images_dir = Path("1_extracted_images")
    output_dir = Path("2_animated_clips")
    output_dir.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    )

    if device == "cuda":
        pipe.to(device)
        pipe.enable_model_cpu_offload()
        pipe.unet.enable_forward_chunking()
    else:
        print("Warning: CUDA not available; generation will be very slow.")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    images = sorted(images_dir.glob("image_*.png"))
    if not images:
        print("No images found in '1_extracted_images/'. Please run Step 1 first.")
        return

    for img_path in tqdm(images, desc="I2V(SVD)"):
        out_name = output_dir / f"{img_path.stem}.mp4"
        if out_name.exists():
            print(f"Clip for {img_path.name} already exists, skipping.")
            continue

        image = load_image(str(img_path), height, width)
        result = pipe(
            image=image,
            num_frames=num_frames,
            decode_chunk_size=decode_chunk_size,
            generator=generator,
            fps=fps,
        )
        frames = result.frames[0]
        frames_np = [np.asarray(f) for f in frames]

        iio.imwrite(
            out_name,
            frames_np,
            fps=fps,
            codec="libx264",
            quality=8,
        )

    print("\n--- STEP 2 Done ---")
    print(f"✅ Generated {len(images)} clips in '{output_dir}'.")