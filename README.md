# Podcast Image-to-Video Pipeline

This project automates the process of converting static-image-based podcast videos into dynamic ones.

The pipeline follows these steps:
1.  **Scene & Image Extraction**: Extracts key scenes and their representative images.
2.  **Local Image-to-Video (I2V)**: Uses Stable Video Diffusion to generate video clips from images locally.
3.  **Reassembly & Audio Sync**: Recombines the video clips according to the original timestamps and merges the original audio.

This is ideal for educational or children's podcasts that use a series of static illustrations, bringing the visuals to life and enhancing the viewing experience.

## Features

-   **Scene Detection**: Uses `PySceneDetect` to extract scenes and their timestamps into a `timestamps.csv` file.
-   **Local Image-to-Video**: Runs Stable Video Diffusion (via Diffusers) on your local machine (Windows/WSL2), avoiding per-second processing fees.
-   **Audio-Visual Sync**: Automatically aligns generated clips to the original audio duration. Shorter clips can be extended using looping or freezing frames.
-   **One-Click Reassembly**: Batch-processes and stitches all dynamic clips into a final H.264 + AAC MP4 file.

## Directory Layout

```
repo-root/
├── animate_podcast_v2.py      # Step 1 & 3 (Scene Extraction / Reassembly)
├── step2_local_i2v_svd.py     # Step 2 (Local I2V: Stable Video Diffusion)
├── 001.mp4                    # Your original podcast video (example)
├── 1_extracted_images/        # Output of Step 1 (image_001.png, ...)
├── 2_animated_clips/          # Output of Step 2 (image_001.mp4, ...)
├── requirements.txt           # Optional Python dependencies
└── README.md
```

## Environment Setup

### A. Windows 11 (Native)

1.  Install Python 3.10/3.11 (64-bit), ensuring you check "Add Python to PATH" during installation.
2.  Install [FFmpeg](https://ffmpeg.org/download.html) and add its `bin` directory to your system's `PATH`.
3.  Create a virtual environment:
    ```bash
    cd <your repo>
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
4.  Install PyTorch (CUDA 12.4 version) and other dependencies:
    ```bash
    pip install -U pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install diffusers transformers accelerate safetensors imageio-ffmpeg opencv-python tqdm pandas "scenedetect[opencv]"
    ```
5.  Verify GPU setup:
    ```python
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0), "CUDA:", torch.version.cuda)
    ```
6.  Log in to Hugging Face to download model weights. You may need to accept the terms on the [model page](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) first.
    ```bash
    huggingface-cli login
    ```

### B. WSL2 (Ubuntu 22.04/24.04)

> **Note:** Ensure your NVIDIA drivers are up-to-date on the Windows host and run `wsl --update`. You do not need to install Linux-specific drivers or the CUDA Toolkit inside WSL.

1.  Install prerequisites:
    ```bash
    sudo apt update
    sudo apt install -y python3-venv python3-pip git ffmpeg
    ```
2.  Set up the project and virtual environment:
    ```bash
    mkdir -p ~/podcast-i2vid && cd ~/podcast-i2vid # Or your repo path
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -U pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install diffusers transformers accelerate safetensors imageio-ffmpeg opencv-python tqdm pandas "scenedetect[opencv]"
    ```
4.  Verify GPU setup:
    ```python
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), "CUDA:", torch.version.cuda)
    ```
5.  Log in to Hugging Face:
    ```bash
    huggingface-cli login
    ```
> **Performance Tip:** For better I/O performance, store your project files within the WSL filesystem (e.g., `~/podcast-i2vid`) rather than on a mounted Windows drive (e.g., `/mnt/c`).

## How to Use

### Step 1: Scene Detection & Image Extraction

Run the following command in your activated virtual environment (PowerShell or Bash):
```bash
python animate_podcast_v2.py "001.mp4" --step 1 -t 30
```
This will generate:
-   `1_extracted_images/image_001.png`, `image_002.png`, ...
-   `timestamps.csv` (contains start time, end time, and duration for each scene).

The `-t`/`--threshold` parameter controls sensitivity (default is `30`). A lower value is more sensitive to scene changes.

### Step 2: Local Image-to-Video Conversion

This step converts each extracted image into a short video clip.
```bash
python step2_local_i2v_svd.py
```
-   **Model**: Uses `stabilityai/stable-video-diffusion-img2vid` by default.
-   **Input**: Reads `*.png` files from `1_extracted_images/`.
-   **Output**: Saves `*.mp4` files to `2_animated_clips/`.

You can adjust key parameters inside the script:
-   `HEIGHT`, `WIDTH`: Resolution (e.g., `576x1024`). Higher values require more VRAM and time.
-   `NUM_FRAMES`: Number of frames to generate (e.g., `14` or `25`).
-   `FPS`: Frames per second (e.g., `8`–`12`).
-   `GUIDANCE_SCALE`: Motion intensity (e.g., `1.0`–`1.5`).
-   `DECODE_CHUNK_SIZE`: A memory-saving trick; set to `1` or `2` on low-VRAM GPUs.
-   `SEED`: A fixed value for reproducible results.

### Step 3: Reassembly and Audio Sync

This step combines the generated clips and syncs them with the original audio.
```bash
python animate_podcast_v2.py "001.mp4" --step 3 -p loop
```
The `-p`/`--padding_strategy` can be `loop` or `freeze` to fill gaps if a video clip is shorter than its scene duration.

The script will:
1.  Read `timestamps.csv` and the video clips from `2_animated_clips/`.
2.  Apply the padding strategy (`loop`/`freeze`) to match each clip's duration to its timestamp.
3.  Normalize resolutions.
4.  Concatenate all clips and merge them with the original audio track.
5.  Output the final video to `001_animated_v2.mp4`.

## Optional `requirements.txt`

For convenience, you can create a `requirements.txt` file. Note that PyTorch should be installed separately based on your platform and CUDA version.
```
# requirements.txt
diffusers
transformers
accelerate
safetensors
imageio-ffmpeg
opencv-python
tqdm
pandas
scenedetect
moviepy
```

## Troubleshooting

-   **`CUDA available: False`**:
    -   On Windows, verify `nvidia-smi` runs correctly.
    -   If using WSL, run `wsl --update`.
    -   Re-activate your virtual environment.
    -   Re-install the correct PyTorch version for your CUDA driver.

-   **Hugging Face `403 Forbidden` Error**:
    -   Run `huggingface-cli login` again.
    -   Ensure you have accepted the terms on the model's Hugging Face page.

-   **Out of Memory (OOM) Errors**:
    -   Lower the `HEIGHT`/`WIDTH`, `NUM_FRAMES`, or `FPS`.
    -   Set `DECODE_CHUNK_SIZE=1`.
    -   Close other GPU-intensive applications (including hardware-accelerated browsers).

-   **Incorrect Output Duration or Stuttering**:
    -   Step 3 aligns the video to the audio length by default. If the video is shorter, it pads the end with a frozen frame.

-   **Slow Performance**:
    -   Test your pipeline with low-quality settings first to ensure it works.
    -   Use a fixed `SEED` to compare quality changes between runs.

-   **File Naming Mismatches**:
    -   The pipeline requires a strict `image_001.png` -> `image_001.mp4` naming convention.

## Roadmap

-   [ ] Add CLI arguments for Step 2 (`--height`, `--width`, `--num-frames`, etc.).
-   [ ] Create a `Provider` interface to support cloud-based APIs (Runway, Pika, Luma) as fallbacks.
-   [ ] Implement post-processing for Ken Burns effects (pan/zoom).
-   [ ] Add drift correction to ensure audio-visual sync error is < 10ms.

## License

This project is licensed under the MIT License.

## Disclaimer

-   Ensure you have the legal rights to use all image and vector assets.
-   If your content involves children, comply with all relevant platform policies and legal regulations.
-   Usage of third-party models is subject to their respective licenses (e.g., Stable Video Diffusion, Diffusers).

## Acknowledgements

-   [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
-   [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
-   [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
-   [MoviePy](https://github.com/Zulko/moviepy)