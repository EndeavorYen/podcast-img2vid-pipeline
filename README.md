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
├── animate_podcast_v2.py      # Main pipeline script
├── modules/                   #
│   ├── step1_scene_detection.py # Module for scene detection
│   ├── step2_i2v.py             # Module for image-to-video
│   └── step3_recombine.py       # Module for video recombination
├── 001.mp4                    # Your original podcast video (example)
├── 1_extracted_images/        # Output of Step 1
├── 2_animated_clips/          # Output of Step 2
├── requirements.txt           # Python dependencies
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

The entire pipeline is now controlled by `animate_podcast_v2.py`. You can run all steps at once or specify which ones to execute.

### Example 1: Run the full pipeline

This command runs all three steps in sequence: Scene Detection -> I2V Generation -> Recombination.
```bash
python animate_podcast_v2.py "001.mp4"
```

### Example 2: Run specific steps

If you want to re-run only the video generation and recombination steps:
```bash
python animate_podcast_v2.py "001.mp4" --steps "2,3"
```

### Command-Line Arguments

-   `video_file`: Path to your source video. (Required)
-   `--steps`: Comma-separated list of steps to run (e.g., `1,2,3`). Default is all steps.

**Step 1: Scene Detection**
-   `-t`, `--threshold`: Scene detection sensitivity (lower = more sensitive). Default: `30`.

**Step 2: Image-to-Video Generation**
-   `--height`, `--width`: Output video resolution. Default: `576x1024`.
-   `--num_frames`: Number of frames per clip. Default: `25`.
-   `--fps`: Frames per second for generated clips. Default: `12`.
-   `--guidance_scale`: Motion intensity. Default: `1.2`.
-   `--seed`: A fixed value for reproducible results. Default: `1234`.

**Step 3: Video Recombination**
-   `-p`, `--policy`: Strategy for short clips (`loop` or `freeze`). Default: `loop`.

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