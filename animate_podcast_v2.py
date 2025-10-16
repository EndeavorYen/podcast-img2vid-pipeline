import argparse
from pathlib import Path

# Import the refactored modules
from modules.step1_scene_detection import find_scenes
from modules.step2_i2v import generate_videos
from modules.step3_recombine import recombine_videos

def main():
    parser = argparse.ArgumentParser(
        description="Unified Podcast Animation Pipeline (v3.0)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("video_file", type=str, help="Path to the original podcast video.")

    # Step selection
    parser.add_argument(
        "--steps",
        type=str,
        default="1,2,3",
        help="Comma-separated list of steps to run (e.g., '1,2,3' or '2,3').\n"
             "1: Scene Detection\n"
             "2: Image-to-Video Generation\n"
             "3: Video Recombination"
    )

    # Step 1: Scene Detection
    step1_group = parser.add_argument_group("Step 1: Scene Detection")
    step1_group.add_argument("-t", "--threshold", type=int, default=30,
                             help="Scene-detection sensitivity (lower=more sensitive). Default: 30.")

    # Step 2: Image-to-Video
    step2_group = parser.add_argument_group("Step 2: Image-to-Video Generation")
    step2_group.add_argument("--model_id", type=str, default="stabilityai/stable-video-diffusion-img2vid",
                             help="Hugging Face model ID for SVD.")
    step2_group.add_argument("--height", type=int, default=576, help="Output video height.")
    step2_group.add_argument("--width", type=int, default=1024, help="Output video width.")
    step2_group.add_argument("--num_frames", type=int, default=25, help="Number of frames to generate per clip.")
    step2_group.add_argument("--fps", type=int, default=12, help="FPS of generated clips.")
    step2_group.add_argument("--decode_chunk_size", type=int, default=2, help="Memory-saving decode chunk size.")
    step2_group.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")

    # Step 3: Recombination
    step3_group = parser.add_argument_group("Step 3: Video Recombination")
    step3_group.add_argument("-p", "--policy", type=str, default="loop", choices=["loop", "freeze"],
                             help="Policy for short clips: 'loop' or 'freeze'. Default: loop.")

    args = parser.parse_args()

    source_video_path = Path(args.video_file)
    if not source_video_path.exists():
        print(f"Error: Video file not found: {source_video_path}")
        raise SystemExit(1)

    steps_to_run = {int(s.strip()) for s in args.steps.split(',')}
    timestamps_file = Path("timestamps.csv")

    # --- Execute Pipeline Steps ---
    if 1 in steps_to_run:
        find_scenes(source_video_path, threshold=args.threshold)

    if 2 in steps_to_run:
        if not Path("1_extracted_images").is_dir():
            print("Error: '1_extracted_images' directory not found. Please run Step 1 first.")
            raise SystemExit(1)
        generate_videos(
            model_id=args.model_id,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=args.fps,
            decode_chunk_size=args.decode_chunk_size,
            seed=args.seed
        )

    if 3 in steps_to_run:
        if not timestamps_file.exists():
            print(f"Error: '{timestamps_file}' not found. Please run Step 1 first.")
            raise SystemExit(1)
        recombine_videos(source_video_path, timestamps_file, duration_policy=args.policy)

    print("\n🎉 Pipeline finished for steps: " + ", ".join(map(str, sorted(list(steps_to_run)))))

if __name__ == "__main__":
    main()