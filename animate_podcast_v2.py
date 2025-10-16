import argparse
import pandas as pd
from pathlib import Path
import math

# MoviePy v2 imports (no more `moviepy.editor`)
from moviepy import VideoFileClip, concatenate_videoclips, ImageClip

# PySceneDetect for accurate scene detection
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images


# -------- Step 1: Detect scenes & export representative images --------
def find_scenes(video_path: Path, threshold: int = 30):
    """
    STEP 1 (enhanced): Use PySceneDetect to detect scene boundaries and
    export one representative image per scene. Also write timestamps.csv.
    """
    print("--- STEP 1 (Enhanced): Precise scene detection ---")
    video = open_video(str(video_path))
    scene_manager = SceneManager()

    # ContentDetector is robust to content-level changes.
    # Lower threshold => more sensitive. 27~30 fits most cases.
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Run detection
    scene_manager.detect_scenes(video, show_progress=True)

    # Get list of (start, end) timecodes
    scene_list = scene_manager.get_scene_list()

    output_images_dir = Path("1_extracted_images")
    output_images_dir.mkdir(exist_ok=True)

    if not scene_list:
        print("Warning: No scene change detected. Treat whole video as one scene.")
        # Fallback to single scene covering the whole video.
        scene_list = [(video.base_timecode, video.duration)]

    print(f"Detected {len(scene_list)} scenes.")

    # Save the first frame of each scene as representative image.
    # Force PNG to match timestamps.csv entries.
    save_images(
        scene_list=scene_list,
        video=video,
        num_images=1,
        output_dir=str(output_images_dir),
        image_name_template='image_$SCENE_NUMBER',  # image_001, image_002, ...
        image_extension='png'
    )

    # Build timestamps.csv (use .png filenames)
    timestamps = []
    for i, (start, end) in enumerate(scene_list):
        start_s = start.get_seconds()
        end_s = end.get_seconds()
        dur_s = max(0.0, end_s - start_s)  # guard against rare negative
        timestamps.append({
            "image_file": f"image_{i+1:03d}.png",
            "start_time": start_s,
            "end_time": end_s,
            "duration": dur_s,
        })

    timestamps_csv_path = Path("timestamps.csv")
    df = pd.DataFrame(timestamps)
    df.to_csv(timestamps_csv_path, index=False, float_format="%.4f")

    print("\n--- STEP 1 Done ---")
    print(f"✅ Extracted {len(timestamps)} images -> '{output_images_dir}'.")
    print(f"✅ Wrote timestamps -> '{timestamps_csv_path}'.")
    return timestamps_csv_path


# -------- Utility: Fit a clip to the target duration (loop/freeze) --------
def fit_duration(clip: VideoFileClip, target_sec: float, policy: str = "loop") -> VideoFileClip:
    """
    Ensure `clip` duration >= target_sec, then trim to exactly target_sec.
    policy: 'loop' (repeat) or 'freeze' (freeze last frame).
    """
    # If already long enough, just trim precisely
    if clip.duration >= target_sec:
        return clip.subclipped(0, target_sec)  # v2 API

    # Need to extend
    if policy == "loop":
        # Repeat the clip enough times, then trim to exact length
        times = max(2, math.ceil(target_sec / max(clip.duration, 1e-6)))
        looped = concatenate_videoclips([clip] * times, method="compose")
        return looped.subclipped(0, target_sec)

    elif policy == "freeze":
        # Take the last frame and freeze it
        fps = clip.fps or 30.0
        t_last = max(0.0, clip.duration - (1.0 / fps))
        last_frame = clip.get_frame(t_last)
        tail = ImageClip(last_frame).with_duration(target_sec - clip.duration)
        return concatenate_videoclips([clip, tail], method="compose")

    # Fallback: force duration (not preferred, but safe)
    return clip.with_duration(target_sec)


# -------- Step 3: Recombine animated clips to match timestamps --------
def recombine_videos(video_path: Path, timestamps_csv_path: Path, duration_policy: str):
    """
    STEP 3 (enhanced): Recombine I2V short clips according to timestamps.csv,
    normalize resolution, align total length to original audio, and export.
    """
    print("\n--- STEP 3 (Enhanced): Robust video recombination ---")

    animated_clips_dir = Path("2_animated_clips")
    if not animated_clips_dir.is_dir() or not any(animated_clips_dir.iterdir()):
        print(f"Error: '{animated_clips_dir}' not found or empty. Please prepare animated clips first.")
        return

    df = pd.read_csv(timestamps_csv_path)

    print("Extracting audio from original video...")
    original_video_clip = VideoFileClip(str(video_path))
    original_audio = original_video_clip.audio
    if original_audio is None:
        print("Note: Original video has NO audio track. Output will be silent.")

    # Use the source video's resolution as the target
    target_resolution = original_video_clip.size

    final_clips = []
    print(f"Reading animated clips using policy='{duration_policy}' ...")
    for _, row in df.iterrows():
        # `image_001.png` -> expect `image_001.mp4` in 2_animated_clips/
        image_basename = Path(row['image_file']).stem
        clip_path = animated_clips_dir / f"{image_basename}.mp4"

        if not clip_path.exists():
            print(f"⚠️  Missing clip: '{clip_path}', skip this segment.")
            continue

        seg_clip = VideoFileClip(str(clip_path))
        seg_dur = float(row['duration'])

        # Ensure clip duration matches segment duration
        processed = fit_duration(seg_clip, seg_dur, policy=duration_policy)

        # Normalize resolution (v2: resized)
        if processed.size != target_resolution:
            processed = processed.resized(target_resolution)

        final_clips.append(processed)

    if not final_clips:
        print("Error: No usable clips to concatenate.")
        original_video_clip.close()
        return

    print("Concatenating all segments...")
    final_video_no_audio = concatenate_videoclips(final_clips, method="compose")

    # Align total length with original audio if exists
    if original_audio is not None:
        if final_video_no_audio.duration > original_audio.duration:
            # Trim video to audio length
            final_video_no_audio = final_video_no_audio.subclipped(0, original_audio.duration)
        elif final_video_no_audio.duration < original_audio.duration:
            # Pad tail by freezing the last frame to match audio length
            pad = original_audio.duration - final_video_no_audio.duration
            last_frame = final_video_no_audio.get_frame(
                max(0.0, final_video_no_audio.duration - (1.0 / 30.0))
            )
            tail = ImageClip(last_frame).with_duration(pad)
            final_video_no_audio = concatenate_videoclips([final_video_no_audio, tail], method="compose")

    # Set audio (if any)
    final_video_with_audio = (
        final_video_no_audio.with_audio(original_audio)
        if original_audio is not None else
        final_video_no_audio
    )

    output_filename = f"{video_path.stem}_animated_v2.mp4"
    print(f"Writing output to '{output_filename}' ...")
    final_video_with_audio.write_videofile(
        output_filename,
        codec="libx264",
        audio_codec="aac",
        threads=8
    )

    # Cleanup resources
    original_video_clip.close()
    for c in final_clips:
        try:
            c.close()
        except Exception:
            pass

    print("\n--- STEP 3 Done ---")
    print(f"🎉 Finished: '{output_filename}'")


# -------- Main CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Podcast Animation Script (v2.0)")
    parser.add_argument("video_file", type=str, help="Path to original podcast video.")
    parser.add_argument("--step", type=int, choices=[1, 3], required=True,
                        help="Step to run: 1 (extract) or 3 (recombine).")

    # Step 1 params
    parser.add_argument("-t", "--threshold", type=int, default=30,
                        help="[Step 1] Scene-detection sensitivity (lower=more sensitive). Default: 30.")

    # Step 3 params
    parser.add_argument("-p", "--policy", type=str, default="loop", choices=["loop", "freeze"],
                        help="[Step 3] If a generated short clip is shorter than target duration: "
                             "'loop' to repeat, 'freeze' to freeze the last frame. Default: loop.")

    args = parser.parse_args()

    source_video_path = Path(args.video_file)
    if not source_video_path.exists():
        print(f"Error: Video file not found: {source_video_path}")
        raise SystemExit(1)

    timestamps_file = Path("timestamps.csv")

    if args.step == 1:
        find_scenes(source_video_path, threshold=args.threshold)
        print("\nNext: Run your I2V tool & put clips as MP4 into '2_animated_clips/'.")
    elif args.step == 3:
        if not timestamps_file.exists():
            print(f"Error: '{timestamps_file}' not found. Please run --step 1 first.")
            raise SystemExit(1)
        recombine_videos(source_video_path, timestamps_file, duration_policy=args.policy)
    else:
        raise SystemExit(1)