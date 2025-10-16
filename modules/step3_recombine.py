import pandas as pd
from pathlib import Path
import math
from moviepy import VideoFileClip, concatenate_videoclips, ImageClip

def fit_duration(clip: VideoFileClip, target_sec: float, policy: str = "loop") -> VideoFileClip:
    """
    Ensure `clip` duration >= target_sec, then trim to exactly target_sec.
    policy: 'loop' (repeat) or 'freeze' (freeze last frame).
    """
    if clip.duration >= target_sec:
        return clip.subclipped(0, target_sec)

    if policy == "loop":
        times = max(2, math.ceil(target_sec / max(clip.duration, 1e-6)))
        looped = concatenate_videoclips([clip] * times, method="compose")
        return looped.subclipped(0, target_sec)
    elif policy == "freeze":
        fps = clip.fps or 30.0
        t_last = max(0.0, clip.duration - (1.0 / fps))
        last_frame = clip.get_frame(t_last)
        tail = ImageClip(last_frame).with_duration(target_sec - clip.duration)
        return concatenate_videoclips([clip, tail], method="compose")

    return clip.with_duration(target_sec)

def recombine_videos(video_path: Path, timestamps_csv_path: Path, duration_policy: str):
    """
    STEP 3: Recombine I2V short clips according to timestamps.csv,
    normalize resolution, align total length to original audio, and export.
    """
    print("\n--- STEP 3: Recombining videos ---")

    animated_clips_dir = Path("2_animated_clips")
    if not animated_clips_dir.is_dir() or not any(animated_clips_dir.iterdir()):
        print(f"Error: '{animated_clips_dir}' not found or empty. Please run Step 2 first.")
        return

    df = pd.read_csv(timestamps_csv_path)

    print("Extracting audio from original video...")
    original_video_clip = VideoFileClip(str(video_path))
    original_audio = original_video_clip.audio
    if original_audio is None:
        print("Note: Original video has NO audio track. Output will be silent.")

    target_resolution = original_video_clip.size
    final_clips = []
    print(f"Reading animated clips using policy='{duration_policy}' ...")

    for _, row in df.iterrows():
        image_basename = Path(row['image_file']).stem
        clip_path = animated_clips_dir / f"{image_basename}.mp4"

        if not clip_path.exists():
            print(f"⚠️  Missing clip: '{clip_path}', skipping this segment.")
            continue

        seg_clip = VideoFileClip(str(clip_path))
        seg_dur = float(row['duration'])
        processed = fit_duration(seg_clip, seg_dur, policy=duration_policy)

        if processed.size != target_resolution:
            processed = processed.resized(target_resolution)

        final_clips.append(processed)

    if not final_clips:
        print("Error: No usable clips to concatenate.")
        original_video_clip.close()
        return

    print("Concatenating all segments...")
    final_video_no_audio = concatenate_videoclips(final_clips, method="compose")

    if original_audio is not None:
        if final_video_no_audio.duration > original_audio.duration:
            final_video_no_audio = final_video_no_audio.subclipped(0, original_audio.duration)
        elif final_video_no_audio.duration < original_audio.duration:
            pad = original_audio.duration - final_video_no_audio.duration
            last_frame = final_video_no_audio.get_frame(max(0.0, final_video_no_audio.duration - (1.0 / 30.0)))
            tail = ImageClip(last_frame).with_duration(pad)
            final_video_no_audio = concatenate_videoclips([final_video_no_audio, tail], method="compose")

    final_video_with_audio = final_video_no_audio.with_audio(original_audio) if original_audio is not None else final_video_no_audio

    output_filename = f"{video_path.stem}_animated_v3.mp4"
    print(f"Writing output to '{output_filename}' ...")
    final_video_with_audio.write_videofile(
        output_filename,
        codec="libx264",
        audio_codec="aac",
        threads=8
    )

    original_video_clip.close()
    for c in final_clips:
        try:
            c.close()
        except Exception:
            pass

    print("\n--- STEP 3 Done ---")
    print(f"🎉 Finished: '{output_filename}'")