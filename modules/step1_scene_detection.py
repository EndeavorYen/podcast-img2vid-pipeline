import pandas as pd
from pathlib import Path
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images


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