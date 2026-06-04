import time
from pathlib import Path
from annotate_clip_gemini_updated import annotate_video

# --- CONFIGURATION ---
# Users: Update these paths to match your local repository structure
INPUT_DIR = Path("videos/all_videos")
OUTPUT_DIR = Path("predictions_gemini3pro")

def main():
    # 1. Validation
    if not INPUT_DIR.exists():
        print(f"Error: Input directory not found: {INPUT_DIR}")
        print("Please check the folder path in the script.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Get Video Files
    raw_files = list(INPUT_DIR.glob("*.mp4"))
    try:
        video_files = sorted(raw_files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    except Exception:
        video_files = sorted(raw_files)

    total_clips = len(video_files)
    print(f"\nFound {total_clips} videos in {INPUT_DIR}")
    print(f"Saving predictions to {OUTPUT_DIR}\n")

    # 3. Batch Loop
    success_count = 0
    total_start = time.time()

    for i, video_path in enumerate(video_files, 1):
        output_file = OUTPUT_DIR / f"{video_path.stem}_gemini.json"

        print(f"[{i}/{total_clips}] Processing {video_path.name}...", end=" ", flush=True)

        if output_file.exists():
            print("Skipped (Already exists).")
            continue

        try:
            result_json = annotate_video(video_path)

            with open(output_file, "w") as f:
                f.write(result_json)

            print("Done.")
            success_count += 1

            # API rate limit buffer
            time.sleep(1)

        except Exception as e:
            print(f"\n FAILED: {e}")
            with open(str(output_file) + ".error", "w") as f:
                f.write(str(e))

    # 4. Summary
    total_time = time.time() - total_start
    print("\n=================== SUMMARY ===================")
    print(f"Successfully processed: {success_count}/{total_clips}")
    print(f"Total runtime: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print("================================================")

if __name__ == "__main__":
    main()