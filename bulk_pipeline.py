import sys
import subprocess
from pathlib import Path


# Used to run batches of videos

# base folders
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "Videos" / "Raw Video"

# scripts to run
PREPROCESS_SCRIPT = PROJECT_ROOT / "video_processing.py"
RECON_SCRIPT = PROJECT_ROOT / "3d_reconstruction.py"


def run_script(script_path, filename):
    # run a script and pass the filename
    subprocess.run(
        [sys.executable, str(script_path)],
        input=filename + "\n",
        text=True,
        check=True,
    )


def main():
    # check raw video folder
    if not RAW_DIR.exists():
        print("Raw Video directory not found.")
        return

    # all mp4 files
    videos = sorted(
        p for p in RAW_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".mp4"
    )
    if not videos:
        print("No mp4 files found.")
        return

    # run both scripts for each file
    for video in videos:
        name = video.name
        print(f"Processing {name}")

        try:
            run_script(PREPROCESS_SCRIPT, name)
            run_script(RECON_SCRIPT, name)
            print(f"Finished {name}")
        except subprocess.CalledProcessError as e:
            print(f"Error on {name}: {e}")
            print("Skipping this file.")

    print("Done.")


if __name__ == "__main__":
    main()
