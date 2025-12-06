# Burning Velocity Processing Pipeline

## Overview
This project processes high-speed flame videos to estimate burning velocity using two main scripts:

1. **`video_processing.py`** – isolates flame views, selects an optimal frame, and computes initial velocity.
2. **`3d_reconstruction.py`** – reconstructs a 3D flame surface from silhouettes and calculates corrected burning velocity.

Intermediate data is written to `Working/<video_stem>/`, and summary outputs are stored in CSV files.

---

## Installation

### Clone the repository
```bash
git clone <repo-url>
cd <repo-folder>
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### 1. Video Preprocessing
Run:
```bash
python video_processing.py
```

When prompted, enter the name of a video located in:

```
Videos/Raw Video/
```

Include the `.mp4` extension (e.g., `GX010283_trimmed.mp4`).

The preprocessing script will:

- Threshold and isolate flame regions.
- Detect three flame blobs (left, center, right).
- Compute blob geometry for every frame.
- Identify the *optimal alignment frame*.
- Compute pixel-to-physical scaling and initial centroid velocity.
- Create a folder under `Working/<video_stem>/`.
- Append results needed for reconstruction to `optimal_frame_summary.csv`.

---

### 2. 3D Reconstruction
Run:
```bash
python 3d_reconstruction.py
```

When prompted, enter the video name **without** `.mp4`, for example:
```
GX010283_trimmed
```

The reconstruction script will:

- Load preprocessing outputs for the selected video.
- Extract contours for all flame views at the optimal frame.
- Scale and vertically align mirror views to match the center view.
- Project silhouettes onto a cylindrical surface.
- Perform voxel carving using the most informative views (based on fill ratio).

You will see three visualizations:

1. **Cylinder projections of the three views**
2. **Voxel carving / intersection visualization**
3. **Reconstructed flame surface (height map)**

The script then computes:

- Flame surface area (cm²)
- Corrected burning velocity (cm/s)

These results are appended to:

```
BV_output.csv
```

---

## Output Structure

### Per-video directory
```
Working/<video_stem>/
```
Contains:
- Annotated video frames  
- Blob bounding-box CSV  
- Alignment-score plot  
- Enhanced optimal frame  

### Global output files
- **`optimal_frame_summary.csv`** – summary of preprocessing results  
- **`BV_output.csv`** – reconstructed flame surface area and corrected burning velocity  

---

