import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

video_path = r"H:\My Drive\Final Project\Videos\Raw Video\GX010312_trimmed.mp4"
output_video = r"H:\My Drive\Final Project\Videos\Processed Videos\flame_with_framenumber.mp4"
alignment_video = r"H:\My Drive\Final Project\Videos\Processed Videos\alignment_example.mp4"
out_csv = r"H:\My Drive\Final Project\Data\blob_bounding_boxes.csv"

MIN_AREA = 50
SAT_THRESH = 80
BLUE_THRESH = 10

def classify_blob(contour):
    """Return area, perimeter, angle, shape classification"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    angle = None
    shape = "unknown"

    if len(contour) >= 5:
        (_, _), (MA, ma), angle = cv2.fitEllipse(contour)
        # Dome if nearly vertical
        if 80 < angle < 100:
            shape = "dome"
        else:
            # Determine open direction (C left or right) via centroid symmetry
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                xs = contour[:,0,0]
                if np.sum(xs < cx) > np.sum(xs > cx):
                    shape = "C-left"
                else:
                    shape = "C-right"
    return area, perimeter, angle, shape

# Storage (added x,y)
data = {
    "L": {"frame": [], "area": [], "perimeter": [], "angle": [], "shape": [],
          "x": [], "y": [], "w": [], "h": [], "cx": [], "cy": []},
    "C": {"frame": [], "area": [], "perimeter": [], "angle": [], "shape": [],
          "x": [], "y": [], "w": [], "h": [], "cx": [], "cy": []},
    "R": {"frame": [], "area": [], "perimeter": [], "angle": [], "shape": [],
          "x": [], "y": [], "w": [], "h": [], "cx": [], "cy": []},
}

# --- pass 1: process video and collect data ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_idx = 0
frames_to_write = []
alignment_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- flame mask ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    _, mask_s = cv2.threshold(s, SAT_THRESH, 255, cv2.THRESH_BINARY)

    b, g, r = cv2.split(frame)
    _, mask_b = cv2.threshold(b, BLUE_THRESH, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask_s, mask_b)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]

    alignment_frame = frame.copy()
    
    # Brighten blue pixels above threshold
    b, g, r = cv2.split(alignment_frame)
    blue_mask = b > BLUE_THRESH
    b[blue_mask] = np.clip(b[blue_mask] * 1.5 + 30, 0, 255).astype(np.uint8)
    alignment_frame = cv2.merge([b, g, r])

    if len(contours) >= 3:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        labels = ["L", "C", "R"]
        colors_vis = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR

        for label, c, color in zip(labels, contours, colors_vis):
            area, perimeter, angle, shape = classify_blob(c)
            x, y, w, h = cv2.boundingRect(c)

            # Centroid
            M = cv2.moments(c)
            cx, cy = None, None
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

            # Store metrics
            data[label]["frame"].append(frame_idx)
            data[label]["area"].append(area)
            data[label]["perimeter"].append(perimeter)
            data[label]["angle"].append(angle)
            data[label]["shape"].append(shape)
            data[label]["x"].append(x)
            data[label]["y"].append(y)
            data[label]["w"].append(w)
            data[label]["h"].append(h)
            data[label]["cx"].append(cx)
            data[label]["cy"].append(cy)

            # Draw on alignment frame
            cv2.drawContours(alignment_frame, [c], -1, color, 2)
            cv2.rectangle(alignment_frame, (x, y), (x+w, y+h), color, 1)
            if cx is not None:
                cv2.circle(alignment_frame, (cx, cy), 5, color, -1)
                cv2.putText(alignment_frame, label, (cx-10, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            if len(c) >= 5:
                ellipse = cv2.fitEllipse(c)
                cv2.ellipse(alignment_frame, ellipse, color, 1)

    # overlay frame number
    text = f"Frame {frame_idx}"
    cv2.putText(frame, text, (width-300, height-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(alignment_frame, text, (width-300, height-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    
    frames_to_write.append(frame)
    alignment_frames.append(alignment_frame)

    frame_idx += 1

cap.release()

# --- save annotated video ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 5, (width, height))
for f in frames_to_write:
    out.write(f)
out.release()
print(f"Video saved to {output_video}")

# --- save alignment video ---
out_align = cv2.VideoWriter(alignment_video, fourcc, 5, (width, height))
for f in alignment_frames:
    out_align.write(f)
out_align.release()
print(f"Alignment video saved to {alignment_video}")

# --- save CSV ---
os.makedirs(os.path.dirname(out_csv), exist_ok=True)

records = []
for label in ["L", "C", "R"]:
    for i in range(len(data[label]["frame"])):
        x = data[label]["x"][i]
        w = data[label]["w"][i]
        records.append({
            "label": label,
            "frame": data[label]["frame"][i],
            "area": data[label]["area"][i],
            "perimeter": data[label]["perimeter"][i],
            "angle": data[label]["angle"][i],
            "shape": data[label]["shape"][i],
            "x": x,
            "y": data[label]["y"][i],
            "w": w,
            "h": data[label]["h"][i],
            "cx": data[label]["cx"][i],
            "cy": data[label]["cy"][i],
            # new edge fields
            "uL": x,
            "uR": x + w
        })

df = pd.DataFrame.from_records(records)
df.to_csv(out_csv, index=False)
print(f"Bounding box CSV saved to {out_csv}")

# --- plot 3x3 grid ---
fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
metrics = ["angle", "perimeter", "area"]
titles = ["Angle (deg)", "Perimeter (px)", "Area (px²)"]

color_map_angle = {"C-left": "blue", "dome": "green", "C-right": "red"}

for row, label in enumerate(["L", "C", "R"]):
    for col, metric in enumerate(metrics):
        ax = axes[row, col]
        if metric == "angle":
            colors = [color_map_angle.get(s, "gray") for s in data[label]["shape"]]
        else:
            colors = "black"
        ax.scatter(data[label]["frame"], data[label][metric], c=colors, s=8)
        if row == 0:
            ax.set_title(titles[col])
        if col == 0:
            ax.set_ylabel(f"{label} blob")
        if row == 2:
            ax.set_xlabel("Frame #")

plt.tight_layout()
plt.show()
