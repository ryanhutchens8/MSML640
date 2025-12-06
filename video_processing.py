import cv2
import numpy as np
import pandas as pd
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# base folder for files
PROJECT_ROOT = Path(__file__).resolve().parent

# thresholds
MIN_AREA = 50
SAT_THRESH = 80
BLUE_THRESH = 10


def prep_frame(frame):
    # basic preprocessing to highlight the blue flame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    _, mask_s = cv2.threshold(s, SAT_THRESH, 255, cv2.THRESH_BINARY)

    b, g, r = cv2.split(frame)
    _, mask_b = cv2.threshold(b, BLUE_THRESH, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_and(mask_s, mask_b)

    b2 = b.copy()
    blue_pixels = b > BLUE_THRESH
    b2[blue_pixels] = np.clip(b2[blue_pixels] * 1.5 + 30, 0, 255).astype(np.uint8)

    frame_out = cv2.merge([b2, g, r])
    return frame_out, mask


def process_video(video_path, work_dir):
    # scan video, find blobs, save annotated video and CSV
    name = video_path.stem

    out_video = work_dir / f"flame_with_framenumber_{name}.mp4"
    out_csv = work_dir / f"blob_bounding_boxes_{name}.csv"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video:", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    idx = 0

    # blob data by label
    blobs = {
        "L": {
            "frame": [], "area": [], "perimeter": [],
            "x": [], "y": [], "w": [], "h": [],
            "cx": [], "cy": []
        },
        "C": {
            "frame": [], "area": [], "perimeter": [],
            "x": [], "y": [], "w": [], "h": [],
            "cx": [], "cy": []
        },
        "R": {
            "frame": [], "area": [], "perimeter": [],
            "x": [], "y": [], "w": [], "h": [],
            "cx": [], "cy": []
        },
    }

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_proc, mask = prep_frame(frame)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # drop tiny blobs
        good = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]

        frame_draw = frame.copy()

        # need at least 3 blobs
        if len(good) >= 3:
            # pick 3 biggest and sort left to right
            good = sorted(good, key=cv2.contourArea, reverse=True)[:3]
            good = sorted(good, key=lambda c: cv2.boundingRect(c)[0])
            labels = ["L", "C", "R"]

            for lab, c in zip(labels, good):
                area = cv2.contourArea(c)
                peri = cv2.arcLength(c, True)
                x, y, ww, hh = cv2.boundingRect(c)

                M = cv2.moments(c)
                cx = cy = None
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                d = blobs[lab]
                d["frame"].append(idx)
                d["area"].append(area)
                d["perimeter"].append(peri)
                d["x"].append(x)
                d["y"].append(y)
                d["w"].append(ww)
                d["h"].append(hh)
                d["cx"].append(cx)
                d["cy"].append(cy)

        # frame index text
        txt = f"Frame {idx}"
        cv2.putText(
            frame_draw,
            txt,
            (w - 250, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        frames.append(frame_draw)
        idx += 1

    cap.release()

    # write annotated video
    fps_out = fps if fps and fps > 0 else 5.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vout = cv2.VideoWriter(str(out_video), fourcc, fps_out, (w, h))
    for f in frames:
        vout.write(f)
    vout.release()

    # flatten blob dict for CSV
    rows = []
    for lab in ["L", "C", "R"]:
        d = blobs[lab]
        n = len(d["frame"])
        for i in range(n):
            x = d["x"][i]
            ww = d["w"][i]
            rows.append(
                {
                    "label": lab,
                    "frame": d["frame"][i],
                    "area": d["area"][i],
                    "perimeter": d["perimeter"][i],
                    "x": x,
                    "y": d["y"][i],
                    "w": ww,
                    "h": d["h"][i],
                    "cx": d["cx"][i],
                    "cy": d["cy"][i],
                    "uL": x,
                    "uR": x + ww,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(str(out_csv), index=False)

    return out_csv


def load_blob_data(csv_path):
    # simple wrapper around read_csv
    return pd.read_csv(csv_path)


def get_frames_with_LCR(df):
    # frames that have L, C and R blobs
    out = []
    for f in df["frame"].unique():
        sub = df[df["frame"] == f]
        labs = set(sub["label"].values)
        if {"L", "C", "R"}.issubset(labs):
            out.append(f)
    return out


def align_score(df, frame_num):
    # score = vertical misalignment of centroids
    sub = df[df["frame"] == frame_num]

    cent = {}
    for lab in ["L", "C", "R"]:
        b = sub[sub["label"] == lab]
        if len(b) == 0:
            return float("inf"), None
        cent[lab] = b["cy"].values[0]

    cy_c = cent["C"]
    score = abs(cy_c - cent["L"]) + abs(cy_c - cent["R"])
    return score, cent


def pick_optimal_frame(df, frames_lcr, img_h):
    # pick frame with lowest score and center in middle third
    scores = {}
    all_cent = {}

    y_lo = img_h / 3.0
    y_hi = 2.0 * img_h / 3.0

    best_frame = None
    best_score = None

    for f in sorted(frames_lcr):
        s, c = align_score(df, f)
        scores[f] = s
        all_cent[f] = c

        if c is None:
            continue

        cy_c = c["C"]
        if cy_c < y_lo or cy_c > y_hi:
            continue

        if best_score is None or s < best_score:
            best_score = s
            best_frame = f

    return best_frame, scores, all_cent


def grab_frame(video_path, frame_num):
    # read a single frame by index
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def plot_alignment(scores, best_frame, out_path):
    # plot score vs frame and mark best frame
    xs = sorted(scores.keys())
    ys = [scores[f] for f in xs]

    plt.figure(figsize=(14, 5))
    plt.plot(xs, ys, linewidth=1.2)

    if best_frame in scores:
        plt.axvline(best_frame, linestyle="--", linewidth=2, color="steelblue")
        plt.plot(best_frame, scores[best_frame], "o", markersize=8, color="darkorange")

    plt.title("Centroid Vertical Alignment Score")
    plt.xlabel("Frame Number")
    plt.ylabel("Alignment Score (pixels)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def draw_opt_frame(df, video_path, frame_num, out_path):
    # save a PNG of the chosen frame with boxes and centroid lines
    frame = grab_frame(video_path, frame_num)
    if frame is None:
        print("Could not read frame", frame_num)
        return

    frame_proc, _ = prep_frame(frame)
    sub = df[df["frame"] == frame_num]

    colors = {"L": (0, 255, 0), "C": (255, 0, 0), "R": (0, 0, 255)}

    for _, row in sub.iterrows():
        lab = row["label"]
        x = int(row["x"])
        y = int(row["y"])
        ww = int(row["w"])
        hh = int(row["h"])
        cx = int(row["cx"])
        cy = int(row["cy"])

        col = colors[lab]
        cv2.rectangle(frame_proc, (x, y), (x + ww, y + hh), col, 2)
        cv2.circle(frame_proc, (cx, cy), 6, col, -1)

    for _, row in sub.iterrows():
        cy = int(row["cy"])
        cv2.line(
            frame_proc,
            (0, cy),
            (frame_proc.shape[1], cy),
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    score, cent = align_score(df, frame_num)
    if cent is None:
        cent = {"L": 0, "C": 0, "R": 0}

    y0 = 30
    cv2.putText(
        frame_proc,
        f"Frame: {frame_num}",
        (20, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y0 += 25
    cv2.putText(
        frame_proc,
        f"Align score: {score:.1f}",
        (20, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    y0 += 22
    cv2.putText(
        frame_proc,
        f"L cy: {cent['L']:.1f}",
        (20, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    y0 += 22
    cv2.putText(
        frame_proc,
        f"C cy: {cent['C']:.1f}",
        (20, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )
    y0 += 22
    cv2.putText(
        frame_proc,
        f"R cy: {cent['R']:.1f}",
        (20, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(out_path), frame_proc)


def save_optimal_frame_info(
    optimal_frame,
    score,
    centroids,
    aspect_ratios,
    working_dir,
    video_stem,
    in_per_pixel,
    frames_ahead,
    centroid_disp_in,
    centroid_vel_in_s,
):
    # append one row to optimal_frame_summary.csv
    summary_path = PROJECT_ROOT / "optimal_frame_summary.csv"

    headers = [
        "video_stem",
        "optimal_frame",
        "alignment_score_px",
        "centroid_L_y_px",
        "centroid_C_y_px",
        "centroid_R_y_px",
        "aspect_L_h_over_w",
        "aspect_C_h_over_w",
        "aspect_R_h_over_w",
        "in_per_pixel",
        "px_per_in",
        "frames_ahead",
        "centroid_disp_in",
        "centroid_vel_in_s",
    ]

    px_per_in = 1.0 / in_per_pixel

    row = [
        video_stem,
        optimal_frame,
        score,
        centroids["L"],
        centroids["C"],
        centroids["R"],
        aspect_ratios["L"],
        aspect_ratios["C"],
        aspect_ratios["R"],
        in_per_pixel,
        px_per_in,
        frames_ahead,
        centroid_disp_in if centroid_disp_in is not None else "",
        centroid_vel_in_s if centroid_vel_in_s is not None else "",
    ]

    write_header = not summary_path.exists()

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow(row)


def save_opt_frame_image(video_path, frame_num, dest_dir, video_stem):
    # save preprocessed optimal frame for 3D code
    frame = grab_frame(video_path, frame_num)
    if frame is None:
        print("Could not grab frame", frame_num)
        return

    frame_proc, _ = prep_frame(frame)
    fname = f"optimal_frame_enhanced_{video_stem}_frame_{frame_num}.png"
    out_path = dest_dir / fname
    cv2.imwrite(str(out_path), frame_proc)


def main():
    fname = input("Video filename: ").strip()

    video_path = PROJECT_ROOT / "Videos" / "Raw Video" / fname
    if not video_path.exists():
        print("File not found:", video_path)
        return

    video_stem = video_path.stem
    work_dir = PROJECT_ROOT / "Working" / video_stem
    work_dir.mkdir(parents=True, exist_ok=True)

    csv_path = process_video(video_path, work_dir)
    if csv_path is None:
        return

    df = load_blob_data(csv_path)
    frames_lcr = get_frames_with_LCR(df)
    if len(frames_lcr) == 0:
        print("No frame with all three blobs (L, C, R).")
        return

    cap = cv2.VideoCapture(str(video_path))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    opt_frame, scores, all_cent = pick_optimal_frame(df, frames_lcr, img_h)
    if opt_frame is None:
        print("Could not find an optimal frame.")
        return

    cent_opt = all_cent[opt_frame]
    score_opt = scores[opt_frame]

    plot_path = work_dir / f"alignment_plot_{video_stem}.png"
    plot_alignment(scores, opt_frame, plot_path)

    # aspect ratios for L, C, R at optimal frame
    frame_opt = df[df["frame"] == opt_frame]
    aspect = {}
    center_w = None

    for lab in ["L", "C", "R"]:
        b = frame_opt[frame_opt["label"] == lab]
        if len(b) == 0:
            aspect[lab] = float("nan")
        else:
            ww = float(b["w"].values[0])
            hh = float(b["h"].values[0])
            aspect[lab] = hh / ww if ww > 0 else float("nan")
            if lab == "C":
                center_w = ww

    if center_w is None or center_w <= 0:
        print("Center width invalid, cannot compute pixel-to-inch scale.")
        return

    # assume center tube is 2 in wide
    in_per_pixel = 2.0 / center_w

    # estimate centroid speed using a few frames ahead
    frames_ahead = 4
    fps_cam = 240.0

    center_df = df[df["label"] == "C"]
    row0 = center_df[center_df["frame"] == opt_frame]
    row1 = center_df[center_df["frame"] == opt_frame + frames_ahead]

    centroid_disp_in = None
    centroid_vel_in_s = None

    if len(row0) > 0 and len(row1) > 0:
        cy0 = float(row0["cy"].values[0])
        cy1 = float(row1["cy"].values[0])
        dy_px = cy1 - cy0
        centroid_disp_in = dy_px * in_per_pixel
        dt = frames_ahead / fps_cam
        if dt > 0:
            # force velocity to be positive
            centroid_vel_in_s = abs(centroid_disp_in / dt)
    else:
        print("Not enough frames to compute center centroid velocity.")

    save_optimal_frame_info(
        optimal_frame=opt_frame,
        score=score_opt,
        centroids=cent_opt,
        aspect_ratios=aspect,
        working_dir=work_dir,
        video_stem=video_stem,
        in_per_pixel=in_per_pixel,
        frames_ahead=frames_ahead,
        centroid_disp_in=centroid_disp_in,
        centroid_vel_in_s=centroid_vel_in_s,
    )

    out_annot = work_dir / f"optimal_frame_annotated_{video_stem}_frame_{opt_frame}.png"
    draw_opt_frame(df, video_path, opt_frame, out_annot)

    save_opt_frame_image(video_path, opt_frame, work_dir, video_stem)

    print(f"Done. Optimal frame = {opt_frame} for video {video_stem}.")


if __name__ == "__main__":
    main()
