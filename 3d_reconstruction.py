import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# base folder
PROJECT_ROOT = Path(__file__).resolve().parent

# basic settings
VOXEL_SIZE = 1
MIRROR_ANGLE_DEG = 38
TUBE_DIAMETER_CM = 2.0 * 2.54  # 2 in in cm


def load_frame_blobs(csv_path, frame_num):
    # read blobs for one frame from CSV
    df = pd.read_csv(csv_path)
    frame_data = df[df["frame"] == frame_num]

    blobs = {}
    for _, row in frame_data.iterrows():
        label = row["label"]
        blobs[label] = {
            "x": int(row["x"]),
            "y": int(row["y"]),
            "w": int(row["w"]),
            "h": int(row["h"]),
            "cx": int(row["cx"]),
            "cy": int(row["cy"]),
        }
    return blobs


def extract_contour_from_frame(frame, blob_bbox):
    # simple flame mask and contour inside one bounding box
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    _, mask_s = cv2.threshold(s, 80, 255, cv2.THRESH_BINARY)

    b, g, r = cv2.split(frame)
    _, mask_b = cv2.threshold(b, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask_s, mask_b)

    x, y, w, h = blob_bbox["x"], blob_bbox["y"], blob_bbox["w"], blob_bbox["h"]

    H, W = mask.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None

    roi_mask = mask[y0:y1, x0:x1]

    contours, _ = cv2.findContours(
        roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    contour[:, 0, 0] += x0
    contour[:, 0, 1] += y0
    return contour


def create_profile_mask(contour, img_shape, voxel_size):
    # binary mask of the contour, downsampled
    h, w = img_shape
    new_h = h // voxel_size
    new_w = w // voxel_size

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    mask_small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return mask_small > 127


def align_contours_vertically(contours, ref_label, target_label):
    # shift target contour so vertical midlines match
    ref = contours[ref_label]
    tgt = contours[target_label]

    ref_y = ref[:, 0, 1]
    tgt_y = tgt[:, 0, 1]

    ref_min, ref_max = ref_y.min(), ref_y.max()
    tgt_min, tgt_max = tgt_y.min(), tgt_y.max()

    ref_mid = 0.5 * (ref_min + ref_max)
    tgt_mid = 0.5 * (tgt_min + tgt_max)

    dy = ref_mid - tgt_mid

    pts = tgt.astype(np.float32)
    pts[:, 0, 1] += dy
    contours[target_label] = pts.astype(np.int32)


def visualize_two_view_intersection(
    voxels_per_view, label1, label2, title_suffix="", stride=4, max_points=60000
):
    # scatter plot showing overlap between two voxel sets
    v1 = voxels_per_view[label1]
    v2 = voxels_per_view[label2]

    nx, ny, nz = v1.shape

    v1_thin = v1[::stride, ::stride, ::stride]
    v2_thin = v2[::stride, ::stride, ::stride]

    both = v1_thin & v2_thin
    only1 = v1_thin & ~v2_thin
    only2 = v2_thin & ~v1_thin

    x1, y1, z1 = np.where(only1)
    x2, y2, z2 = np.where(only2)
    xb, yb, zb = np.where(both)

    x1 = x1 * stride
    y1 = y1 * stride
    z1 = z1 * stride
    x2 = x2 * stride
    y2 = y2 * stride
    z2 = z2 * stride
    xb = xb * stride
    yb = yb * stride
    zb = zb * stride

    cx = nx / 2.0
    cy = ny / 2.0

    x1 = x1 - cx
    y1 = y1 - cy
    x2 = x2 - cx
    y2 = y2 - cy
    xb = xb - cx
    yb = yb - cy

    def limit_points(x, y, z, max_pts):
        n = x.size
        if n <= max_pts:
            return x, y, z
        idx = np.random.choice(n, max_pts, replace=False)
        return x[idx], y[idx], z[idx]

    x1, y1, z1 = limit_points(x1, y1, z1, max_points)
    x2, y2, z2 = limit_points(x2, y2, z2, max_points)
    xb, yb, zb = limit_points(xb, yb, zb, max_points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x1, y1, z1, s=3, alpha=0.4, label=f"{label1} only")
    ax.scatter(x2, y2, z2, s=3, alpha=0.4, label=f"{label2} only")
    ax.scatter(xb, yb, zb, s=6, alpha=0.8, label="intersection")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    max_range = max(nx, ny, nz) / 2.0
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, nz])

    title = f"Intersection ({label1} vs {label2})"
    if title_suffix:
        title += " - " + title_suffix
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.show()


def smooth_height_map(height_map, kernel_size=3, iterations=1):
    # simple local averaging to smooth height map
    Z = height_map.copy()
    nx, ny = Z.shape
    k = kernel_size // 2

    for _ in range(iterations):
        newZ = Z.copy()
        for i in range(nx):
            i0 = max(0, i - k)
            i1 = min(nx, i + k + 1)
            for j in range(ny):
                if np.isnan(Z[i, j]):
                    continue
                j0 = max(0, j - k)
                j1 = min(ny, j + k + 1)

                window = Z[i0:i1, j0:j1]
                valid = ~np.isnan(window)
                if not np.any(valid):
                    continue
                newZ[i, j] = np.mean(window[valid])
        Z = newZ

    return Z


def surface_area_from_height_map(height_map, voxel_size, cm_per_pixel):
    # approximate surface area from height map using gradients
    nx, ny = height_map.shape

    dx = voxel_size * cm_per_pixel
    dy = voxel_size * cm_per_pixel
    dz = voxel_size * cm_per_pixel

    Z = height_map.astype(float) * dz

    total_area = 0.0

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if np.isnan(Z[i, j]):
                continue

            zx1 = Z[i + 1, j]
            zx0 = Z[i - 1, j]
            zy1 = Z[i, j + 1]
            zy0 = Z[i, j - 1]

            if (
                np.isnan(zx1)
                or np.isnan(zx0)
                or np.isnan(zy1)
                or np.isnan(zy0)
            ):
                continue

            dzdx = (zx1 - zx0) / (2.0 * dx)
            dzdy = (zy1 - zy0) / (2.0 * dy)

            area_cell = np.sqrt(1.0 + dzdx * dzdx + dzdy * dzdy) * dx * dy
            total_area += area_cell

    return total_area


def voxels_to_height_map(voxels):
    # take topmost occupied voxel in each (x,y) column
    nx, ny, nz = voxels.shape
    height_map = np.full((nx, ny), np.nan, dtype=float)

    for x in range(nx):
        for y in range(ny):
            for z in range(nz - 1, -1, -1):
                if voxels[x, y, z]:
                    height_map[x, y] = z
                    break
    return height_map


def visualize_height_map(height_map, title="Top surface height map"):
    # 3D surface plot of height map
    nx, ny = height_map.shape
    if np.all(np.isnan(height_map)):
        return

    z_min = np.nanmin(height_map)
    z_max = np.nanmax(height_map)
    nz = z_max

    xs = np.arange(nx) - nx / 2.0
    ys = np.arange(ny) - ny / 2.0
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    Z = np.ma.masked_invalid(height_map)
    Z_norm = (Z - z_min) / (z_max - z_min + 1e-9)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=plt.cm.coolwarm_r(Z_norm),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (height index)")
    ax.set_title(title)

    max_range = max(nx, ny, nz) / 2.0
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, nz])

    plt.tight_layout()
    plt.show()


def visualize_cylinder_with_projections(
    blobs, contours, img_shape, voxel_size, mirror_angle_deg=MIRROR_ANGLE_DEG
):
    # show cylinder with projectd silhouettes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    img_h, img_w = img_shape
    grid_h = img_h // voxel_size

    tube_diameter_px = blobs["C"]["w"]
    grid_size = tube_diameter_px // voxel_size
    R_vox = grid_size / 2.0

    theta = np.linspace(0, 2 * np.pi, 60)
    z_cyl = np.linspace(0, grid_h, 40)
    Theta, Z = np.meshgrid(theta, z_cyl)
    Xc = R_vox * np.cos(Theta)
    Yc = R_vox * np.sin(Theta)
    ax.plot_wireframe(Xc, Yc, Z - grid_h / 2, linewidth=0.5, alpha=0.2)

    mask_L = np.flip(create_profile_mask(contours["L"], img_shape, voxel_size), axis=0)
    mask_C = np.flip(create_profile_mask(contours["C"], img_shape, voxel_size), axis=0)
    mask_R = np.flip(create_profile_mask(contours["R"], img_shape, voxel_size), axis=0)

    def bbox_vox(b):
        return (
            b["x"] // voxel_size,
            b["y"] // voxel_size,
            max(1, b["w"] // voxel_size),
            max(1, b["h"] // voxel_size),
        )

    xC, yC, wC, hC = bbox_vox(blobs["C"])
    xL, yL, wL, hL = bbox_vox(blobs["L"])
    xR, yR, wR, hR = bbox_vox(blobs["R"])

    e = np.array([1.0, 0.0])
    e = e / np.linalg.norm(e)

    angle_front = np.deg2rad(270.0)
    angle_back = np.deg2rad(90.0)
    delta = np.deg2rad(mirror_angle_deg)

    theta_C = angle_front
    theta_L = angle_back + delta
    theta_R = angle_back - delta

    dC = np.array([np.cos(theta_C), np.sin(theta_C)])
    dL = np.array([np.cos(theta_L), np.sin(theta_L)])
    dR = np.array([np.cos(theta_R), np.sin(theta_R)])

    dC = dC / np.linalg.norm(dC)
    dL = dL / np.linalg.norm(dL)
    dR = dR / np.linalg.norm(dR)

    def intersect_on_cylinder(u, d):
        ed = np.dot(e, d)

        A = 1.0
        B = 2.0 * u * ed
        C = u * u - R_vox * R_vox
        disc = B * B - 4 * A * C
        if disc < 0:
            return None

        sqrt_disc = np.sqrt(disc)
        t1 = (-B + sqrt_disc) / (2 * A)
        t2 = (-B - sqrt_disc) / (2 * A)

        r1 = u * e + t1 * d
        r2 = u * e + t2 * d
        r = r1 if np.dot(r1, d) > np.dot(r2, d) else r2
        return r[0], r[1]

    def col_to_u(col, x0, w_span):
        t = (col - x0 + 0.5) / float(w_span)
        return (t - 0.5) * 2.0 * R_vox

    def project_view(mask, x0, w_span, d, color, label, stride=4):
        xs, ys, zs = [], [], []
        h_mask, w_mask = mask.shape

        for z in range(0, h_mask, stride):
            row = mask[z, :]
            cols = np.where(row)[0]
            for col in cols[::stride]:
                if col < x0 or col >= x0 + w_span:
                    continue

                u = col_to_u(col, x0, w_span)
                if abs(u) > R_vox:
                    continue

                xy = intersect_on_cylinder(u, d)
                if xy is None:
                    continue
                x, y = xy
                z_world = z - grid_h / 2

                xs.append(x)
                ys.append(y)
                zs.append(z_world)

        if xs:
            xs = np.array(xs)
            ys = np.array(ys)
            zs = np.array(zs)
            ax.scatter(xs, ys, zs, s=5, alpha=0.7, c=color, label=label, depthshade=True)

    project_view(mask_C, xC, wC, dC, color="b", label="center")
    project_view(mask_L, xL, wL, dL, color="g", label="left")
    project_view(mask_R, xR, wR, dR, color="r", label="right")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Cylinder projections, mirror angle = {mirror_angle_deg:.2f} deg")
    ax.legend()

    max_range = max(R_vox * 2, grid_h) / 2.0
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range])

    plt.tight_layout()
    plt.show()


def compute_voxels_per_view(
    blobs, contours, img_shape, voxel_size, mirror_angle_deg=MIRROR_ANGLE_DEG
):
    # build voxel grids for each view
    h, w = img_shape
    grid_h = h // voxel_size

    tube_diameter_px = blobs["C"]["w"]
    grid_size = tube_diameter_px // voxel_size
    R_vox = grid_size / 2.0

    skew_info = {}
    for label in ["L", "C", "R"]:
        cnt = contours[label]
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        bbox_area = max(1, bw * bh)
        fill_ratio = area / float(bbox_area)
        skew_info[label] = {
            "fill_ratio": fill_ratio,
            "area": area,
            "bbox_area": bbox_area,
        }

    skew_order = sorted(skew_info.keys(), key=lambda k: skew_info[k]["fill_ratio"])

    masks = {}
    for label in ["L", "C", "R"]:
        mask = create_profile_mask(contours[label], img_shape, voxel_size)
        masks[label] = np.flip(mask, axis=0)

    def bbox_vox(b):
        return (
            b["x"] // voxel_size,
            b["y"] // voxel_size,
            max(1, b["w"] // voxel_size),
            max(1, b["h"] // voxel_size),
        )

    bboxes = {}
    for label in ["L", "C", "R"]:
        bboxes[label] = bbox_vox(blobs[label])

    angle_front = np.deg2rad(270.0)
    angle_back = np.deg2rad(90.0)
    delta = np.deg2rad(mirror_angle_deg)

    theta_C = angle_front
    theta_L = angle_back + delta
    theta_R = angle_back - delta

    dC = np.array([np.cos(theta_C), np.sin(theta_C)])
    dL = np.array([np.cos(theta_L), np.sin(theta_L)])
    dR = np.array([np.cos(theta_R), np.sin(theta_R)])

    def ortho_basis(d):
        d = d / np.linalg.norm(d)
        e = np.array([-d[1], d[0]])
        return d, e

    dC, eC = ortho_basis(dC)
    dL, eL = ortho_basis(dL)
    dR, eR = ortho_basis(dR)

    view_dirs = {
        "C": (dC, eC),
        "L": (dL, eL),
        "R": (dR, eR),
    }

    def u_to_col(u, R_vox, x0, w_span, mask_width):
        t = 0.5 + 0.5 * (u / R_vox)
        col = x0 + t * w_span
        col = int(np.floor(col))
        if col < 0 or col >= mask_width:
            return -1
        return col

    voxels_per_view = {
        "L": np.zeros((grid_size, grid_size, grid_h), dtype=bool),
        "C": np.zeros((grid_size, grid_size, grid_h), dtype=bool),
        "R": np.zeros((grid_size, grid_size, grid_h), dtype=bool),
    }

    cx = cy = (grid_size - 1) / 2.0

    for ix in range(grid_size):
        dx = ix - cx
        for iy in range(grid_size):
            dy = iy - cy

            if dx * dx + dy * dy > R_vox * R_vox:
                continue

            p = np.array([dx, dy])

            for label in ["L", "C", "R"]:
                mask = masks[label]
                d, e = view_dirs[label]
                x0, y0, w_span, h_span = bboxes[label]

                u = p.dot(e)
                if label == "L":
                    u = -u

                col = u_to_col(u, R_vox, x0, w_span, mask.shape[1])
                if col < 0:
                    continue

                for iz in range(grid_h):
                    z = iz
                    if z >= mask.shape[0]:
                        continue
                    if mask[z, col]:
                        voxels_per_view[label][ix, iy, iz] = True

    return voxels_per_view, skew_order


def smooth_voxels(voxels, min_neighbors=4):
    # simple 3D majority filter
    nx, ny, nz = voxels.shape
    out = np.zeros_like(voxels, dtype=bool)

    padded = np.pad(voxels, 1, mode="constant", constant_values=False)

    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            for k in range(1, nz + 1):
                block = padded[i - 1 : i + 2, j - 1 : j + 2, k - 1 : k + 2]
                if np.count_nonzero(block) >= min_neighbors:
                    out[i - 1, j - 1, k - 1] = True
    return out


def load_optimal_from_summary(video_stem):
    # read best frame info from optimal_frame_summary.csv
    summary_path = PROJECT_ROOT / "optimal_frame_summary.csv"
    if not summary_path.exists():
        return None, None, None

    df = pd.read_csv(summary_path)
    if "video_stem" not in df.columns:
        return None, None, None

    rows = df[df["video_stem"] == video_stem]
    if len(rows) == 0:
        return None, None, None

    row = rows.iloc[-1]

    try:
        frame_num = int(row["optimal_frame"])
    except Exception:
        frame_num = None

    try:
        in_per_pixel = float(row["in_per_pixel"])
    except Exception:
        in_per_pixel = None

    centroid_vel_in_s = None
    if "centroid_vel_in_s" in row.index:
        val = row["centroid_vel_in_s"]
        if pd.notna(val):
            try:
                centroid_vel_in_s = float(val)
            except Exception:
                centroid_vel_in_s = None

    
    return frame_num, in_per_pixel, centroid_vel_in_s


def append_surface_summary(
    video_stem,
    frame_num,
    tube_diameter_px,
    pixels_per_cm,
    cm_per_pixel,
    base_area_cm2,
    area_cm2,
    views_used,
    centroid_speed_cm_s,
    Su_cm_s,
):
    # append one row to BV_output.csv 
    summary_path = PROJECT_ROOT / "BV_output.csv"

    data = {
        "video_stem": [video_stem],
        "frame": [frame_num],
        "tube_diameter_px": [tube_diameter_px],
        "tube_diameter_cm": [TUBE_DIAMETER_CM],
        "pixels_per_cm": [pixels_per_cm],
        "cm_per_pixel": [cm_per_pixel],
        "tube_base_area_cm2": [base_area_cm2],
        "flame_surface_area_cm2": [area_cm2],
        "views_used": [",".join(views_used)],
        "centroid_speed_cm_s": [
            centroid_speed_cm_s if centroid_speed_cm_s is not None else ""
        ],
        "burning_velocity_cm_s": [Su_cm_s if Su_cm_s is not None else ""],
    }

    new_row = pd.DataFrame(data)

    if summary_path.exists():
        df_old = pd.read_csv(summary_path)
        df_all = pd.concat([df_old, new_row], ignore_index=True)
    else:
        df_all = new_row

    df_all.to_csv(summary_path, index=False)


def main():
    # ask user for video name (same as first script)
    filename = input("Video filename: ").strip()
    video_stem = Path(filename).stem

    working_dir = PROJECT_ROOT / "Working" / video_stem
    blob_csv = working_dir / f"blob_bounding_boxes_{video_stem}.csv"

    if not blob_csv.exists():
        print("Blob CSV not found.")
        return

    frame_num, in_per_pixel, centroid_vel_in_s = load_optimal_from_summary(video_stem)
    if frame_num is None:
        print("No optimal frame info in summary CSV.")
        return

    frame_path = (
        working_dir
        / f"optimal_frame_enhanced_{video_stem}_frame_{frame_num}.png"
    )
    if not frame_path.exists():
        print("Optimal frame image not found.")
        return

    frame = cv2.imread(str(frame_path))
    if frame is None:
        print("Could not read optimal frame image.")
        return

    img_h, img_w = frame.shape[:2]

    blobs = load_frame_blobs(blob_csv, frame_num)
    if set(blobs.keys()) != {"L", "C", "R"}:
        print("Did not find blobs L, C, R for that frame.")
        return

    # extract contours for all three blobs
    contours = {}
    for label in ["L", "C", "R"]:
        c = extract_contour_from_frame(frame, blobs[label])
        if c is None:
            print("Could not extract contour.")
            return
        contours[label] = c

    # scale side views to match center bounding box size
    xC, yC, wC, hC = cv2.boundingRect(contours["C"])
    for label in ["L", "R"]:
        cnt = contours[label]
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        sx = wC / float(w)
        sy = hC / float(h)
        cx = x + w / 2
        cy = y + h / 2
        pts = cnt.astype(np.float32)
        pts[:, 0, 0] = (pts[:, 0, 0] - cx) * sx + cx
        pts[:, 0, 1] = (pts[:, 0, 1] - cy) * sy + cy
        contours[label] = pts.astype(np.int32)

    # vertically align L and R to C
    align_contours_vertically(contours, "C", "L")
    align_contours_vertically(contours, "C", "R")

    # update blob boxes to fit new contours
    for label in ["L", "C", "R"]:
        x, y, w, h = cv2.boundingRect(contours[label])
        blobs[label]["x"] = x
        blobs[label]["y"] = y
        blobs[label]["w"] = w
        blobs[label]["h"] = h
        blobs[label]["cx"] = x + w // 2
        blobs[label]["cy"] = y + h // 2

    tube_diameter_px = blobs["C"]["w"]

    # convert pixel scale
    if in_per_pixel is None:
        pixels_per_cm = tube_diameter_px / TUBE_DIAMETER_CM
        cm_per_pixel = 1.0 / pixels_per_cm
    else:
        cm_per_pixel = in_per_pixel * 2.54
        pixels_per_cm = 1.0 / cm_per_pixel

    centroid_speed_cm_s = None
    if centroid_vel_in_s is not None:
        centroid_speed_cm_s = centroid_vel_in_s * 2.54

    # visualization of the cylinder and projections
    visualize_cylinder_with_projections(
        blobs, contours, (img_h, img_w), VOXEL_SIZE
    )

    # compute voxel grids for each view
    voxels_per_view, skew_order = compute_voxels_per_view(
        blobs, contours, (img_h, img_w), VOXEL_SIZE, MIRROR_ANGLE_DEG
    )

    # use the two best-filled views
    two_labels = skew_order[:2]
    visualize_two_view_intersection(
        voxels_per_view,
        two_labels[0],
        two_labels[1],
        title_suffix="aligned silhouettes",
        stride=5,
        max_points=30000,
    )

    active_labels = two_labels
    combined_voxels = voxels_per_view[active_labels[0]].copy()
    for lab in active_labels[1:]:
        combined_voxels &= voxels_per_view[lab]

    # convert voxels to height map, then smooth
    height_map_raw = voxels_to_height_map(combined_voxels)
    height_map_smooth = smooth_height_map(height_map_raw, kernel_size=3, iterations=2)

    # estimate surface area in cm^2
    area_cm2 = surface_area_from_height_map(
        height_map_smooth, VOXEL_SIZE, cm_per_pixel
    )

    radius_cm = TUBE_DIAMETER_CM / 2.0
    base_area_cm2 = np.pi * radius_cm * radius_cm

    Su_cm_s = None
    if centroid_speed_cm_s is not None and area_cm2 > 0:
        Ss_cm_s = centroid_speed_cm_s
        Su_cm_s = Ss_cm_s * (base_area_cm2 / area_cm2)

    # plot height map
    visualize_height_map(height_map_smooth, title="Top surface height map")

    # write output row
    append_surface_summary(
        video_stem=video_stem,
        frame_num=frame_num,
        tube_diameter_px=tube_diameter_px,
        pixels_per_cm=pixels_per_cm,
        cm_per_pixel=cm_per_pixel,
        base_area_cm2=base_area_cm2,
        area_cm2=area_cm2,
        views_used=active_labels,
        centroid_speed_cm_s=centroid_speed_cm_s,
        Su_cm_s=Su_cm_s,
    )

    if Su_cm_s is not None:
        print(
            f"{video_stem}, frame {frame_num}: area {area_cm2:.3f} cm^2, Su {Su_cm_s:.2f} cm/s"
        )
    else:
        print(f"{video_stem}, frame {frame_num}: area {area_cm2:.3f} cm^2")


if __name__ == "__main__":
    main()
