import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from config import *

# ============================================================
# GEOMETRY / PARAMETERS
# ============================================================

# Tube geometry in world units (inches)
TUBE_RADIUS_IN = 1.0       # 2.0" diameter
TUBE_HEIGHT_IN = 8.0       # adjust as needed

# Camera / virtual camera layout (top view, world coords)
CAMERA_TUBE_DISTANCE_IN = 12.0   # real camera below tube
MIRROR_ANGLE_DEG = 38.26         # around back
VIRTUAL_CAM_DISTANCE_IN = 6.0    # distance of virtual cams from tube

# Voxel resolution in world space (inches per voxel)
VOXEL_SIZE_IN = 0.1  # finer = more detail, slower

# Downsample masks (pixels per voxel in image space)
VOXEL_SIZE_PIX = 1   # 1 = max image resolution

CALIB_FILE = Path("calibration_GeneralCal.npz")

# ============================================================
# CAMERA CALIBRATION / UNDISTORT
# ============================================================

if CALIB_FILE.exists():
    calib = np.load(str(CALIB_FILE))
    CAMERA_K = calib["K"]
    DIST_COEFFS = calib["dist"]
    NEW_CAMERA_K = calib.get("new_K", CAMERA_K)
    print("Loaded camera calibration from calibration_GeneralCal.npz")
else:
    CAMERA_K = None
    DIST_COEFFS = None
    NEW_CAMERA_K = None
    print("WARNING: calibration_GeneralCal.npz not found – running without undistortion")


def undistort_frame(frame):
    if CAMERA_K is None or DIST_COEFFS is None:
        return frame
    return cv2.undistort(frame, CAMERA_K, DIST_COEFFS, None, NEW_CAMERA_K)


# ============================================================
# BLOBS / CONTOURS
# ============================================================

def load_frame_blobs(csv_path, frame_num):
    df = pd.read_csv(csv_path)
    frame_data = df[df['frame'] == frame_num]

    blobs = {}
    for _, row in frame_data.iterrows():
        label = row['label']
        blobs[label] = {
            'x': int(row['x']),
            'y': int(row['y']),
            'w': int(row['w']),
            'h': int(row['h']),
            'cx': int(row['cx']),
            'cy': int(row['cy'])
        }
    return blobs


def extract_contour_from_video(video_path, frame_num, blob_bbox):
    """Threshold on UNDISTORTED frame, return full-frame contour."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame = undistort_frame(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    _, mask_s = cv2.threshold(s, 80, 255, cv2.THRESH_BINARY)

    b, g, r = cv2.split(frame)
    _, mask_b = cv2.threshold(b, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask_s, mask_b)

    x, y, w, h = blob_bbox['x'], blob_bbox['y'], blob_bbox['w'], blob_bbox['h']
    H, W = mask.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)

    if x1 <= x0 or y1 <= y0:
        return None

    roi_mask = mask[y0:y1, x0:x1]
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    contour[:, 0, 0] += x0
    contour[:, 0, 1] += y0
    return contour


# ============================================================
# CAMERA GEOMETRY (MULTI-VIEW)
# ============================================================

def look_at(cam_pos, target=np.array([0., 0., 0.]), up=np.array([0., 0., 1.])):
    """
    Build world->camera rotation R and translation t such that
    camera looks from cam_pos to target with given up vector.
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - cam_pos
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    true_up = np.cross(forward, right)

    R = np.vstack([right, true_up, forward])  # 3x3
    t = -R @ cam_pos
    return R, t


def build_camera_matrices():
    """
    Build 3 camera matrices P = K [R|t] for:
      - center camera (front)
      - left virtual camera (back-left, +angle around back)
      - right virtual camera (back-right, -angle around back)

    World coords:
      X: right, Y: toward camera (so camera is at negative Y), Z: up.
      Tube center at origin.

    Also returns:
      cam_positions: dict label -> 3D position
      view_dirs:     dict label -> 3D unit "look" direction
    """
    if CAMERA_K is None or NEW_CAMERA_K is None:
        raise RuntimeError("Need CAMERA_K from calibration_GeneralCal.npz")

    # Center camera in front (negative Y)
    cam_center = np.array([0., -CAMERA_TUBE_DISTANCE_IN, 0.0])
    R_C, t_C = look_at(cam_center)
    forward_C = (np.array([0., 0., 0.]) - cam_center)
    forward_C /= np.linalg.norm(forward_C)

    # Virtual cameras behind tube
    angle_back = np.deg2rad(90.0)
    delta = np.deg2rad(MIRROR_ANGLE_DEG)

    theta_L = angle_back + delta
    theta_R = angle_back - delta

    dir_L = np.array([np.cos(theta_L), np.sin(theta_L), 0.0])
    dir_R = np.array([np.cos(theta_R), np.sin(theta_R), 0.0])

    # Behind tube, so no minus sign
    cam_left = dir_L * VIRTUAL_CAM_DISTANCE_IN
    cam_right = dir_R * VIRTUAL_CAM_DISTANCE_IN

    R_L, t_L = look_at(cam_left)
    R_R, t_R = look_at(cam_right)

    forward_L = (np.array([0., 0., 0.]) - cam_left)
    forward_L /= np.linalg.norm(forward_L)
    forward_R = (np.array([0., 0., 0.]) - cam_right)
    forward_R /= np.linalg.norm(forward_R)

    def P_from_Rt(R, t):
        Rt = np.hstack([R, t.reshape(3, 1)])
        return NEW_CAMERA_K @ Rt

    P_C = P_from_Rt(R_C, t_C)
    P_L = P_from_Rt(R_L, t_L)
    P_R = P_from_Rt(R_R, t_R)

    cam_positions = {'C': cam_center, 'L': cam_left, 'R': cam_right}
    view_dirs = {'C': forward_C, 'L': forward_L, 'R': forward_R}

    return P_C, P_L, P_R, cam_positions, view_dirs


def visualize_camera_layout(cam_positions, view_dirs):
    """Top-view visualization of tube and virtual cameras."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')

    # Tube circle
    circle = plt.Circle((0.0, 0.0), TUBE_RADIUS_IN,
                        fill=False, linestyle='--')
    ax.add_artist(circle)
    ax.scatter([0.0], [0.0], c='k', s=30)
    ax.text(0.0, 0.0, "Tube", ha='left', va='bottom')

    colors = {'C': 'b', 'L': 'g', 'R': 'r'}
    for label in ['C', 'L', 'R']:
        pos = cam_positions[label]
        d = view_dirs[label]
        x, y = pos[0], pos[1]
        dx, dy = d[0], d[1]

        ax.scatter([x], [y], c=colors[label], s=40)
        ax.text(x, y, f" {label}", color=colors[label],
                ha='left', va='bottom')

        scale = TUBE_RADIUS_IN * 0.8
        ax.arrow(x, y, dx * scale, dy * scale,
                 head_width=TUBE_RADIUS_IN * 0.15,
                 length_includes_head=True,
                 color=colors[label], alpha=0.7)

    margin = max(TUBE_RADIUS_IN, CAMERA_TUBE_DISTANCE_IN,
                 VIRTUAL_CAM_DISTANCE_IN) * 1.2
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)

    ax.set_xlabel("X (inches, right)")
    ax.set_ylabel("Y (inches, toward camera)")
    ax.set_title("Top-view camera / virtual-camera layout")
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ============================================================
# PROJECTION / OFFSETS
# ============================================================

def project_points(P, X_world):
    """
    Project 3D points (N,3) in world coordinates to pixel coords (u,v).
    Returns (u,v,valid_mask).
    """
    N = X_world.shape[0]
    homog = np.hstack([X_world, np.ones((N, 1))])  # (N,4)
    cam = (P @ homog.T).T  # (N,3)
    zs = cam[:, 2]
    valid = zs > 1e-6
    u = cam[:, 0] / zs
    v = cam[:, 1] / zs
    return u, v, valid


def estimate_tube_radius_from_center(contour_C, P_C, offset_C,
                                     img_shape, z_ref=None,
                                     n_samples=200,
                                     R_initial=1.0):
    """
    Estimate tube radius in *world units* such that, when a circle of that
    radius is projected with P_C (plus offset), its apparent width in the
    image matches the center blob width.
    """
    H, W = img_shape

    if z_ref is None:
        z_ref = TUBE_HEIGHT_IN * 0.5

    # center blob width in pixels (UNDISTORTED coords)
    x, y, w, h = cv2.boundingRect(contour_C)
    blob_width = float(w)

    # sample circle of radius R_initial at z_ref
    theta = np.linspace(0, 2.0 * np.pi, n_samples, endpoint=False)
    xs = R_initial * np.cos(theta)
    ys = R_initial * np.sin(theta)
    zs = np.full_like(xs, z_ref)
    pts_world = np.vstack([xs, ys, zs]).T  # (N,3)

    # project
    u, v, valid = project_points(P_C, pts_world)
    du, dv = offset_C
    u = u + du
    v = v + dv

    in_bounds = (valid &
                 (u >= 0) & (u < W) &
                 (v >= 0) & (v < H))
    if not np.any(in_bounds):
        print("Tube radius estimation: projected test circle isn't in frame.")
        return R_initial  # fallback

    u_in = u[in_bounds]
    proj_width = float(u_in.max() - u_in.min())
    if proj_width <= 0:
        print("Tube radius estimation: projected width is zero/unusable.")
        return R_initial

    # width ∝ radius for fixed camera geometry => scale radius linearly
    scale = blob_width / proj_width
    R_est = R_initial * scale

    print("\n--- Tube radius from center blob ---")
    print(f"Center blob width (bbox): {blob_width:.2f} px")
    print(f"Test circle width (@R={R_initial:.3f}): {proj_width:.2f} px")
    print(f"=> tube_radius_world ≈ {R_est:.4f} (arbitrary units)")
    print(f"Nominal tube radius (in): {TUBE_RADIUS_IN:.4f}\n")

    return R_est


def compute_view_offsets(contours, P_mats, img_shape):
    """
    For each view (C,L,R), compute a 2D image offset so that the
    world tube center at mid-height projects to the blob's centroid.
    """
    offsets = {}
    H, W = img_shape
    z_ref = TUBE_HEIGHT_IN * 0.5
    p_ref = np.array([[0.0, 0.0, z_ref]])  # tube center at mid-height

    for label in ['C', 'L', 'R']:
        P = P_mats[label]

        u, v, valid = project_points(P, p_ref)
        u0, v0 = float(u[0]), float(v[0])

        x, y, w, h = cv2.boundingRect(contours[label])
        cx = x + w * 0.5
        cy = y + h * 0.5

        du = cx - u0
        dv = cy - v0
        offsets[label] = (du, dv)

        print(f"View {label}: proj center=({u0:.1f},{v0:.1f}), "
              f"blob center=({cx:.1f},{cy:.1f}), offset=(du={du:.1f}, dv={dv:.1f})")

    return offsets


def debug_overlay_centerlines(frame, P_mats, offsets):
    """
    Draw tube centerline projection for each view onto the undistorted frame,
    using different colors, and show with matplotlib.
    """
    img = frame.copy()
    H, W = img.shape[:2]

    colors_bgr = {'C': (255, 0, 0),  # blue-ish
                  'L': (0, 255, 0),  # green
                  'R': (0, 0, 255)}  # red

    zs = np.linspace(0.0, TUBE_HEIGHT_IN, 40)
    pts = np.vstack([np.zeros_like(zs),
                     np.zeros_like(zs),
                     zs]).T  # (N,3)

    for label in ['C', 'L', 'R']:
        P = P_mats[label]
        du, dv = offsets[label]
        color = colors_bgr[label]

        u, v, valid = project_points(P, pts)
        u = u + du
        v = v + dv

        for ui, vi, ok in zip(u, v, valid):
            if not ok:
                continue
            x = int(round(ui))
            y = int(round(vi))
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(img, (x, y), 2, color, -1)

    # convert BGR->RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title("Projected tube centerlines on undistorted frame\n"
              "Blue=C, Green=L, Red=R")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================
# VOXEL GRID SETUP (WORLD SPACE)
# ============================================================

def build_voxel_grid(tube_radius):
    """World-space voxel centers inside cylindrical tube of given radius."""
    R = tube_radius
    H = TUBE_HEIGHT_IN

    xs = np.arange(-R, R + VOXEL_SIZE_IN, VOXEL_SIZE_IN)
    ys = np.arange(-R, R + VOXEL_SIZE_IN, VOXEL_SIZE_IN)
    zs = np.arange(0.0, H + VOXEL_SIZE_IN, VOXEL_SIZE_IN)

    X, Y = np.meshgrid(xs, ys, indexing='ij')
    inside = (X ** 2 + Y ** 2) <= R ** 2  # (nx, ny)

    nx, ny, nz = len(xs), len(ys), len(zs)
    occupied = np.zeros((nx, ny, nz), dtype=bool)
    occupied[inside, :] = True

    print(f"Voxel grid in world units: {occupied.shape} (nx, ny, nz)")
    print(f"Tube radius used in grid: {R:.4f}")
    return xs, ys, zs, occupied


# ============================================================
# VISUAL HULL CARVING
# ============================================================

def carve_visual_hull(xs, ys, zs, occupied,
                      masks,
                      P_mats, offsets,
                      img_shape, voxel_size_pix):
    """
    Standard visual hull carving (intersection across C,L,R).
    """
    H, W = img_shape
    Hs, Ws = H // voxel_size_pix, W // voxel_size_pix

    nx, ny, nz = occupied.shape

    Xs, Ys, Zs = np.meshgrid(xs, ys, zs, indexing='ij')
    points = np.vstack([Xs.ravel(), Ys.ravel(), Zs.ravel()]).T  # (N,3)

    keep = occupied.ravel().copy()

    for label in ['C', 'L', 'R']:
        print(f"Carving with view {label}...")
        P = P_mats[label]
        mask = masks[label]
        du, dv = offsets[label]

        u, v, valid = project_points(P, points)

        u = u + du
        v = v + dv

        u_small = (u / voxel_size_pix).astype(int)
        v_small = (v / voxel_size_pix).astype(int)

        in_bounds = (u_small >= 0) & (u_small < Ws) & \
                    (v_small >= 0) & (v_small < Hs)
        valid_all = valid & in_bounds

        ok = np.zeros_like(keep, dtype=bool)
        idx = np.where(valid_all)[0]
        ok[idx] = mask[v_small[idx], u_small[idx]]

        keep &= ok

    occupied[:] = keep.reshape(occupied.shape)
    print(f"Occupied voxels after carving: {occupied.sum()}")
    return occupied


# ============================================================
# TOP SURFACE / VISUALIZATION
# ============================================================

def compute_top_surface(xs, ys, zs, occupied, tube_radius):
    """
    Compute top surface height_map[x_i, y_j] in world units.

    Inside tube radius:
        height of highest occupied voxel, or NaN if none.
    Outside tube radius:
        NaN (masked).
    """
    R = tube_radius
    nx, ny, nz = occupied.shape
    height_map = np.zeros((nx, ny), dtype=float)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if x * x + y * y > R * R:
                height_map[i, j] = np.nan
                continue
            col = occupied[i, j, :]
            if np.any(col):
                k_max = np.where(col)[0][-1]
                height_map[i, j] = zs[k_max]
            else:
                # mark as "no flame here" -> NaN so it disappears
                height_map[i, j] = np.nan

    return height_map


def smooth_height_map(height_map, ksize=9, sigma=2.0):
    """
    Smooth height_map with a Gaussian while respecting NaNs (holes).

    ksize: odd kernel size in pixels (e.g. 5, 7, 9)
    sigma: Gaussian sigma in pixels
    """
    H = height_map.astype(np.float32)

    # Mask: 1 where valid, 0 where NaN
    valid = ~np.isnan(H)
    weight = valid.astype(np.float32)

    # Replace NaNs with 0 for convolution
    data = H.copy()
    data[~valid] = 0.0

    # 2D Gaussian kernel
    g1d = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    kernel = g1d @ g1d.T

    # Convolve data and weights
    data_blur = cv2.filter2D(data, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    weight_blur = cv2.filter2D(weight, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    with np.errstate(invalid='ignore', divide='ignore'):
        smoothed = data_blur / weight_blur
    smoothed[weight_blur == 0] = np.nan

    return smoothed


def visualize_top_surface(xs, ys, height_map, tube_radius):
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    Z = np.ma.masked_invalid(height_map)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    linewidth=0, antialiased=True)

    ax.set_xlabel('X (units)')
    ax.set_ylabel('Y (units)')
    ax.set_zlabel('Z (units)')
    ax.set_title('Top surface (visual hull)')

    max_xy = tube_radius * 1.1
    ax.set_xlim(-max_xy, max_xy)
    ax.set_ylim(-max_xy, max_xy)

    plt.tight_layout()
    plt.show()


def make_scaled_masks(contours, img_shape, voxel_size_pix):
    """
    Create full-res binary masks for C, L, R, then scale L and R so that
    their bounding boxes (width & height) match the CENTER's bbox.
    Finally downsample by voxel_size_pix.

    This enforces: all three views see a flame with the same apparent
    width/height in the image, centered at each blob's centroid.
    """
    H, W = img_shape

    def contour_mask(cnt):
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(m, [cnt], -1, 255, -1)
        return m

    # full-res masks
    full = {lbl: contour_mask(contours[lbl]) for lbl in ['C', 'L', 'R']}

    # center bbox as reference
    xC, yC, wC, hC = cv2.boundingRect(contours['C'])
    cxC = xC + wC * 0.5
    cyC = yC + hC * 0.5

    def scale_mask_to_center(label):
        if label == 'C':
            return full['C']
        cnt = contours[label]
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w * 0.5
        cy = y + h * 0.5

        if w == 0 or h == 0:
            return full[label]

        sx = wC / float(w)
        sy = hC / float(h)

        # scale around this blob's own centroid
        M = np.array([
            [sx, 0.0, cx * (1.0 - sx)],
            [0.0, sy, cy * (1.0 - sy)]
        ], dtype=np.float32)

        scaled = cv2.warpAffine(full[label], M, (W, H),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        return scaled

    full_scaled = {
        'C': full['C'],
        'L': scale_mask_to_center('L'),
        'R': scale_mask_to_center('R')
    }

    # downsample to voxel resolution
    def downsample(m):
        new_H = H // voxel_size_pix
        new_W = W // voxel_size_pix
        m_small = cv2.resize(m, (new_W, new_H), interpolation=cv2.INTER_AREA)
        return m_small > 127

    masks = {lbl: downsample(full_scaled[lbl]) for lbl in ['C', 'L', 'R']}

    print("\nMask bbox after scaling to center:")
    for lbl in ['C', 'L', 'R']:
        ys_idx, xs_idx = np.where(full_scaled[lbl] > 0)
        if len(xs_idx) == 0:
            print(f"  {lbl}: EMPTY")
            continue
        w = xs_idx.max() - xs_idx.min() + 1
        h = ys_idx.max() - ys_idx.min() + 1
        print(f"  {lbl}: w={w}, h={h}")
    print()

    return masks


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("VISUAL HULL RECONSTRUCTION (OPTION 1)")
    print("=" * 60)

    if CAMERA_K is None or NEW_CAMERA_K is None:
        print("Error: camera calibration not loaded. Aborting visual hull.")
        return

    if not BLOB_CSV.exists():
        print(f"Error: {BLOB_CSV} not found. Run video_preprocessing.py first.")
        return

    optimal_frame_file = OUTPUT_DIR / "optimal_frame.txt"
    if optimal_frame_file.exists():
        frame_num = int(optimal_frame_file.read_text().strip())
        print(f"Loaded optimal frame: {frame_num}")
    else:
        frame_num = 70
        print(f"No optimal frame, using frame {frame_num}")

    blobs = load_frame_blobs(BLOB_CSV, frame_num)
    print(f"Found blobs: {list(blobs.keys())}")
    if set(blobs.keys()) != {'L', 'C', 'R'}:
        print("Need exactly blobs L, C, R")
        return

    # grab undistorted frame for size + debug overlay
    cap = cv2.VideoCapture(str(VIDEO_INPUT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read frame")
        return
    frame = undistort_frame(frame)
    img_h, img_w = frame.shape[:2]
    print(f"Undistorted frame: {img_w} x {img_h}")

    # contours for each blob
    contours = {}
    for label in ['L', 'C', 'R']:
        cnt = extract_contour_from_video(VIDEO_INPUT, frame_num, blobs[label])
        if cnt is None:
            print(f"Could not extract contour for {label}")
            return
        contours[label] = cnt
        print(f"{label}: {len(cnt)} contour points")

    # silhouettes (downsampled) with L/R scaled to match C bbox
    masks = make_scaled_masks(contours, (img_h, img_w), VOXEL_SIZE_PIX)
    for label in ['C', 'L', 'R']:
        print(f"Mask {label} shape (downsampled): {masks[label].shape}")

    # camera matrices + layout
    P_C, P_L, P_R, cam_positions, view_dirs = build_camera_matrices()
    P_mats = {'C': P_C, 'L': P_L, 'R': P_R}

    visualize_camera_layout(cam_positions, view_dirs)

    # offsets to align tube center with each blob center
    offsets = compute_view_offsets(contours, P_mats, (img_h, img_w))

    # estimate tube radius from center blob width
    tube_radius_world = estimate_tube_radius_from_center(
        contour_C=contours['C'],
        P_C=P_mats['C'],
        offset_C=offsets['C'],
        img_shape=(img_h, img_w),
        z_ref=TUBE_HEIGHT_IN * 0.5,
        R_initial=1.0
    )

    # debug overlay of centerlines
    debug_overlay_centerlines(frame, P_mats, offsets)

    # voxel grid using that radius
    xs, ys, zs, occupied = build_voxel_grid(tube_radius_world)

    # carve
    occupied = carve_visual_hull(xs, ys, zs, occupied,
                                 masks, P_mats, offsets,
                                 (img_h, img_w),
                                 VOXEL_SIZE_PIX)

    if occupied.sum() == 0:
        print("No occupied voxels after carving – geometry/offsets still off.")
        return

    # top surface
    height_map = compute_top_surface(xs, ys, zs, occupied, tube_radius_world)

    # explicitly drop any <= 0 values if they appear
    height_map[height_map <= 0] = np.nan

    # smooth to avoid boxy / plateau look
    height_map_smooth = smooth_height_map(height_map, ksize=9, sigma=2.0)

    visualize_top_surface(xs, ys, height_map_smooth, tube_radius_world)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_file = OUTPUT_DIR / "visual_hull_top_surface.npy"
    np.save(str(out_file), height_map_smooth)
    print(f"Saved smoothed top surface height map to {out_file}")


if __name__ == "__main__":
    main()
