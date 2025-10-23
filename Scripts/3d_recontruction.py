import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# Paths
video_path = r"H:\My Drive\Final Project\Videos\Raw Video\GX010312_trimmed.mp4"
calibration_json = r"H:\My Drive\Final Project\Data\calibration_from_blobs.json"

# Thresholds (same as preprocessing)
MIN_AREA = 50
SAT_THRESH = 80
BLUE_THRESH = 10

# Animation settings
CREATE_ANIMATION = False  # Set to True after testing
OUTPUT_FPS = 5

def load_calibration(json_path):
    """Load calibration parameters from JSON"""
    with open(json_path, 'r') as f:
        calib = json.load(f)
    return calib

def extract_contour_from_frame(frame, label_position):
    """Extract contour for a specific flame position (L, C, or R)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    _, mask_s = cv2.threshold(s, SAT_THRESH, 255, cv2.THRESH_BINARY)
    
    b, g, r = cv2.split(frame)
    _, mask_b = cv2.threshold(b, BLUE_THRESH, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask_s, mask_b)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    
    if len(contours) >= 3:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        return contours[label_position]  # 0=L, 1=C, 2=R
    return None

def ray_plane_intersection(ray_origin, ray_dir, plane_point, plane_normal):
    """
    Find intersection of ray with plane
    Returns: intersection point or None if no intersection
    """
    denom = np.dot(ray_dir, plane_normal)
    if abs(denom) < 1e-6:
        return None  # Ray parallel to plane
    
    t = np.dot(plane_point - ray_origin, plane_normal) / denom
    if t < 0:
        return None  # Intersection behind ray origin
    
    return ray_origin + t * ray_dir

def ray_cylinder_intersection(ray_origin, ray_dir, cylinder_center_y, cylinder_radius, z_min=0, z_max=10):
    """
    Find intersection of ray with vertical cylinder
    Cylinder: center at (0, cylinder_center_y, any_z), radius, extends from z_min to z_max
    Returns: closest intersection point or None
    """
    # Cylinder equation: (x - 0)^2 + (y - cy)^2 = r^2
    # Ray: P = O + t*D
    # Substitute and solve quadratic for t
    
    ox, oy, oz = ray_origin
    dx, dy, dz = ray_dir
    cy = cylinder_center_y
    r = cylinder_radius
    
    # Quadratic coefficients: a*t^2 + b*t + c = 0
    a = dx**2 + dy**2
    b = 2*(ox*dx + (oy - cy)*dy)
    c = ox**2 + (oy - cy)**2 - r**2
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0 or abs(a) < 1e-10:
        return None  # No intersection
    
    # Two solutions
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    
    # Check both intersections, use the closer valid one
    for t in [t1, t2]:
        if t > 1e-6:  # Positive t (in front of ray)
            point = ray_origin + t * ray_dir
            if z_min <= point[2] <= z_max:  # Within cylinder height
                return point
    
    return None

def backproject_center_view(contour, Z_center, intrinsics):
    """
    Back-project center view contour to cylindrical surface at depth Z_center
    """
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    
    points_3d = []
    camera_pos = np.array([0.0, 0.0, 5.0])
    cylinder_radius = 1.5  # 3" diameter
    
    for point in contour:
        u, v = point[0]
        
        # Create ray from camera through pixel
        x_cam = (u - cx) / fx
        y_cam = 1.0
        z_cam = -(v - cy) / fy
        
        ray_dir = np.array([x_cam, y_cam, z_cam])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        # Find intersection with cylinder at Z_center
        intersection = ray_cylinder_intersection(camera_pos, ray_dir, Z_center, cylinder_radius)
        
        if intersection is not None:
            points_3d.append(intersection)
    
    return np.array(points_3d) if len(points_3d) > 0 else None

def backproject_mirror_view(contour, mirror_geom, Z_flame_center, intrinsics, is_left=True):
    """
    Back-project mirror view using ray tracing with reflection
    and intersection with cylindrical flame volume
    """
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    
    mirror_angle_deg = mirror_geom['angle_deg']
    mirror_distance = mirror_geom['distance_behind_in']
    mirror_h_offset = mirror_geom['horizontal_offset_in']
    
    cylinder_radius = 1.5  # 3" diameter tube
    mirror_y = Z_flame_center + mirror_distance
    
    if is_left:
        mirror_x = -abs(mirror_h_offset)
        angle_rad = np.radians(mirror_angle_deg)
        mirror_normal = np.array([np.sin(angle_rad), -np.cos(angle_rad), 0])
    else:
        mirror_x = abs(mirror_h_offset)
        angle_rad = np.radians(mirror_angle_deg)
        mirror_normal = np.array([-np.sin(angle_rad), -np.cos(angle_rad), 0])
    
    mirror_normal = mirror_normal / np.linalg.norm(mirror_normal)
    mirror_point = np.array([mirror_x, mirror_y, 5.0])
    camera_pos = np.array([0.0, 0.0, 5.0])
    
    points_3d = []
    
    for point in contour:
        u, v = point[0]
        
        # Create ray from camera through pixel
        x_cam = (u - cx) / fx
        y_cam = 1.0
        z_cam = -(v - cy) / fy
        
        ray_dir = np.array([x_cam, y_cam, z_cam])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        # Find intersection with mirror plane
        intersection = ray_plane_intersection(camera_pos, ray_dir, mirror_point, mirror_normal)
        
        if intersection is None:
            continue
        
        # Reflect ray off mirror
        d_dot_n = np.dot(ray_dir, mirror_normal)
        reflected_ray = ray_dir - 2 * d_dot_n * mirror_normal
        reflected_ray = reflected_ray / np.linalg.norm(reflected_ray)
        
        # Find where reflected ray intersects cylindrical flame volume
        flame_intersection = ray_cylinder_intersection(
            intersection, reflected_ray, Z_flame_center, cylinder_radius
        )
        
        if flame_intersection is not None:
            points_3d.append(flame_intersection)
    
    return np.array(points_3d) if len(points_3d) > 0 else None

def reconstruct_frame_3d(frame_number, calib):
    """Reconstruct 3D flame for a specific frame using ray tracing"""
    # Load frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Get calibration for this frame
    frame_data = None
    for fd in calib['frames']:
        if fd['frame'] == frame_number:
            frame_data = fd
            break
    
    if frame_data is None:
        return None
    
    intrinsics = calib['intrinsics']
    Z_center = frame_data['Z_center_abs_in']
    
    mirror_left = calib['mirror_left']
    mirror_right = calib['mirror_right']
    
    # Extract contours
    contour_L = extract_contour_from_frame(frame, 0)
    contour_C = extract_contour_from_frame(frame, 1)
    contour_R = extract_contour_from_frame(frame, 2)
    
    points_3d = {'L': None, 'C': None, 'R': None}
    
    if contour_C is not None:
        points_3d['C'] = backproject_center_view(contour_C, Z_center, intrinsics)
    
    if contour_L is not None:
        points_3d['L'] = backproject_mirror_view(contour_L, mirror_left, Z_center, intrinsics, is_left=True)
        if points_3d['L'] is not None:
            print(f"  Left mirror: {len(points_3d['L'])} points generated")
        else:
            print(f"  Left mirror: No valid intersections found")
    
    if contour_R is not None:
        points_3d['R'] = backproject_mirror_view(contour_R, mirror_right, Z_center, intrinsics, is_left=False)
        if points_3d['R'] is not None:
            print(f"  Right mirror: {len(points_3d['R'])} points generated")
        else:
            print(f"  Right mirror: No valid intersections found")
    
    return points_3d, {'Z_center': Z_center, 'mirror_distance': mirror_left['distance_behind_in']}

def plot_3d_flame(points_3d, frame_number, info, calib):
    """Visualize the 3D reconstructed flame"""
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    Z_center = info['Z_center']
    mirror_distance = info['mirror_distance']
    
    # Plot points from each view
    colors = {'L': 'blue', 'C': 'green', 'R': 'red'}
    labels = {'L': 'Left Mirror (reflected view)', 
              'C': f'Center Direct View', 
              'R': 'Right Mirror (reflected view)'}
    
    for view, points in points_3d.items():
        if points is not None and len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=colors[view], label=labels[view], s=15, alpha=0.7)
    
    # Draw tube boundaries at flame location
    theta = np.linspace(0, 2*np.pi, 50)
    tube_radius = 1.5
    
    for z in np.linspace(0, 10, 6):
        x_circle = tube_radius * np.cos(theta)
        y_circle = np.ones_like(theta) * Z_center
        z_circle = np.ones_like(theta) * z
        ax.plot(x_circle, y_circle, z_circle, 'gray', alpha=0.2, linewidth=0.5)
    
    # Draw mirror planes
    mirror_left = calib['mirror_left']
    mirror_right = calib['mirror_right']
    mirror_y = Z_center + mirror_distance
    
    # Left mirror visualization
    left_x = -abs(mirror_left['horizontal_offset_in'])
    left_angle = np.radians(mirror_left['angle_deg'])
    mirror_width = 4
    left_x1 = left_x - mirror_width/2 * np.cos(left_angle)
    left_x2 = left_x + mirror_width/2 * np.cos(left_angle)
    left_y1 = mirror_y - mirror_width/2 * np.sin(left_angle)
    left_y2 = mirror_y + mirror_width/2 * np.sin(left_angle)
    
    for z in np.linspace(0, 10, 6):
        ax.plot([left_x1, left_x2], [left_y1, left_y2], [z, z], 
                'b-', linewidth=1, alpha=0.3)
    
    # Right mirror visualization
    right_x = abs(mirror_right['horizontal_offset_in'])
    right_angle = np.radians(mirror_right['angle_deg'])
    right_x1 = right_x - mirror_width/2 * np.cos(right_angle)
    right_x2 = right_x + mirror_width/2 * np.cos(right_angle)
    right_y1 = mirror_y - mirror_width/2 * np.sin(right_angle)
    right_y2 = mirror_y + mirror_width/2 * np.sin(right_angle)
    
    for z in np.linspace(0, 10, 6):
        ax.plot([right_x1, right_x2], [right_y1, right_y2], [z, z], 
                'r-', linewidth=1, alpha=0.3)
    
    # Camera position indicator
    ax.scatter([0], [0], [5], c='black', s=200, marker='^', label='Camera')
    
    # Labels and formatting
    ax.set_xlabel('X (inches) - Horizontal', fontsize=11)
    ax.set_ylabel('Y (inches) - Depth from camera', fontsize=11)
    ax.set_zlabel('Z (inches) - Height', fontsize=11)
    ax.set_title(f'3D Flame Reconstruction with Ray Tracing - Frame {frame_number}', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    
    # Set viewing limits
    ax.set_xlim([-6, 6])
    ax.set_ylim([0, mirror_y + 3])
    ax.set_zlim([0, 10])
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("3D FLAME RECONSTRUCTION WITH RAY TRACING")
    print("="*70)
    
    # Load calibration
    calib = load_calibration(calibration_json)
    
    print(f"\nCalibration loaded:")
    print(f"  Camera distance: {calib['camera_distance_inches']:.1f} inches")
    print(f"  Left mirror: {calib['mirror_left']['angle_deg']:.1f}° at X={-calib['mirror_left']['horizontal_offset_in']:.2f}\"")
    print(f"  Right mirror: {calib['mirror_right']['angle_deg']:.1f}° at X={calib['mirror_right']['horizontal_offset_in']:.2f}\"")
    print(f"  Mirror distance behind: {calib['mirror_left']['distance_behind_in']}\"")
    
    # Find a good frame
    good_frames = []
    for fd in calib['frames']:
        if not np.isnan(fd.get('Z_left_rel', np.nan)) and not np.isnan(fd.get('Z_right_rel', np.nan)):
            good_frames.append(fd['frame'])
    
    if len(good_frames) > 0:
        frame_to_plot = good_frames[len(good_frames) // 2]
        print(f"\nUsing frame {frame_to_plot} (middle of {len(good_frames)} valid frames)")
    else:
        frame_to_plot = len(calib['frames']) // 2
        print(f"\nUsing frame {frame_to_plot}")
    
    # Reconstruct single frame
    result = reconstruct_frame_3d(frame_to_plot, calib)
    
    if result:
        points_3d, info = result
        
        print(f"\nReconstruction results:")
        for view in ['C', 'L', 'R']:
            if points_3d[view] is not None:
                print(f"  {view}: {len(points_3d[view])} points")
            else:
                print(f"  {view}: No points")
        
        # Visualize
        fig = plot_3d_flame(points_3d, frame_to_plot, info, calib)
        plt.savefig(r"H:\My Drive\Final Project\Data\flame_3d_reconstruction_raytraced.png", 
                   dpi=150, bbox_inches='tight')
        print(f"\n✓ 3D reconstruction saved!")
        plt.show()
    else:
        print("Could not reconstruct frame")
    
    print("\n" + "="*70)
    print("Set CREATE_ANIMATION = True to generate full video")
    print("="*70)