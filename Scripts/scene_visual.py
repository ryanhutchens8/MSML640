import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Visualize the 3D scene geometry from enhanced calibration
Shows camera, flame, and angled mirrors based on calculated parameters
"""

# Load calibration
calibration_json = r"H:\My Drive\Final Project\Data\calibration_from_blobs.json"

with open(calibration_json, 'r') as f:
    calib = json.load(f)

print("="*70)
print("SCENE GEOMETRY FROM CALIBRATION")
print("="*70)

# Extract parameters
img_width, img_height = calib['image_size_px']
ppi = calib['pixels_per_inch_center_median']
Z_camera = calib['camera_distance_inches']

intrinsics = calib['intrinsics']
fx = intrinsics['fx']
fy = intrinsics['fy']
cx = intrinsics['cx']
cy = intrinsics['cy']

mirror_left = calib['mirror_left']
mirror_right = calib['mirror_right']

left_angle = mirror_left['angle_deg']
left_offset = mirror_left['horizontal_offset_in']
mirror_distance = mirror_left['distance_behind_in']

right_angle = mirror_right['angle_deg']
right_offset = mirror_right['horizontal_offset_in']

print(f"\nCalibration Parameters:")
print(f"  Camera distance: {Z_camera:.1f} inches from flame")
print(f"  Pixel scaling: {ppi:.2f} px/inch")
print(f"  Focal length: fx={fx:.1f}, fy={fy:.1f} pixels")
print(f"\nMirror Geometry:")
print(f"  Distance behind flame: {mirror_distance} inches")
print(f"  Left mirror: {left_angle:.1f}° angle, {left_offset:.2f}\" offset")
print(f"  Right mirror: {right_angle:.1f}° angle, {right_offset:.2f}\" offset")

# Calculate field of view
fov_h_deg = 2 * np.degrees(np.arctan(img_width / (2 * fx)))
fov_v_deg = 2 * np.degrees(np.arctan(img_height / (2 * fy)))

print(f"\nField of View:")
print(f"  Horizontal: {fov_h_deg:.1f}°")
print(f"  Vertical: {fov_v_deg:.1f}°")

# ============================================================================
# Create visualization
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# ============================================================================
# Plot 1: Top view (X-Y plane)
# ============================================================================
ax1 = plt.subplot(2, 3, 1)

# Camera at origin
ax1.plot(0, 0, 'k^', markersize=15, label='Camera', zorder=5)

# Flame tube at Z_camera distance
flame_y = Z_camera
tube_radius = 1.5  # 3" diameter
theta = np.linspace(0, 2*np.pi, 50)
tube_x = tube_radius * np.cos(theta)
tube_y = tube_radius * np.sin(theta) + flame_y
ax1.fill(tube_x, tube_y, color='orange', alpha=0.3, label='Flame tube')
ax1.plot(tube_x, tube_y, 'r-', linewidth=2)
ax1.plot(0, flame_y, 'ro', markersize=10)

# Mirrors behind flame with calculated angles
mirror_y = flame_y + mirror_distance

# Left mirror (angled inward from left side)
# Mirror is at angle θ from perpendicular, positioned at negative X
left_mirror_center_x = -left_offset
left_mirror_length = 6  # visual length
left_angle_rad = np.radians(left_angle)
# Mirror normal vector
left_nx = np.sin(left_angle_rad)
left_ny = np.cos(left_angle_rad)
# Mirror endpoints
left_x1 = left_mirror_center_x - left_mirror_length/2 * np.cos(left_angle_rad)
left_x2 = left_mirror_center_x + left_mirror_length/2 * np.cos(left_angle_rad)
left_y1 = mirror_y - left_mirror_length/2 * np.sin(left_angle_rad)
left_y2 = mirror_y + left_mirror_length/2 * np.sin(left_angle_rad)
ax1.plot([left_x1, left_x2], [left_y1, left_y2], 'b-', linewidth=5, 
         label=f'Left mirror ({left_angle:.0f}°)', zorder=3)

# Right mirror (angled inward from right side)
right_mirror_center_x = right_offset
right_mirror_length = 6  # visual length (same as left)
right_angle_rad = np.radians(right_angle)
# For right mirror, we need to flip the angle direction
right_x1 = right_mirror_center_x - right_mirror_length/2 * np.cos(right_angle_rad)
right_x2 = right_mirror_center_x + right_mirror_length/2 * np.cos(right_angle_rad)
right_y1 = mirror_y + right_mirror_length/2 * np.sin(right_angle_rad)
right_y2 = mirror_y - right_mirror_length/2 * np.sin(right_angle_rad)
ax1.plot([right_x1, right_x2], [right_y1, right_y2], 'r-', linewidth=5, 
         label=f'Right mirror ({right_angle:.0f}°)', zorder=3)

# Draw sight lines from camera to mirrors
ax1.plot([0, (left_x1+left_x2)/2], [0, (left_y1+left_y2)/2], 'b--', alpha=0.3, linewidth=1)
ax1.plot([0, (right_x1+right_x2)/2], [0, (right_y1+right_y2)/2], 'r--', alpha=0.3, linewidth=1)
ax1.plot([0, 0], [0, flame_y], 'g--', alpha=0.3, linewidth=1, label='Direct view')

# Draw field of view cone
fov_angle = np.radians(fov_h_deg / 2)
fov_dist = mirror_y + 5
ax1.plot([0, fov_dist * np.sin(fov_angle)], [0, fov_dist * np.cos(fov_angle)], 
         'k:', alpha=0.2, linewidth=1)
ax1.plot([0, -fov_dist * np.sin(fov_angle)], [0, fov_dist * np.cos(fov_angle)], 
         'k:', alpha=0.2, linewidth=1)

ax1.set_xlabel('X (inches) - Horizontal', fontsize=11)
ax1.set_ylabel('Y (inches) - Distance from camera', fontsize=11)
ax1.set_title(f'Top View\nCamera FOV: {fov_h_deg:.1f}° horizontal', 
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axis('equal')
ax1.set_xlim([-12, 12])
ax1.set_ylim([-2, fov_dist])

# ============================================================================
# Plot 2: Side view (Y-Z plane) - Camera looking HORIZONTALLY
# ============================================================================
ax2 = plt.subplot(2, 3, 2)

# Camera at origin looking horizontally (along Y axis)
# Z axis is vertical (up/down), Y axis is depth (away from camera)
ax2.plot(0, 5, 'k^', markersize=15, label='Camera')

# Flame tube (vertical cylinder, flame travels bottom to top)
ax2.fill([flame_y-0.5, flame_y+0.5, flame_y+0.5, flame_y-0.5], 
         [0, 0, 10, 10], color='orange', alpha=0.3)
ax2.plot([flame_y, flame_y], [0, 10], 'r-', linewidth=3, label='Flame tube (vertical)')
ax2.plot(flame_y, 5, 'ro', markersize=10)
ax2.arrow(flame_y-1, 2, 0, 3, head_width=0.5, head_length=0.3, 
         fc='orange', ec='orange', alpha=0.5, linewidth=1)
ax2.text(flame_y-2, 3.5, 'Flame\nrises', fontsize=8, ha='right')

# Mirrors (vertical lines at mirror_y)
ax2.plot([mirror_y, mirror_y], [0, 10], 'purple', linewidth=3, 
         label=f'Mirrors ({mirror_distance}" behind)')

# Optical paths at mid-height (camera looks horizontally at z=5)
camera_height = 5
ax2.plot([0, flame_y], [camera_height, camera_height], 'g--', alpha=0.5, 
         linewidth=2, label='Direct path')
ax2.plot([0, mirror_y], [camera_height, camera_height], 'b--', alpha=0.3, linewidth=1)
ax2.plot([mirror_y, flame_y], [camera_height, camera_height], 'b--', alpha=0.3, linewidth=1)

# Vertical FOV (camera sees from bottom to top of frame)
fov_v_angle = np.radians(fov_v_deg / 2)
fov_v_dist = flame_y
ax2.plot([0, fov_v_dist], [camera_height, camera_height + fov_v_dist * np.tan(fov_v_angle)], 
         'k:', alpha=0.2)
ax2.plot([0, fov_v_dist], [camera_height, camera_height - fov_v_dist * np.tan(fov_v_angle)], 
         'k:', alpha=0.2)

ax2.set_xlabel('Y (inches) - Distance from camera', fontsize=11)
ax2.set_ylabel('Z (inches) - Height (flame rises)', fontsize=11)
ax2.set_title(f'Side View (Camera Looking Horizontally)\nVertical FOV: {fov_v_deg:.1f}°', 
              fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-2, mirror_y + 5])
ax2.set_ylim([0, 10])

# ============================================================================
# Plot 3: 3D view
# ============================================================================
ax3 = plt.subplot(2, 3, 3, projection='3d')

# Camera
ax3.scatter([0], [0], [5], c='black', s=300, marker='^', label='Camera')

# Flame tube (cylinder)
z_tube = np.linspace(0, 10, 20)
for z in [0, 2.5, 5, 7.5, 10]:
    x_circle = tube_radius * np.cos(theta)
    y_circle = tube_radius * np.sin(theta) + flame_y
    z_circle = np.ones_like(theta) * z
    ax3.plot(x_circle, y_circle, z_circle, 'orange', alpha=0.4, linewidth=1)

# Draw top and bottom circles solid
ax3.plot(tube_x, tube_y, np.zeros_like(tube_x), 'r-', linewidth=2)
ax3.plot(tube_x, tube_y, np.ones_like(tube_x)*10, 'r-', linewidth=2)

# Left mirror (vertical plane at angle)
mirror_height = np.linspace(0, 10, 10)
for h in mirror_height:
    ax3.plot([left_x1, left_x2], [left_y1, left_y2], [h, h], 'b-', alpha=0.6, linewidth=2)

# Right mirror (vertical plane at angle)
for h in mirror_height:
    ax3.plot([right_x1, right_x2], [right_y1, right_y2], [h, h], 'r-', alpha=0.6, linewidth=2)

# Sight lines
ax3.plot([0, 0], [0, flame_y], [5, 5], 'g--', alpha=0.5, linewidth=2, label='Direct')
ax3.plot([0, left_mirror_center_x], [0, mirror_y], [5, 5], 'b--', alpha=0.3, linewidth=1)
ax3.plot([0, right_mirror_center_x], [0, mirror_y], [5, 5], 'r--', alpha=0.3, linewidth=1)

ax3.set_xlabel('X (inches)', fontsize=10)
ax3.set_ylabel('Y (inches)', fontsize=10)
ax3.set_zlabel('Z (inches)', fontsize=10)
ax3.set_title('3D Scene Geometry', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.view_init(elev=15, azim=45)
ax3.set_xlim([-8, 8])
ax3.set_ylim([0, mirror_y+3])
ax3.set_zlim([0, 10])

# ============================================================================
# Plot 4: Optical paths diagram
# ============================================================================
ax4 = plt.subplot(2, 3, 4)

# Distances (mirrors should have same optical path length)
distances = ['Center\n(Direct)', 'Left\nMirror', 'Right\nMirror']
path_lengths = [Z_camera, Z_camera + 2*mirror_distance, Z_camera + 2*mirror_distance]
colors = ['green', 'blue', 'red']

bars = ax4.barh(distances, path_lengths, color=colors, alpha=0.7)

# Add distance labels
for bar, dist in zip(bars, path_lengths):
    width = bar.get_width()
    ax4.text(width + 1, bar.get_y() + bar.get_height()/2, 
            f'{dist:.1f}"', ha='left', va='center', fontweight='bold')

# Add breakdown for mirror paths
mirror_path = Z_camera + 2*mirror_distance
ax4.text(Z_camera/2, 1.3, f'{Z_camera:.1f}"', ha='center', fontsize=8, color='gray')
ax4.text(Z_camera + mirror_distance, 1.3, f'+{2*mirror_distance:.1f}"', ha='center', fontsize=8, color='gray')

ax4.set_xlabel('Optical Path Length (inches)', fontsize=11)
ax4.set_title('Optical Path Comparison', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')
ax4.set_xlim([0, max(path_lengths) + 5])

# ============================================================================
# Plot 5: Mirror angle diagram
# ============================================================================
ax5 = plt.subplot(2, 3, 5)

# Draw from above showing mirror angles
ax5.plot([0], [0], 'k^', markersize=15)
ax5.text(0, -2, 'Camera', ha='center', fontsize=10, fontweight='bold')

# Perpendicular reference line
ax5.plot([0, 0], [0, 25], 'k--', alpha=0.3, linewidth=1, label='Perpendicular')

# Left mirror ray (negative X, angled toward perpendicular)
left_ray_x = -25 * np.sin(np.radians(left_angle))
left_ray_y = 25 * np.cos(np.radians(left_angle))
ax5.arrow(0, 0, left_ray_x*0.9, left_ray_y*0.9, head_width=1, head_length=1, 
         fc='blue', ec='blue', linewidth=2, label=f'Left ({left_angle:.1f}°)')

# Right mirror ray (positive X, angled toward perpendicular)
right_ray_x = 25 * np.sin(np.radians(right_angle))
right_ray_y = 25 * np.cos(np.radians(right_angle))
ax5.arrow(0, 0, right_ray_x*0.9, right_ray_y*0.9, head_width=1, head_length=1, 
         fc='red', ec='red', linewidth=2, label=f'Right ({right_angle:.1f}°)')

# Draw angle arcs
arc_radius = 8
# Left arc (from perpendicular toward left)
arc_left = np.linspace(0, -np.radians(left_angle), 30)
ax5.plot(arc_radius * np.sin(arc_left), arc_radius * np.cos(arc_left), 'b-', linewidth=1.5)
ax5.text(-3, 7, f'{left_angle:.1f}°', color='blue', fontsize=9, fontweight='bold')

# Right arc (from perpendicular toward right)
arc_right = np.linspace(0, np.radians(right_angle), 30)
ax5.plot(arc_radius * np.sin(arc_right), arc_radius * np.cos(arc_right), 'r-', linewidth=1.5)
ax5.text(3, 7, f'{right_angle:.1f}°', color='red', fontsize=9, fontweight='bold')

ax5.set_xlabel('X (inches)', fontsize=11)
ax5.set_ylabel('Y (inches)', fontsize=11)
ax5.set_title('Mirror Angles (Top View)', fontsize=12, fontweight='bold')
ax5.legend(loc='upper right', fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.axis('equal')
ax5.set_xlim([-15, 15])
ax5.set_ylim([-5, 30])

# ============================================================================
# Plot 6: Summary
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary = f"""
CALIBRATED SCENE GEOMETRY

Setup Configuration:
  • Camera: {Z_camera:.1f}" from flame center
  • Mirrors: {mirror_distance}" behind flame
  • Total distance: {Z_camera + mirror_distance:.1f}"

Mirror Geometry:
  • Left: {left_angle:.1f}° angle, {abs(left_offset):.2f}" left
  • Right: {right_angle:.1f}° angle, {right_offset:.2f}" right
  • Average angle: {(left_angle+right_angle)/2:.1f}°
  • V-shaped, angled horizontally

Optical Paths:
  • Direct (center): {Z_camera:.1f}"
  • Mirror views: {Z_camera + 2*mirror_distance:.1f}"
  • Path ratio: {(Z_camera + 2*mirror_distance)/Z_camera:.2f}×

Camera Parameters:
  • FOV: {fov_h_deg:.1f}° × {fov_v_deg:.1f}°
  • Resolution: {img_width} × {img_height} px
  • Focal: fx={fx:.0f}, fy={fy:.0f} px
  • Scaling: {ppi:.2f} px/inch

This geometry matches your physical setup
with vertically-oriented mirrors angled
inward in a V-shape.
"""

ax6.text(0.1, 0.5, summary, fontsize=10, family='monospace',
        verticalalignment='center', 
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig(r"H:\My Drive\Final Project\Data\scene_geometry_from_calibration.png", 
           dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved!")
plt.show()

print("\n" + "="*70)
print("SCENE VALIDATION")
print("="*70)
print(f"\nThe calibration predicts:")
print(f"  • Camera {Z_camera:.0f}\" from flame")
print(f"  • Mirrors at {left_angle:.0f}° and {right_angle:.0f}° angles")
print(f"  • {fov_h_deg:.0f}° field of view captures all three views")
print(f"\nDoes this match your physical setup?")
print("="*70)