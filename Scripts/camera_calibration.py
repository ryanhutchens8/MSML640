import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the bounding box data
csv_path = r"H:\My Drive\Final Project\Data\blob_bounding_boxes.csv"
df = pd.read_csv(csv_path)

# Known physical constraints
TUBE_DIAMETER_INCHES = 3.0
MIRROR_DISTANCE_INCHES = 9.0

# Separate data by view
df_L = df[df['label'] == 'L'].copy()
df_C = df[df['label'] == 'C'].copy()
df_R = df[df['label'] == 'R'].copy()

print("="*60)
print("CAMERA CALIBRATION ANALYSIS")
print("="*60)
print(f"Total frames analyzed: {df['frame'].max() + 1}")
print(f"Total detections: {len(df)} ({len(df_L)} L, {len(df_C)} C, {len(df_R)} R)")

# --- Analysis 1: Basic Statistics ---
print("\n1. BOUNDING BOX STATISTICS")
print("-"*60)

for label, df_view, color_name in [('L', df_L, 'Left'), 
                                     ('C', df_C, 'Center'), 
                                     ('R', df_R, 'Right')]:
    print(f"\n{color_name} view:")
    print(f"  Detections: {len(df_view)}")
    print(f"  Width:  mean={df_view['w'].mean():.1f}px, std={df_view['w'].std():.1f}px, range=[{df_view['w'].min()}-{df_view['w'].max()}]")
    print(f"  Height: mean={df_view['h'].mean():.1f}px, std={df_view['h'].std():.1f}px, range=[{df_view['h'].min()}-{df_view['h'].max()}]")
    print(f"  Area:   mean={df_view['area'].mean():.0f}px², std={df_view['area'].std():.0f}px²")
    print(f"  Aspect ratio (h/w): {df_view['h'].mean() / df_view['w'].mean():.2f}")
    if len(df_view) > 0:
        shape_counts = df_view['shape'].value_counts()
        print(f"  Shapes: {dict(shape_counts)}")

# --- Analysis 2: Pixel-to-Physical Scaling ---
print("\n2. PIXEL-TO-PHYSICAL SCALING")
print("-"*60)

# Use center view width as reference for 3" tube diameter
median_width_C = df_C['w'].median()
mean_width_C = df_C['w'].mean()

pixels_per_inch_median = median_width_C / TUBE_DIAMETER_INCHES
pixels_per_inch_mean = mean_width_C / TUBE_DIAMETER_INCHES

print(f"\nCenter view horizontal width (should represent ~3\" tube):")
print(f"  Median width: {median_width_C:.2f} pixels")
print(f"  Mean width: {mean_width_C:.2f} pixels")
print(f"  Scaling (median): {pixels_per_inch_median:.2f} pixels/inch")
print(f"  Scaling (mean): {pixels_per_inch_mean:.2f} pixels/inch")
print(f"  Inverse: {1/pixels_per_inch_median:.4f} inches/pixel")

# Use the median for more robust estimate
pixels_per_inch = pixels_per_inch_median

# --- Analysis 3: Perspective Analysis ---
print("\n3. PERSPECTIVE & MIRROR GEOMETRY")
print("-"*60)

# Compare widths across views
print(f"\nWidth comparison (should show perspective effects):")
print(f"  Center mean width: {df_C['w'].mean():.1f}px")
print(f"  Left mean width:   {df_L['w'].mean():.1f}px ({df_L['w'].mean()/df_C['w'].mean()*100:.1f}% of center)")
print(f"  Right mean width:  {df_R['w'].mean():.1f}px ({df_R['w'].mean()/df_C['w'].mean()*100:.1f}% of center)")

# Correlation between vertical position and width
for label, df_view in [('Left', df_L), ('Right', df_R)]:
    if len(df_view) > 1:
        corr = df_view[['cy', 'w']].corr().iloc[0, 1]
        print(f"\n{label} mirror - correlation(cy, width): {corr:.3f}")
        if abs(corr) > 0.3:
            print(f"  → Strong correlation suggests perspective effect")

# --- Analysis 4: Mirror Angle Estimation ---
print("\n4. MIRROR ANGLE ESTIMATION")
print("-"*60)

def perspective_model(cy, scale_factor, perspective_strength):
    """
    Simplified model: width decreases with vertical position due to perspective
    cy = vertical pixel position (centroid)
    scale_factor = base width
    perspective_strength = how much perspective affects width
    """
    return scale_factor / (1 + perspective_strength * cy / 1000)

for view_name, df_view in [('Left', df_L), ('Right', df_R)]:
    df_valid = df_view.dropna(subset=['cy', 'w'])
    
    if len(df_valid) > 10:
        try:
            popt, pcov = curve_fit(
                perspective_model,
                df_valid['cy'],
                df_valid['w'],
                p0=[df_valid['w'].mean(), 0.5],
                bounds=([10, -2], [200, 2])
            )
            scale_factor, perspective_strength = popt
            
            # Estimate mirror angle from perspective strength
            # Higher perspective strength → steeper mirror angle
            estimated_angle = np.arctan(perspective_strength * 10) * 180 / np.pi
            
            print(f"\n{view_name} mirror:")
            print(f"  Scale factor: {scale_factor:.1f}px")
            print(f"  Perspective strength: {perspective_strength:.4f}")
            print(f"  Estimated mirror angle: {abs(estimated_angle):.1f}°")
            
        except Exception as e:
            print(f"\n{view_name} mirror: Could not fit model - {e}")

# --- Analysis 5: Centroid Positions ---
print("\n5. CENTROID ANALYSIS")
print("-"*60)

print(f"\nHorizontal positions (cx):")
print(f"  Left:   mean={df_L['cx'].mean():.1f}px")
print(f"  Center: mean={df_C['cx'].mean():.1f}px")
print(f"  Right:  mean={df_R['cx'].mean():.1f}px")
print(f"  Spacing L-C: {df_C['cx'].mean() - df_L['cx'].mean():.1f}px")
print(f"  Spacing C-R: {df_R['cx'].mean() - df_C['cx'].mean():.1f}px")

print(f"\nVertical positions (cy):")
print(f"  Left:   mean={df_L['cy'].mean():.1f}px, std={df_L['cy'].std():.1f}px")
print(f"  Center: mean={df_C['cy'].mean():.1f}px, std={df_C['cy'].std():.1f}px")
print(f"  Right:  mean={df_R['cy'].mean():.1f}px, std={df_R['cy'].std():.1f}px")

# --- Visualization ---
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Width vs Vertical Position (cy)
for idx, (label, df_view, color) in enumerate([('L', df_L, 'blue'), 
                                                 ('C', df_C, 'green'), 
                                                 ('R', df_R, 'red')]):
    ax = fig.add_subplot(gs[0, idx])
    df_valid = df_view.dropna(subset=['cy', 'w'])
    ax.scatter(df_valid['cy'], df_valid['w'], alpha=0.4, s=15, c=color)
    
    # Add trend line
    if len(df_valid) > 1:
        z = np.polyfit(df_valid['cy'], df_valid['w'], 1)
        p = np.poly1d(z)
        cy_range = np.linspace(df_valid['cy'].min(), df_valid['cy'].max(), 100)
        ax.plot(cy_range, p(cy_range), 'k--', linewidth=2, alpha=0.7, label=f'Slope: {z[0]:.3f}')
        ax.legend(fontsize=9)
    
    ax.set_xlabel('Vertical Position (cy, pixels)', fontsize=10)
    ax.set_ylabel('Width (pixels)', fontsize=10)
    ax.set_title(f'{label} View: Width vs Height', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Row 2: Width and Height Over Time
for idx, (label, df_view, color) in enumerate([('L', df_L, 'blue'), 
                                                 ('C', df_C, 'green'), 
                                                 ('R', df_R, 'red')]):
    ax = fig.add_subplot(gs[1, idx])
    ax.scatter(df_view['frame'], df_view['w'], alpha=0.4, s=10, c=color, label='Width')
    ax.scatter(df_view['frame'], df_view['h'], alpha=0.4, s=10, c='orange', label='Height')
    ax.axhline(df_view['w'].mean(), color=color, linestyle='--', linewidth=1.5, 
               label=f'Mean W: {df_view["w"].mean():.1f}px')
    ax.set_xlabel('Frame Number', fontsize=10)
    ax.set_ylabel('Dimension (pixels)', fontsize=10)
    ax.set_title(f'{label} View: Dimensions Over Time', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

# Row 3: Width Distribution and Centroid Positions
ax1 = fig.add_subplot(gs[2, 0])
ax1.hist([df_L['w'], df_C['w'], df_R['w']], bins=20, alpha=0.6, 
         label=['Left', 'Center', 'Right'], color=['blue', 'green', 'red'])
ax1.set_xlabel('Width (pixels)', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.set_title('Width Distribution Comparison', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

ax2 = fig.add_subplot(gs[2, 1])
ax2.scatter(df_L['cx'], df_L['cy'], alpha=0.3, s=5, c='blue', label='Left')
ax2.scatter(df_C['cx'], df_C['cy'], alpha=0.3, s=5, c='green', label='Center')
ax2.scatter(df_R['cx'], df_R['cy'], alpha=0.3, s=5, c='red', label='Right')
ax2.set_xlabel('Horizontal Position (cx)', fontsize=10)
ax2.set_ylabel('Vertical Position (cy)', fontsize=10)
ax2.set_title('Centroid Positions', fontsize=11, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.invert_yaxis()  # Image coordinates: y increases downward

ax3 = fig.add_subplot(gs[2, 2])
aspect_L = df_L['h'] / df_L['w']
aspect_C = df_C['h'] / df_C['w']
aspect_R = df_R['h'] / df_R['w']
ax3.hist([aspect_L, aspect_C, aspect_R], bins=20, alpha=0.6,
         label=['Left', 'Center', 'Right'], color=['blue', 'green', 'red'])
ax3.set_xlabel('Aspect Ratio (h/w)', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('Aspect Ratio Distribution', fontsize=11, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

plt.savefig(r"H:\My Drive\Final Project\Data\calibration_analysis.png", dpi=150, bbox_inches='tight')
print(f"\nCalibration plots saved to: H:\\My Drive\\Final Project\\Data\\calibration_analysis.png")
plt.show()

# --- Summary Output ---
print("\n" + "="*60)
print("CALIBRATION SUMMARY")
print("="*60)
print(f"Pixels per inch (from center view): {pixels_per_inch:.2f}")
print(f"Physical resolution: {1/pixels_per_inch:.4f} inches/pixel")
print(f"Mirror distance: {MIRROR_DISTANCE_INCHES} inches (known)")
print(f"Tube diameter: {TUBE_DIAMETER_INCHES} inches (known)")
print(f"\nWidth ratios (relative to center):")
print(f"  Left/Center:  {df_L['w'].mean()/df_C['w'].mean():.3f}")
print(f"  Right/Center: {df_R['w'].mean()/df_C['w'].mean():.3f}")
print("="*60)