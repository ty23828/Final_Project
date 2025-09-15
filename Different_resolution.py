# This script generates three separate figures comparing grid resolutions over a 1°×1°
# window inside Portugal (lat 39–40°N, lon -9–-8°E). It saves PNGs to current directory.
# Modified Different_resolution.py - Display all three grids in one figure

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Region (choose a representative 1°×1° window) ---
lat_min, lat_max = 39.0, 40.0
lon_min, lon_max = -9.0, -8.0
lat0 = 0.5*(lat_min + lat_max)

# Helpers for km/deg conversions (approximate at mid-latitude)
km_per_deg_lat = 110.574  # ~ constant
km_per_deg_lon = 111.320 * np.cos(np.deg2rad(lat0))

def plot_single_grid(ax, res_lat_deg, res_lon_deg, title, color='blue', alpha=0.7):
    """Plot a single grid on given axes"""
    # Create arrays of grid lines
    eps = 1e-9
    lons = np.arange(lon_min, lon_max + eps, res_lon_deg)
    lats = np.arange(lat_min, lat_max + eps, res_lat_deg)

    # Draw vertical lines (longitudes)
    for x in lons:
        ax.plot([x, x], [lat_min, lat_max], color=color, linewidth=0.8, alpha=alpha)

    # Draw horizontal lines (latitudes)
    for y in lats:
        ax.plot([lon_min, lon_max], [y, y], color=color, linewidth=0.8, alpha=alpha)

    # Count grid cells for statistics
    n_cells_lat = len(lats) - 1 if len(lats) > 1 else 0
    n_cells_lon = len(lons) - 1 if len(lons) > 1 else 0
    total_cells = n_cells_lat * n_cells_lon
    
    return total_cells

# --- Define three resolutions ---
print("Generating combined grid resolution comparison plot...")
print("Region: Portugal (39°-40°N, 9°-8°W)")

# 50 km ≈ 0.5° (both lat & lon, for illustration)
res50_lat_deg = 0.5
res50_lon_deg = 0.5

# 25 km ≈ 0.25°
res25_lat_deg = 0.25
res25_lon_deg = 0.25

# 1 km: convert to degrees using km_per_deg at this latitude
res1_lat_deg = 1.0 / km_per_deg_lat
res1_lon_deg = 1.0 / km_per_deg_lon

print(f"\nGrid spacing calculations:")
print(f"   50km: {res50_lat_deg:.3f}° lat × {res50_lon_deg:.3f}° lon")
print(f"   25km: {res25_lat_deg:.3f}° lat × {res25_lon_deg:.3f}° lon") 
print(f"   1km:  {res1_lat_deg:.3f}° lat × {res1_lon_deg:.3f}° lon")

print(f"\nCreating combined comparison plot...")

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Grid Resolution Comparison - Portugal FWI Study Area', fontsize=16, fontweight='bold')

# Colors for different resolutions
colors = ['red', 'orange', 'blue']
alphas = [0.8, 0.7, 0.6]

# Plot 50km grid
cells_50 = plot_single_grid(axes[0], res50_lat_deg, res50_lon_deg, 
                           "50 km Resolution", colors[0], alphas[0])
axes[0].set_title(f'50 km Grid\n({cells_50} cells in 1°×1°)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Longitude (°)')
axes[0].set_ylabel('Latitude (°)')
axes[0].set_xlim(lon_min, lon_max)
axes[0].set_ylim(lat_min, lat_max)
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# Plot 25km grid
cells_25 = plot_single_grid(axes[1], res25_lat_deg, res25_lon_deg, 
                           "25 km Resolution", colors[1], alphas[1])
axes[1].set_title(f'25 km Grid\n({cells_25} cells in 1°×1°)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Longitude (°)')
axes[1].set_ylabel('Latitude (°)')
axes[1].set_xlim(lon_min, lon_max)
axes[1].set_ylim(lat_min, lat_max)
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)

# Plot 1km grid (show only a subset for visibility)
cells_1 = plot_single_grid(axes[2], res1_lat_deg, res1_lon_deg, 
                          "1 km Resolution", colors[2], alphas[2])
axes[2].set_title(f'1 km Grid\n({cells_1:,} cells in 1°×1°)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Longitude (°)')
axes[2].set_ylabel('Latitude (°)')
axes[2].set_xlim(lon_min, lon_max)
axes[2].set_ylim(lat_min, lat_max)
axes[2].set_aspect('equal')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# Save combined plot
combined_filename = "grid_resolution_comparison.png"
plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
print(f"Combined plot saved: {combined_filename}")

# Show the plot
plt.show()

# Create an overlay comparison plot (all grids on one axis)
print(f"\nCreating overlay comparison plot...")

fig2, ax = plt.subplots(1, 1, figsize=(10, 10))

# Plot all three grids on the same axes with different colors
plot_single_grid(ax, res50_lat_deg, res50_lon_deg, "50 km", 'red', 0.9)
plot_single_grid(ax, res25_lat_deg, res25_lon_deg, "25 km", 'orange', 0.7)
plot_single_grid(ax, res1_lat_deg, res1_lon_deg, "1 km", 'blue', 0.5)

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel('Longitude (°)', fontsize=12)
ax.set_ylabel('Latitude (°)', fontsize=12)
ax.set_title('Grid Resolution Overlay Comparison\nPortugal FWI Study Area (39°-40°N, 9°-8°W)', 
             fontsize=14, fontweight='bold')
ax.set_aspect('equal')

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=2, alpha=0.9, label='50 km grid'),
    Line2D([0], [0], color='orange', linewidth=2, alpha=0.7, label='25 km grid'),
    Line2D([0], [0], color='blue', linewidth=2, alpha=0.5, label='1 km grid')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()

# Save overlay plot
overlay_filename = "grid_resolution_overlay.png"
plt.savefig(overlay_filename, dpi=300, bbox_inches='tight')
print(f"Overlay plot saved: {overlay_filename}")

plt.show()

# Create summary table
summary = pd.DataFrame({
    "Resolution": ["50 km", "25 km", "1 km"],
    "Lat spacing (deg)": [res50_lat_deg, res25_lat_deg, res1_lat_deg],
    "Lon spacing (deg)": [res50_lon_deg, res25_lon_deg, res1_lon_deg],
    "Lat spacing (km)": [res50_lat_deg*km_per_deg_lat, res25_lat_deg*km_per_deg_lat, 1.0],
    "Lon spacing (km)": [res50_lon_deg*km_per_deg_lon, res25_lon_deg*km_per_deg_lon, 1.0],
    "Cells in 1°×1°": [cells_50, cells_25, cells_1],
    "Enhancement vs 50km": [1, cells_25/cells_50, cells_1/cells_50]
}).round(6)

print(f"\nGrid spacing summary (Portugal 39.5°N):")
print(summary.to_string(index=False))

# Save summary to CSV
summary.to_csv("grid_spacing_summary.csv", index=False)
print(f"\nSummary saved to: grid_spacing_summary.csv")

print(f"\nAll plots generated successfully!")
print(f"Files created in current directory:")
print(f"   {combined_filename} (side-by-side comparison)")
print(f"   {overlay_filename} (overlay comparison)")
print(f"   grid_spacing_summary.csv (detailed statistics)")

# Enhanced statistics
print(f"\nResolution Enhancement Statistics:")
print(f"   25km vs 50km: {cells_25/cells_50:.1f}x more detail")
print(f"   1km vs 25km: {cells_1/cells_25:.0f}x more detail") 
print(f"   1km vs 50km: {cells_1/cells_50:.0f}x more detail")
print(f"   Total 1km cells in Portugal study area: ~{cells_1:,}")
