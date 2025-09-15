import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.colors as colors
from matplotlib.patches import Rectangle

def create_fwi_spatial_maps():
    """Create spatial distribution maps for both real and predicted FWI data"""
    
    print("Creating FWI Spatial Distribution Maps...")
    
    # Load data
    try:
        # Real FWI data (25km resolution)
        real_data = pd.read_csv("experiment/ERA5_reanalysis_fwi/era5_fwi_2017_portugal_3decimal.csv")
        real_data['time'] = pd.to_datetime(real_data['time'])
        print(f"   Real data loaded: {real_data.shape}")
        
        # Predicted FWI data (1km resolution)  
        pred_data = pd.read_csv("fwi_1km_predictions_random_forest.csv")
        if 'date' in pred_data.columns:
            pred_data['time'] = pd.to_datetime(pred_data['date'])
        elif 'time' in pred_data.columns:
            pred_data['time'] = pd.to_datetime(pred_data['time'])
        print(f"   Predicted data loaded: {pred_data.shape}")
        
    except Exception as e:
        print(f"ERROR: Error loading data: {e}")
        return
    
    # Specify high FWI date (July 21, 2017)
    target_date = datetime(2017, 7, 21).date()
    print(f"   Target high FWI date: {target_date}")
    
    # Check if target date exists in both datasets
    real_dates = set(real_data['time'].dt.date)
    pred_dates = set(pred_data['time'].dt.date)
    
    if target_date in real_dates and target_date in pred_dates:
        selected_dates = {'common': [target_date]}
        print(f"   Target date found in both datasets")
        
        # Check FWI levels for this date
        real_day = real_data[real_data['time'].dt.date == target_date]
        if len(real_day) > 0:
            mean_fwi = real_day['fwi'].mean()
            max_fwi = real_day['fwi'].max()
            high_risk_count = (real_day['fwi'] > 24).sum()
            print(f"   July 21 FWI stats - Mean: {mean_fwi:.1f}, Max: {max_fwi:.1f}, High Risk: {high_risk_count}")
    else:
        print(f"   WARNING: Target date {target_date} not found in datasets")
        print(f"   Available real dates: {sorted(real_dates)[:5]}...")
        print(f"   Available pred dates: {sorted(pred_dates)[:5]}...")
        
        # Fall back to finding highest FWI date
        common_dates = sorted(real_dates.intersection(pred_dates))
        if len(common_dates) == 0:
            print("WARNING: No common dates found between datasets")
            return
        
        # Find date with highest mean FWI
        date_stats = []
        for date in common_dates:
            real_day = real_data[real_data['time'].dt.date == date]
            if len(real_day) > 0:
                mean_fwi = real_day['fwi'].mean()
                max_fwi = real_day['fwi'].max()
                date_stats.append((date, mean_fwi, max_fwi))
        
        # Sort by mean FWI and select highest
        date_stats.sort(key=lambda x: x[1], reverse=True)
        highest_fwi_date = date_stats[0][0]
        selected_dates = {'common': [highest_fwi_date]}
        print(f"   Using highest FWI date instead: {highest_fwi_date} (Mean FWI: {date_stats[0][1]:.1f})")
    
    print(f"   Selected dates for visualization: {selected_dates}")
    
    # Create comprehensive spatial visualization
    _create_comprehensive_maps(real_data, pred_data, selected_dates)
    _create_comparison_maps(real_data, pred_data, selected_dates)
    _create_high_resolution_detail_maps(pred_data, selected_dates)
    _create_aggregated_comparison_maps(real_data, pred_data, selected_dates)

def _create_comprehensive_maps(real_data, pred_data, selected_dates):
    """Create comprehensive spatial maps with aggregated comparison"""
    
    # Portugal extent
    lon_min, lon_max = -10.5, -5.5
    lat_min, lat_max = 35.5, 43.5
    
    # Custom FWI colormap
    fwi_colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF8C00', '#FF4500', '#DC143C', '#8B0000']
    fwi_levels = [0, 5, 12, 24, 38, 50, 75, 100]
    fwi_cmap = colors.ListedColormap(fwi_colors)
    fwi_norm = colors.BoundaryNorm(fwi_levels, fwi_cmap.N)
    
    # Create figure with 3 columns (Real, Aggregated, Predicted)
    fig = plt.figure(figsize=(24, 22))
    
    # Process dates
    if 'common' in selected_dates:
        dates_to_plot = selected_dates['common']
        plot_both = True
    else:
        dates_to_plot = selected_dates.get('real', []) + selected_dates.get('pred', [])
        plot_both = False
    
    n_dates = min(len(dates_to_plot), 3)  # Maximum 3 dates
    
    for i, date in enumerate(dates_to_plot[:n_dates]):
        
        print(f"   Creating maps for {date}...")
        
        # Filter data for this date
        real_day = real_data[real_data['time'].dt.date == date]
        pred_day = pred_data[pred_data['time'].dt.date == date]
        
        # Get FWI column name for predicted data
        fwi_col = 'fwi_predicted' if 'fwi_predicted' in pred_day.columns else 'fwi'
        
        # Format date for display
        date_str = date.strftime('%Y-%m-%d')
        
        # Column 1: Real Data (25km)
        ax_real = fig.add_subplot(n_dates, 3, i*3 + 1, projection=ccrs.PlateCarree())
        
        if len(real_day) > 0:
            scatter_real = ax_real.scatter(
                real_day['longitude'], real_day['latitude'],
                c=real_day['fwi'], cmap=fwi_cmap, norm=fwi_norm,
                s=60, alpha=0.8, transform=ccrs.PlateCarree(),
                edgecolors='black', linewidth=0.3
            )
            
            title_real = f"ERA5 Real FWI - {date_str}\n(25km, {len(real_day):,} points)"
            fwi_stats_real = f"Range: [{real_day['fwi'].min():.1f}, {real_day['fwi'].max():.1f}]\nMean: {real_day['fwi'].mean():.1f}"
        else:
            title_real = f"ERA5 Real FWI - {date_str}\n(No data available)"
            fwi_stats_real = "No data"
        
        _setup_map(ax_real, lon_min, lon_max, lat_min, lat_max)
        ax_real.set_title(title_real, fontsize=10, fontweight='bold', pad=12)
        ax_real.text(0.02, 0.02, fwi_stats_real, transform=ax_real.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=7, verticalalignment='bottom')
        
        # Column 2: Aggregated Predicted Data (1km to 25km)
        ax_agg = fig.add_subplot(n_dates, 3, i*3 + 2, projection=ccrs.PlateCarree())
        
        if len(pred_day) > 0 and len(real_day) > 0:
            # Aggregate 1km predicted data to 25km grid
            print(f"   Aggregating 1km to 25km for {date}...")
            
            # Define grid based on real data extent
            lat_min_grid, lat_max_grid = real_day['latitude'].min(), real_day['latitude'].max()
            lon_min_grid, lon_max_grid = real_day['longitude'].min(), real_day['longitude'].max()
            
            # Create 25km grid (approximately 0.225° spacing)
            grid_spacing = 0.225
            lat_bins = np.arange(lat_min_grid - grid_spacing/2, lat_max_grid + grid_spacing, grid_spacing)
            lon_bins = np.arange(lon_min_grid - grid_spacing/2, lon_max_grid + grid_spacing, grid_spacing)
            
            # Aggregate data
            pred_aggregated_list = []
            
            for j in range(len(lat_bins)-1):
                for k in range(len(lon_bins)-1):
                    lat_center = (lat_bins[j] + lat_bins[j+1]) / 2
                    lon_center = (lon_bins[k] + lon_bins[k+1]) / 2
                    
                    # Find 1km points within this 25km cell
                    mask = (
                        (pred_day['latitude'] >= lat_bins[j]) & 
                        (pred_day['latitude'] < lat_bins[j+1]) &
                        (pred_day['longitude'] >= lon_bins[k]) & 
                        (pred_day['longitude'] < lon_bins[k+1])
                    )
                    
                    cell_data = pred_day[mask]
                    
                    if len(cell_data) > 0:
                        mean_fwi = cell_data[fwi_col].mean()
                        count = len(cell_data)
                        
                        pred_aggregated_list.append({
                            'latitude': lat_center,
                            'longitude': lon_center,
                            'fwi_mean': mean_fwi,
                            'point_count': count
                        })
            
            if pred_aggregated_list:
                pred_aggregated = pd.DataFrame(pred_aggregated_list)
                
                # Plot aggregated data
                scatter_agg = ax_agg.scatter(
                    pred_aggregated['longitude'], pred_aggregated['latitude'],
                    c=pred_aggregated['fwi_mean'], cmap=fwi_cmap, norm=fwi_norm,
                    s=60, alpha=0.8, transform=ccrs.PlateCarree(),
                    edgecolors='black', linewidth=0.3
                )
                
                title_agg = f"ML Predicted FWI - {date_str}\n(Aggregated to 25km, {len(pred_aggregated):,} cells)"
                avg_points_per_cell = pred_aggregated['point_count'].mean()
                fwi_stats_agg = f"Range: [{pred_aggregated['fwi_mean'].min():.1f}, {pred_aggregated['fwi_mean'].max():.1f}]\nMean: {pred_aggregated['fwi_mean'].mean():.1f}\nAvg points/cell: {avg_points_per_cell:.0f}"
                
            else:
                title_agg = f"ML Predicted FWI - {date_str}\n(Aggregation failed)"
                fwi_stats_agg = "No aggregated data"
        else:
            title_agg = f"ML Predicted FWI - {date_str}\n(No data for aggregation)"
            fwi_stats_agg = "No data"
        
        _setup_map(ax_agg, lon_min, lon_max, lat_min, lat_max)
        ax_agg.set_title(title_agg, fontsize=10, fontweight='bold', pad=12)
        ax_agg.text(0.02, 0.02, fwi_stats_agg, transform=ax_agg.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=7, verticalalignment='bottom')
        
        # Column 3: Predicted Data (1km)
        ax_pred = fig.add_subplot(n_dates, 3, i*3 + 3, projection=ccrs.PlateCarree())
        
        if len(pred_day) > 0:
            # Sample data for visualization
            if len(pred_day) > 15000:
                pred_day_sample = pred_day.sample(15000)
                sample_note = f" (Sample 15k/{len(pred_day):,})"
            else:
                pred_day_sample = pred_day
                sample_note = ""
            
            scatter_pred = ax_pred.scatter(
                pred_day_sample['longitude'], pred_day_sample['latitude'],
                c=pred_day_sample[fwi_col], cmap=fwi_cmap, norm=fwi_norm,
                s=1.5, alpha=0.7, transform=ccrs.PlateCarree()
            )
            
            title_pred = f"ML Predicted FWI - {date_str}\n(1km, {len(pred_day):,} points{sample_note})"
            fwi_stats_pred = f"Range: [{pred_day[fwi_col].min():.1f}, {pred_day[fwi_col].max():.1f}]\nMean: {pred_day[fwi_col].mean():.1f}"
        else:
            title_pred = f"ML Predicted FWI - {date_str}\n(No data available)"
            fwi_stats_pred = "No data"
        
        _setup_map(ax_pred, lon_min, lon_max, lat_min, lat_max)
        ax_pred.set_title(title_pred, fontsize=10, fontweight='bold', pad=12)
        ax_pred.text(0.02, 0.02, fwi_stats_pred, transform=ax_pred.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=7, verticalalignment='bottom')
        
        # Highlight high-risk areas for all three columns
        # Real data high-risk
        if len(real_day) > 0:
            high_risk_real = real_day[real_day['fwi'] > 24]
            if len(high_risk_real) > 0:
                ax_real.scatter(high_risk_real['longitude'], high_risk_real['latitude'],
                              s=120, facecolors='none', edgecolors='red', linewidth=2,
                              transform=ccrs.PlateCarree(), label=f'High Risk ({len(high_risk_real)})')
                ax_real.legend(loc='upper right', fontsize=6)
        
        # Aggregated data high-risk
        if 'pred_aggregated' in locals() and len(pred_aggregated) > 0:
            high_risk_agg = pred_aggregated[pred_aggregated['fwi_mean'] > 24]
            if len(high_risk_agg) > 0:
                ax_agg.scatter(high_risk_agg['longitude'], high_risk_agg['latitude'],
                              s=120, facecolors='none', edgecolors='red', linewidth=2,
                              transform=ccrs.PlateCarree(), label=f'High Risk ({len(high_risk_agg)})')
                ax_agg.legend(loc='upper right', fontsize=6)
        
        # Predicted data high-risk
        if len(pred_day) > 0:
            high_risk_pred = pred_day[pred_day[fwi_col] > 24]
            if len(high_risk_pred) > 0:
                # Sample high-risk points
                if len(high_risk_pred) > 500:
                    high_risk_sample = high_risk_pred.sample(500)
                else:
                    high_risk_sample = high_risk_pred
                
                ax_pred.scatter(high_risk_sample['longitude'], high_risk_sample['latitude'],
                              s=8, facecolors='none', edgecolors='red', linewidth=1,
                              transform=ccrs.PlateCarree(), alpha=0.8, 
                              label=f'High Risk ({len(high_risk_pred):,})')
                ax_pred.legend(loc='upper right', fontsize=6)
    
    # Create colorbar with proper positioning
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    
    if 'scatter_agg' in locals():
        cbar = fig.colorbar(scatter_agg, cax=cbar_ax)
    elif 'scatter_pred' in locals():
        cbar = fig.colorbar(scatter_pred, cax=cbar_ax)
    elif 'scatter_real' in locals():
        cbar = fig.colorbar(scatter_real, cax=cbar_ax)
    else:
        print("   WARNING: No scatter plot found for colorbar")
        return

    cbar.set_label('Fire Weather Index (FWI)', fontsize=12, fontweight='bold')
    
    # Add risk level annotations
    risk_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Extreme', 'Critical']
    for i, (level, label) in enumerate(zip(fwi_levels[:-1], risk_labels)):
        if i < len(fwi_levels) - 1:
            y_pos = (level + fwi_levels[i+1]) / 2
            cbar.ax.text(2.0, y_pos, label, 
                        transform=cbar.ax.transData, 
                        fontsize=8, va='center', ha='left')
    
    plt.suptitle('FWI Spatial Distribution: ERA5 Real vs ML Predicted vs Aggregated\nPortugal Region', 
                fontsize=14, fontweight='bold', y=0.96)
    
    # Adjust layout with more spacing
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.88, 
                       hspace=0.25, wspace=0.15)
    
    filename = "fwi_spatial_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   Comprehensive spatial maps with aggregation saved: {filename}")

def _create_comparison_maps(real_data, pred_data, selected_dates):
    """Create side-by-side comparison maps with difference analysis"""
    
    if 'common' not in selected_dates:
        print("   WARNING: Skipping comparison maps - no common dates")
        return
    
    # Get the target date
    if selected_dates['common']:
        target_date = selected_dates['common'][0]
        
        print(f"   Creating detailed comparison for {target_date}...")
        
        real_day = real_data[real_data['time'].dt.date == target_date]
        pred_day = pred_data[pred_data['time'].dt.date == target_date]
        
        if len(real_day) == 0 or len(pred_day) == 0:
            print("   WARNING: Insufficient data for comparison")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), 
                                subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Portugal extent
        lon_min, lon_max = -10.5, -5.5
        lat_min, lat_max = 35.5, 43.5
        
        # FWI colormap
        fwi_colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF8C00', '#FF4500', '#DC143C', '#8B0000']
        fwi_levels = [0, 5, 12, 24, 38, 50, 75, 100]
        fwi_cmap = colors.ListedColormap(fwi_colors)
        fwi_norm = colors.BoundaryNorm(fwi_levels, fwi_cmap.N)
        
        # Format date for title
        date_str = target_date.strftime('%Y-%m-%d')
        
        # 1. Real data (top-left)
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(real_day['longitude'], real_day['latitude'],
                              c=real_day['fwi'], cmap=fwi_cmap, norm=fwi_norm,
                              s=40, alpha=0.8, edgecolors='black', linewidth=0.3)
        
        _setup_map(ax1, lon_min, lon_max, lat_min, lat_max)
        ax1.set_title(f'ERA5 Real FWI - {date_str}\n(25km, {len(real_day):,} points)', 
                     fontsize=11, fontweight='bold')
        
        # 2. Predicted data (top-right)
        ax2 = axes[0, 1]
        fwi_col = 'fwi_predicted' if 'fwi_predicted' in pred_day.columns else 'fwi'
        
        # Sample if too many points
        if len(pred_day) > 15000:
            pred_sample = pred_day.sample(15000)
            sample_text = f" (Sample 15k/{len(pred_day):,})"
        else:
            pred_sample = pred_day
            sample_text = ""
        
        scatter2 = ax2.scatter(pred_sample['longitude'], pred_sample['latitude'],
                              c=pred_sample[fwi_col], cmap=fwi_cmap, norm=fwi_norm,
                              s=2, alpha=0.7)
        
        _setup_map(ax2, lon_min, lon_max, lat_min, lat_max)
        ax2.set_title(f'ML Predicted FWI - {date_str}\n(1km, {len(pred_day):,} points{sample_text})', 
                     fontsize=11, fontweight='bold')
        
        # 3. High-risk areas overlay (bottom-left)
        ax3 = axes[1, 0]
        
        # Base map with all data
        ax3.scatter(real_day['longitude'], real_day['latitude'],
                   c=real_day['fwi'], cmap=fwi_cmap, norm=fwi_norm,
                   s=30, alpha=0.5, edgecolors='gray', linewidth=0.2)
        
        # Highlight high-risk areas
        high_risk_real = real_day[real_day['fwi'] > 24]
        if len(high_risk_real) > 0:
            ax3.scatter(high_risk_real['longitude'], high_risk_real['latitude'],
                       s=100, facecolors='none', edgecolors='red', linewidth=3,
                       label=f'High Risk: {len(high_risk_real)} points')
        
        extreme_risk_real = real_day[real_day['fwi'] > 50]
        if len(extreme_risk_real) > 0:
            ax3.scatter(extreme_risk_real['longitude'], extreme_risk_real['latitude'],
                       s=150, marker='s', facecolors='none', edgecolors='darkred', linewidth=3,
                       label=f'Extreme Risk: {len(extreme_risk_real)} points')
        
        _setup_map(ax3, lon_min, lon_max, lat_min, lat_max)
        ax3.set_title(f'High-Risk Areas - {date_str}\nFWI > 24 (circles), FWI > 50 (squares)', 
                     fontsize=11, fontweight='bold')
        if len(high_risk_real) > 0 or len(extreme_risk_real) > 0:
            ax3.legend(loc='upper right', fontsize=8)
        
        # 4. Statistics comparison (bottom-right)
        ax4 = axes[1, 1]
        
        # Create histograms
        real_fwi = real_day['fwi']
        pred_fwi = pred_day[fwi_col]
        
        # Remove map projection for histogram
        ax4.remove()
        ax4 = fig.add_subplot(2, 2, 4)
        
        bins = np.linspace(0, max(real_fwi.max(), pred_fwi.max()), 30)
        
        ax4.hist(real_fwi, bins=bins, alpha=0.6, label=f'Real (n={len(real_fwi):,})', 
                color='blue', edgecolor='black')
        ax4.hist(pred_fwi, bins=bins, alpha=0.6, label=f'Predicted (n={len(pred_fwi):,})', 
                color='red', edgecolor='black')
        
        ax4.set_xlabel('FWI Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'FWI Distribution - {date_str}', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics for {date_str}:
Real - Mean: {real_fwi.mean():.1f}, Std: {real_fwi.std():.1f}
Pred - Mean: {pred_fwi.mean():.1f}, Std: {pred_fwi.std():.1f}

High Risk (>24):
Real: {(real_fwi > 24).sum():,} ({(real_fwi > 24).mean()*100:.1f}%)
Pred: {(pred_fwi > 24).sum():,} ({(pred_fwi > 24).mean()*100:.1f}%)

Extreme Risk (>50):
Real: {(real_fwi > 50).sum():,} ({(real_fwi > 50).mean()*100:.1f}%)
Pred: {(pred_fwi > 50).sum():,} ({(pred_fwi > 50).mean()*100:.1f}%)"""
        
        ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                fontsize=8, verticalalignment='top', horizontalalignment='right')
        
        # Create colorbar with proper positioning
        cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
        cbar = fig.colorbar(scatter2, cax=cbar_ax)
        cbar.set_label('Fire Weather Index (FWI)', fontsize=11, fontweight='bold')
        
        plt.suptitle(f'Detailed FWI Comparison - {date_str}\nERA5 Real vs ML Predicted', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = "fwi_detailed_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   Detailed comparison map saved: {filename}")

def _create_high_resolution_detail_maps(pred_data, selected_dates):
    """Create high-resolution detail maps focusing on specific regions"""
    
    dates_to_plot = []
    if 'common' in selected_dates:
        dates_to_plot = selected_dates['common'][:1]  # Use target date
    elif 'pred' in selected_dates:
        dates_to_plot = selected_dates['pred'][:1]
    
    if not dates_to_plot:
        print("   WARNING: No dates available for high-resolution maps")
        return
    
    target_date = dates_to_plot[0]
    print(f"   Creating high-resolution detail map for {target_date}...")
    
    pred_day = pred_data[pred_data['time'].dt.date == target_date]
    if len(pred_day) == 0:
        print(f"   WARNING: No predicted data for {target_date}")
        return
    
    # Create figure with regional zooms
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    fwi_col = 'fwi_predicted' if 'fwi_predicted' in pred_day.columns else 'fwi'
    
    # FWI colormap
    fwi_colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF8C00', '#FF4500', '#DC143C', '#8B0000']
    fwi_levels = [0, 5, 12, 24, 38, 50, 75, 100]
    fwi_cmap = colors.ListedColormap(fwi_colors)
    fwi_norm = colors.BoundaryNorm(fwi_levels, fwi_cmap.N)
    
    # Format date for display
    date_str = target_date.strftime('%Y-%m-%d')
    
    # Define regional extents
    regions = [
        {"name": "Full Portugal", "extent": [-10.5, -5.5, 35.5, 43.5]},
        {"name": "Northern Portugal", "extent": [-9.5, -6.0, 40.5, 42.5]},
        {"name": "Central Portugal", "extent": [-9.5, -6.5, 38.5, 40.5]},
        {"name": "Southern Portugal", "extent": [-9.0, -6.5, 36.0, 38.5]}
    ]
    
    for i, (ax, region) in enumerate(zip(axes.flat, regions)):
        lon_min, lon_max, lat_min, lat_max = region["extent"]
        
        # Filter data for this region
        region_data = pred_day[
            (pred_day['longitude'] >= lon_min) & (pred_day['longitude'] <= lon_max) &
            (pred_day['latitude'] >= lat_min) & (pred_day['latitude'] <= lat_max)
        ]
        
        if len(region_data) > 0:
            # Adjust point size based on zoom level
            point_size = 3 if region["name"] == "Full Portugal" else 8
            
            scatter = ax.scatter(region_data['longitude'], region_data['latitude'],
                               c=region_data[fwi_col], cmap=fwi_cmap, norm=fwi_norm,
                               s=point_size, alpha=0.8)
            
            # Highlight extreme values
            extreme_data = region_data[region_data[fwi_col] > 50]
            if len(extreme_data) > 0:
                ax.scatter(extreme_data['longitude'], extreme_data['latitude'],
                         s=point_size*3, facecolors='none', edgecolors='darkred', 
                         linewidth=2, alpha=0.9)
            
            stats_text = f"Points: {len(region_data):,}\nMean FWI: {region_data[fwi_col].mean():.1f}\nMax FWI: {region_data[fwi_col].max():.1f}"
            if len(extreme_data) > 0:
                stats_text += f"\nExtreme: {len(extreme_data)}"
        else:
            stats_text = "No data in region"
        
        # Setup map
        _setup_map(ax, lon_min, lon_max, lat_min, lat_max)
        ax.set_title(f'{region["name"]} - {date_str}', fontsize=10, fontweight='bold')
        
        # Add statistics box
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
               fontsize=8, verticalalignment='top')
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.6, pad=0.02, aspect=20)
    cbar.set_label('Fire Weather Index (FWI)', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'High-Resolution FWI Detail Maps - {date_str}\nML Predicted (1km Resolution)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filename = "fwi_high_resolution_detail.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   High-resolution detail map saved: {filename}")

def _create_aggregated_comparison_maps(real_data, pred_data, selected_dates):
    """Create comparison maps with 1km data aggregated to 25km resolution"""
    
    print("   Creating 1km to 25km aggregated comparison maps...")
    
    if 'common' not in selected_dates or not selected_dates['common']:
        print("   WARNING: No common dates available for aggregated comparison")
        return
    
    # Use the target date for aggregation analysis
    target_date = selected_dates['common'][0]
    
    real_day = real_data[real_data['time'].dt.date == target_date]
    pred_day = pred_data[pred_data['time'].dt.date == target_date]
    
    if len(real_day) == 0 or len(pred_day) == 0:
        print("   WARNING: Insufficient data for aggregated comparison")
        return
    
    # Format date for display
    date_str = target_date.strftime('%Y-%m-%d')
    
    # Get FWI column name for predicted data
    fwi_col = 'fwi_predicted' if 'fwi_predicted' in pred_day.columns else 'fwi'
    
    # Aggregate 1km predicted data to 25km grid
    print("   Aggregating 1km predictions to 25km grid...")
    
    # Define 25km grid based on real data extent
    lat_min, lat_max = real_day['latitude'].min(), real_day['latitude'].max()
    lon_min, lon_max = real_day['longitude'].min(), real_day['longitude'].max()
    
    # Create 25km grid (approximately 0.225° spacing for 25km)
    grid_spacing = 0.225
    lat_bins = np.arange(lat_min - grid_spacing/2, lat_max + grid_spacing, grid_spacing)
    lon_bins = np.arange(lon_min - grid_spacing/2, lon_max + grid_spacing, grid_spacing)
    
    # Aggregate predicted data to 25km grid
    pred_aggregated_list = []
    
    for i in range(len(lat_bins)-1):
        for j in range(len(lon_bins)-1):
            lat_center = (lat_bins[i] + lat_bins[i+1]) / 2
            lon_center = (lon_bins[j] + lon_bins[j+1]) / 2
            
            # Find 1km points within this 25km cell
            mask = (
                (pred_day['latitude'] >= lat_bins[i]) & 
                (pred_day['latitude'] < lat_bins[i+1]) &
                (pred_day['longitude'] >= lon_bins[j]) & 
                (pred_day['longitude'] < lon_bins[j+1])
            )
            
            cell_data = pred_day[mask]
            
            if len(cell_data) > 0:
                # Calculate statistics for this cell
                mean_fwi = cell_data[fwi_col].mean()
                max_fwi = cell_data[fwi_col].max()
                min_fwi = cell_data[fwi_col].min()
                std_fwi = cell_data[fwi_col].std()
                count = len(cell_data)
                
                pred_aggregated_list.append({
                    'latitude': lat_center,
                    'longitude': lon_center,
                    'fwi_mean': mean_fwi,
                    'fwi_max': max_fwi,
                    'fwi_min': min_fwi,
                    'fwi_std': std_fwi,
                    'point_count': count
                })
    
    if not pred_aggregated_list:
        print("   WARNING: No aggregated data created")
        return
    
    pred_aggregated = pd.DataFrame(pred_aggregated_list)
    print(f"   Aggregated {len(pred_day):,} 1km points to {len(pred_aggregated):,} 25km cells")
    
    # Create figure with multiple comparison views
    fig = plt.figure(figsize=(22, 16))
    
    # Portugal extent
    extent_lon_min, extent_lon_max = -10.5, -5.5
    extent_lat_min, extent_lat_max = 35.5, 43.5
    
    # FWI colormap
    fwi_colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF8C00', '#FF4500', '#DC143C', '#8B0000']
    fwi_levels = [0, 5, 12, 24, 38, 50, 75, 100]
    fwi_cmap = colors.ListedColormap(fwi_colors)
    fwi_norm = colors.BoundaryNorm(fwi_levels, fwi_cmap.N)
    
    # 1. Original ERA5 Real Data (25km) - Top Left
    ax1 = fig.add_subplot(2, 3, 1, projection=ccrs.PlateCarree())
    scatter1 = ax1.scatter(real_day['longitude'], real_day['latitude'],
                          c=real_day['fwi'], cmap=fwi_cmap, norm=fwi_norm,
                          s=80, alpha=0.8, edgecolors='black', linewidth=0.5,
                          transform=ccrs.PlateCarree())
    
    _setup_map(ax1, extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max)
    ax1.set_title(f'ERA5 Real FWI (25km)\n{len(real_day):,} points', 
                 fontsize=12, fontweight='bold')
    
    # Add statistics
    real_stats = f"Mean: {real_day['fwi'].mean():.1f}\nMax: {real_day['fwi'].max():.1f}\nStd: {real_day['fwi'].std():.1f}"
    ax1.text(0.02, 0.98, real_stats, transform=ax1.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=9, verticalalignment='top')
    
    # 2. Aggregated Predicted Data (25km from 1km) - Top Middle
    ax2 = fig.add_subplot(2, 3, 2, projection=ccrs.PlateCarree())
    scatter2 = ax2.scatter(pred_aggregated['longitude'], pred_aggregated['latitude'],
                          c=pred_aggregated['fwi_mean'], cmap=fwi_cmap, norm=fwi_norm,
                          s=80, alpha=0.8, edgecolors='black', linewidth=0.5,
                          transform=ccrs.PlateCarree())
    
    _setup_map(ax2, extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max)
    ax2.set_title(f'ML Predicted FWI (Aggregated to 25km)\n{len(pred_aggregated):,} cells', 
                 fontsize=12, fontweight='bold')
    
    # Add statistics
    agg_stats = f"Mean: {pred_aggregated['fwi_mean'].mean():.1f}\nMax: {pred_aggregated['fwi_mean'].max():.1f}\nStd: {pred_aggregated['fwi_mean'].std():.1f}"
    ax2.text(0.02, 0.98, agg_stats, transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=9, verticalalignment='top')
    
    # 3. Difference Map (Real - Aggregated Predicted) - Top Right
    ax3 = fig.add_subplot(2, 3, 3, projection=ccrs.PlateCarree())
    
    # Match real data points to aggregated predicted data
    differences = []
    matched_lons = []
    matched_lats = []
    
    for _, real_point in real_day.iterrows():
        # Find nearest aggregated cell
        distances = np.sqrt((pred_aggregated['latitude'] - real_point['latitude'])**2 + 
                          (pred_aggregated['longitude'] - real_point['longitude'])**2)
        nearest_idx = distances.idxmin()
        
        if distances.iloc[nearest_idx] < 0.5:  # Within reasonable distance
            diff = real_point['fwi'] - pred_aggregated.loc[nearest_idx, 'fwi_mean']
            differences.append(diff)
            matched_lons.append(real_point['longitude'])
            matched_lats.append(real_point['latitude'])
    
    if differences:
        diff_cmap = plt.cm.RdBu_r
        diff_norm = colors.Normalize(vmin=-20, vmax=20)
        
        scatter3 = ax3.scatter(matched_lons, matched_lats,
                              c=differences, cmap=diff_cmap, norm=diff_norm,
                              s=80, alpha=0.8, edgecolors='black', linewidth=0.5,
                              transform=ccrs.PlateCarree())
        
        # Add difference colorbar - positioned independently
        diff_cbar_ax = fig.add_axes([0.655, 0.68, 0.012, 0.25])
        cbar_diff = fig.colorbar(scatter3, cax=diff_cbar_ax)
        cbar_diff.set_label('FWI Difference\n(Real - Predicted)', fontsize=9)
        
        diff_stats = f"Mean Diff: {np.mean(differences):.2f}\nMAE: {np.mean(np.abs(differences)):.2f}\nRMSE: {np.sqrt(np.mean(np.square(differences))):.2f}"
    else:
        diff_stats = "No matching points"
    
    _setup_map(ax3, extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max)
    ax3.set_title(f'Difference Map (Real - Predicted)\n{len(differences)} matched points', 
                 fontsize=12, fontweight='bold')
    ax3.text(0.02, 0.98, diff_stats, transform=ax3.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=9, verticalalignment='top')
    
    # 4. Original 1km Predicted Data Sample - Bottom Left
    ax4 = fig.add_subplot(2, 3, 4, projection=ccrs.PlateCarree())
    
    # Sample 1km data for visualization
    if len(pred_day) > 10000:
        pred_sample = pred_day.sample(10000)
        sample_text = f" (Sample 10k/{len(pred_day):,})"
    else:
        pred_sample = pred_day
        sample_text = ""
    
    scatter4 = ax4.scatter(pred_sample['longitude'], pred_sample['latitude'],
                          c=pred_sample[fwi_col], cmap=fwi_cmap, norm=fwi_norm,
                          s=2, alpha=0.6, transform=ccrs.PlateCarree())
    
    _setup_map(ax4, extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max)
    ax4.set_title(f'ML Predicted FWI (Original 1km){sample_text}', 
                 fontsize=12, fontweight='bold')
    
    pred_stats = f"Mean: {pred_day[fwi_col].mean():.1f}\nMax: {pred_day[fwi_col].max():.1f}\nStd: {pred_day[fwi_col].std():.1f}"
    ax4.text(0.02, 0.98, pred_stats, transform=ax4.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=9, verticalalignment='top')
    
    # 5. Statistical Comparison - Bottom Middle
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Create comparison histogram
    bins = np.linspace(0, max(real_day['fwi'].max(), pred_aggregated['fwi_mean'].max()), 25)
    
    ax5.hist(real_day['fwi'], bins=bins, alpha=0.6, label=f'Real 25km (n={len(real_day):,})', 
            color='blue', edgecolor='black', density=True)
    ax5.hist(pred_aggregated['fwi_mean'], bins=bins, alpha=0.6, 
            label=f'Aggregated Predicted (n={len(pred_aggregated):,})', 
            color='red', edgecolor='black', density=True)
    
    ax5.set_xlabel('FWI Value')
    ax5.set_ylabel('Density')
    ax5.set_title('FWI Distribution Comparison\n(25km Resolution)', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add correlation if we have matching points
    if differences:
        # Calculate correlation between matched points
        real_matched = []
        pred_matched = []
        for _, real_point in real_day.iterrows():
            distances = np.sqrt((pred_aggregated['latitude'] - real_point['latitude'])**2 + 
                              (pred_aggregated['longitude'] - real_point['longitude'])**2)
            nearest_idx = distances.idxmin()
            if distances.iloc[nearest_idx] < 0.5:
                real_matched.append(real_point['fwi'])
                pred_matched.append(pred_aggregated.loc[nearest_idx, 'fwi_mean'])
        
        if len(real_matched) > 5:
            correlation = np.corrcoef(real_matched, pred_matched)[0, 1]
            ax5.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax5.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # 6. Aggregation Uncertainty - Bottom Right
    ax6 = fig.add_subplot(2, 3, 6, projection=ccrs.PlateCarree())
    
    # Show standard deviation within each 25km cell
    uncertainty_cmap = plt.cm.viridis
    uncertainty_norm = colors.Normalize(vmin=0, vmax=pred_aggregated['fwi_std'].quantile(0.95))
    
    scatter6 = ax6.scatter(pred_aggregated['longitude'], pred_aggregated['latitude'],
                          c=pred_aggregated['fwi_std'], cmap=uncertainty_cmap, norm=uncertainty_norm,
                          s=pred_aggregated['point_count']/50, alpha=0.8, 
                          edgecolors='black', linewidth=0.5,
                          transform=ccrs.PlateCarree())
    
    _setup_map(ax6, extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max)
    ax6.set_title('Aggregation Uncertainty\n(Std Dev within 25km cells)', 
                 fontsize=12, fontweight='bold')
    
    # Add uncertainty colorbar - positioned independently
    uncertainty_cbar_ax = fig.add_axes([0.95, 0.15, 0.012, 0.25])
    cbar_uncertainty = fig.colorbar(scatter6, cax=uncertainty_cbar_ax)
    cbar_uncertainty.set_label('FWI Standard\nDeviation', fontsize=9)
    
    uncertainty_stats = f"Mean Std: {pred_aggregated['fwi_std'].mean():.2f}\nMax Std: {pred_aggregated['fwi_std'].max():.2f}\nPoints/Cell: {pred_aggregated['point_count'].mean():.0f}"
    ax6.text(0.02, 0.98, uncertainty_stats, transform=ax6.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=9, verticalalignment='top')
    
    # Move FWI colorbar and risk labels next to the first subplot
    main_cbar_ax = fig.add_axes([0.33, 0.53, 0.012, 0.4])
    cbar_main = fig.colorbar(scatter2, cax=main_cbar_ax)
    cbar_main.set_label('Fire Weather Index (FWI)', fontsize=10, fontweight='bold')
    
    # Add risk level annotations next to the FWI colorbar
    risk_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Extreme', 'Critical']
    for i, (level, label) in enumerate(zip(fwi_levels[:-1], risk_labels)):
        if i < len(fwi_levels) - 1:
            y_pos = (level + fwi_levels[i+1]) / 2
            # Position labels to the left of the colorbar
            cbar_main.ax.text(-0.5, y_pos, label, 
                             transform=cbar_main.ax.transData, 
                             fontsize=7, va='center', ha='right',
                             bbox=dict(boxstyle="round,pad=0.1", 
                                     facecolor="white", alpha=0.9, 
                                     edgecolor='lightgray', linewidth=0.3))
    
    plt.suptitle(f'FWI Resolution Comparison - {date_str}\n25km Real vs 1km to 25km Aggregated Predictions', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout with more space for the relocated colorbar
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.88, 
                       hspace=0.25, wspace=0.2)
    
    plt.tight_layout()
    filename = "fwi_aggregated_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   Aggregated comparison map saved: {filename}")
    
    # Create summary statistics
    print("   Aggregation Analysis Summary:")
    print(f"      Original 1km predictions: {len(pred_day):,} points")
    print(f"      Aggregated to 25km cells: {len(pred_aggregated):,} cells")
    print(f"      Average points per cell: {pred_aggregated['point_count'].mean():.1f}")
    print(f"      Real data mean FWI: {real_day['fwi'].mean():.2f}")
    print(f"      Aggregated predicted mean FWI: {pred_aggregated['fwi_mean'].mean():.2f}")
    if differences:
        print(f"      Mean absolute error: {np.mean(np.abs(differences)):.2f}")
        print(f"      Root mean square error: {np.sqrt(np.mean(np.square(differences))):.2f}")
        print(f"      Correlation coefficient: {np.corrcoef(real_matched, pred_matched)[0, 1]:.3f}")

def _setup_map(ax, lon_min, lon_max, lat_min, lat_max):
    """Setup map features and styling"""
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, color='navy')
    ax.add_feature(cfeature.BORDERS, linewidth=1, color='gray')
    ax.add_feature(cfeature.LAND, alpha=0.2, color='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
    ax.add_feature(cfeature.LAKES, alpha=0.3, color='lightblue')
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, color='blue', alpha=0.6)
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

# Main execution function
def main():
    """Main function to create all spatial maps"""
    print("FWI SPATIAL MAPPING SYSTEM")
    print("="*50)
    
    try:
        create_fwi_spatial_maps()
        print("\nAll spatial maps generated successfully!")
        print("Generated files:")
        print("   fwi_spatial_comparison.png - Comprehensive comparison")
        print("   fwi_detailed_comparison.png - Detailed analysis") 
        print("   fwi_high_resolution_detail.png - Regional details")
        print("   fwi_aggregated_comparison.png - 1km to 25km aggregation analysis")
        
    except Exception as e:
        print(f"ERROR: Error in spatial mapping: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()