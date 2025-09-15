import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def _setup_map(ax, lon_min, lon_max, lat_min, lat_max):
    """Setup map with Portugal boundaries and features"""
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, color='gray')
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.1)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False

def aggregate_to_25km_grid(pred_data, real_data):
    """Aggregate 1km predicted data to 25km grid matching real data"""
    
    print("   Aggregating 1km predictions to 25km grid...")
    
    # Get unique real data locations (25km grid points)
    real_locations = real_data[['latitude', 'longitude']].drop_duplicates()
    
    # Create aggregated data storage
    aggregated_results = []
    
    # Grid spacing (approximately 0.225° for 25km)
    grid_spacing = 0.225
    
    for _, real_point in real_locations.iterrows():
        real_lat = real_point['latitude']
        real_lon = real_point['longitude']
        
        # Define grid cell boundaries
        lat_min = real_lat - grid_spacing/2
        lat_max = real_lat + grid_spacing/2
        lon_min = real_lon - grid_spacing/2
        lon_max = real_lon + grid_spacing/2
        
        # Find all predicted points within this grid cell
        mask = (
            (pred_data['latitude'] >= lat_min) & 
            (pred_data['latitude'] < lat_max) &
            (pred_data['longitude'] >= lon_min) & 
            (pred_data['longitude'] < lon_max)
        )
        
        cell_pred_data = pred_data[mask]
        
        if len(cell_pred_data) > 0:
            # Group by date and calculate daily mean for this cell
            fwi_col = 'fwi_predicted' if 'fwi_predicted' in cell_pred_data.columns else 'fwi'
            daily_means = cell_pred_data.groupby(cell_pred_data['time'].dt.date)[fwi_col].mean()
            
            # Store aggregated data with location
            for date, fwi_mean in daily_means.items():
                aggregated_results.append({
                    'latitude': real_lat,
                    'longitude': real_lon,
                    'time': pd.Timestamp(date),
                    'fwi_aggregated': fwi_mean
                })
    
    aggregated_df = pd.DataFrame(aggregated_results)
    print(f"   Aggregated to {len(aggregated_df):,} records at {len(real_locations):,} grid points")
    
    return aggregated_df

def calculate_grid_point_metrics(real_data, aggregated_data):
    """Calculate RMSE, correlation, R², and other metrics for each grid point"""
    
    print("   Calculating metrics for each grid point...")
    
    # Get unique grid points
    grid_points = real_data[['latitude', 'longitude']].drop_duplicates()
    
    metrics_results = []
    
    for _, point in grid_points.iterrows():
        lat, lon = point['latitude'], point['longitude']
        
        # Get real data for this grid point
        real_point_data = real_data[
            (real_data['latitude'] == lat) & 
            (real_data['longitude'] == lon)
        ].copy()
        
        # Get aggregated predicted data for this grid point
        agg_point_data = aggregated_data[
            (aggregated_data['latitude'] == lat) & 
            (aggregated_data['longitude'] == lon)
        ].copy()
        
        if len(real_point_data) > 0 and len(agg_point_data) > 0:
            # Merge on time for comparison
            real_point_data['date'] = real_point_data['time'].dt.date
            agg_point_data['date'] = agg_point_data['time'].dt.date
            
            merged = pd.merge(real_point_data[['date', 'fwi']], 
                            agg_point_data[['date', 'fwi_aggregated']], 
                            on='date', how='inner')
            
            if len(merged) >= 10:  # Minimum 10 data points for reliable statistics
                real_values = merged['fwi'].values
                pred_values = merged['fwi_aggregated'].values
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((real_values - pred_values)**2))
                
                # Calculate correlation
                if np.std(real_values) > 0 and np.std(pred_values) > 0:
                    correlation = np.corrcoef(real_values, pred_values)[0, 1]
                    
                    # Calculate R² (coefficient of determination)
                    r_squared = correlation ** 2
                else:
                    correlation = np.nan
                    r_squared = np.nan
                
                # Calculate other metrics
                mae = np.mean(np.abs(real_values - pred_values))
                bias = np.mean(pred_values - real_values)
                real_mean = np.mean(real_values)
                pred_mean = np.mean(pred_values)
                
                # Alternative R² calculation using explained variance
                ss_res = np.sum((real_values - pred_values) ** 2)  # Sum of squares of residuals
                ss_tot = np.sum((real_values - np.mean(real_values)) ** 2)  # Total sum of squares
                r_squared_alt = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                
                metrics_results.append({
                    'latitude': lat,
                    'longitude': lon,
                    'rmse': rmse,
                    'correlation': correlation,
                    'r_squared': r_squared,
                    'r_squared_alt': r_squared_alt,  # Alternative R² calculation
                    'mae': mae,
                    'bias': bias,
                    'real_mean': real_mean,
                    'pred_mean': pred_mean,
                    'n_points': len(merged)
                })
    
    metrics_df = pd.DataFrame(metrics_results)
    print(f"   Calculated metrics for {len(metrics_df):,} grid points")
    
    return metrics_df

def create_annual_performance_maps():
    """Create annual RMSE and correlation maps comparing real vs aggregated predicted data"""
    
    print("Creating Annual Performance Maps...")
    
    # Load data
    try:
        # Load ERA5 real data (25km)
        print("   Loading ERA5 real data...")
        real_data = pd.read_csv("experiment/ERA5_reanalysis_fwi/era5_fwi_2017_portugal_3decimal.csv")
        real_data['time'] = pd.to_datetime(real_data['time'])
        print(f"   Real data: {real_data.shape[0]:,} records")
        
        # Load 1km predicted data
        print("   Loading 1km predicted data...")
        pred_data = pd.read_csv("fwi_1km_predictions_random_forest.csv")
        if 'date' in pred_data.columns:
            pred_data['time'] = pd.to_datetime(pred_data['date'])
        elif 'time' in pred_data.columns:
            pred_data['time'] = pd.to_datetime(pred_data['time'])
        print(f"   Predicted data: {pred_data.shape[0]:,} records")
        
    except Exception as e:
        print(f"ERROR: Error loading data: {e}")
        return
    
    # Filter for 2017 only
    real_data = real_data[real_data['time'].dt.year == 2017]
    pred_data = pred_data[pred_data['time'].dt.year == 2017]
    
    print(f"   2017 data - Real: {len(real_data):,}, Predicted: {len(pred_data):,}")
    
    # Aggregate 1km predictions to 25km grid
    aggregated_data = aggregate_to_25km_grid(pred_data, real_data)
    
    # Calculate metrics for each grid point
    metrics_df = calculate_grid_point_metrics(real_data, aggregated_data)
    
    if len(metrics_df) == 0:
        print("ERROR: No metrics calculated - insufficient data overlap")
        return
    
    # Create performance maps
    create_performance_maps(metrics_df)
    
def create_performance_maps(metrics_df):
    """Create RMSE, correlation, R², and other performance maps"""
    
    print("   Creating performance visualization maps...")
    
    # Create figure with 2x3 subplots to include R²
    fig = plt.figure(figsize=(24, 16))
    
    # Portugal extent
    lon_min, lon_max = -10.5, -5.5
    lat_min, lat_max = 35.5, 43.5
    
    # Remove NaN values for visualization
    metrics_clean = metrics_df.dropna()
    
    print(f"   Visualizing metrics for {len(metrics_clean):,} grid points")
    print(f"   RMSE range: [{metrics_clean['rmse'].min():.2f}, {metrics_clean['rmse'].max():.2f}]")
    print(f"   Correlation range: [{metrics_clean['correlation'].min():.3f}, {metrics_clean['correlation'].max():.3f}]")
    print(f"   R² range: [{metrics_clean['r_squared'].min():.3f}, {metrics_clean['r_squared'].max():.3f}]")
    
    # 1. RMSE Map - Top Left
    ax1 = fig.add_subplot(2, 3, 1, projection=ccrs.PlateCarree())
    
    # RMSE colormap (lower is better)
    rmse_cmap = plt.cm.Reds
    rmse_norm = colors.Normalize(vmin=0, vmax=np.percentile(metrics_clean['rmse'], 95))
    
    scatter1 = ax1.scatter(metrics_clean['longitude'], metrics_clean['latitude'],
                          c=metrics_clean['rmse'], cmap=rmse_cmap, norm=rmse_norm,
                          s=100, alpha=0.8, edgecolors='black', linewidth=0.5,
                          transform=ccrs.PlateCarree())
    
    _setup_map(ax1, lon_min, lon_max, lat_min, lat_max)
    ax1.set_title('Annual RMSE (2017)\nReal vs Aggregated Predicted FWI', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add RMSE colorbar
    cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.8, pad=0.02)
    cbar1.set_label('RMSE (FWI units)', fontsize=10)
    
    # Add RMSE statistics
    rmse_stats = f"Mean: {metrics_clean['rmse'].mean():.2f}\nMedian: {metrics_clean['rmse'].median():.2f}\nStd: {metrics_clean['rmse'].std():.2f}"
    ax1.text(0.02, 0.98, rmse_stats, transform=ax1.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=9, verticalalignment='top')
    
    # 2. Correlation Map - Top Center
    ax2 = fig.add_subplot(2, 3, 2, projection=ccrs.PlateCarree())
    
    # Correlation colormap (higher is better)
    corr_cmap = plt.cm.RdYlBu_r
    corr_norm = colors.Normalize(vmin=-0.5, vmax=1.0)
    
    scatter2 = ax2.scatter(metrics_clean['longitude'], metrics_clean['latitude'],
                          c=metrics_clean['correlation'], cmap=corr_cmap, norm=corr_norm,
                          s=100, alpha=0.8, edgecolors='black', linewidth=0.5,
                          transform=ccrs.PlateCarree())
    
    _setup_map(ax2, lon_min, lon_max, lat_min, lat_max)
    ax2.set_title('Annual Correlation (2017)\nReal vs Aggregated Predicted FWI', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add correlation colorbar
    cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.set_label('Correlation Coefficient (R)', fontsize=10)
    
    # Add correlation statistics
    corr_stats = f"Mean: {metrics_clean['correlation'].mean():.3f}\nMedian: {metrics_clean['correlation'].median():.3f}\nStd: {metrics_clean['correlation'].std():.3f}"
    ax2.text(0.02, 0.98, corr_stats, transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=9, verticalalignment='top')
    
    # 3. R² Map - Top Right
    ax3 = fig.add_subplot(2, 3, 3, projection=ccrs.PlateCarree())
    
    # R² colormap (higher is better, similar to correlation but 0-1 range)
    r2_cmap = plt.cm.viridis
    r2_norm = colors.Normalize(vmin=0, vmax=1.0)
    
    scatter3 = ax3.scatter(metrics_clean['longitude'], metrics_clean['latitude'],
                          c=metrics_clean['r_squared'], cmap=r2_cmap, norm=r2_norm,
                          s=100, alpha=0.8, edgecolors='black', linewidth=0.5,
                          transform=ccrs.PlateCarree())
    
    _setup_map(ax3, lon_min, lon_max, lat_min, lat_max)
    ax3.set_title('Annual R² (2017)\nCoefficient of Determination', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add R² colorbar
    cbar3 = fig.colorbar(scatter3, ax=ax3, shrink=0.8, pad=0.02)
    cbar3.set_label('R² (Coefficient of Determination)', fontsize=10)
    
    # Add R² statistics
    r2_stats = f"Mean: {metrics_clean['r_squared'].mean():.3f}\nMedian: {metrics_clean['r_squared'].median():.3f}\nStd: {metrics_clean['r_squared'].std():.3f}"
    ax3.text(0.02, 0.98, r2_stats, transform=ax3.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=9, verticalalignment='top')
    
    # 4. MAE Map - Bottom Left
    ax4 = fig.add_subplot(2, 3, 4, projection=ccrs.PlateCarree())
    
    # MAE colormap
    mae_cmap = plt.cm.Oranges
    mae_norm = colors.Normalize(vmin=0, vmax=np.percentile(metrics_clean['mae'], 95))
    
    scatter4 = ax4.scatter(metrics_clean['longitude'], metrics_clean['latitude'],
                          c=metrics_clean['mae'], cmap=mae_cmap, norm=mae_norm,
                          s=100, alpha=0.8, edgecolors='black', linewidth=0.5,
                          transform=ccrs.PlateCarree())
    
    _setup_map(ax4, lon_min, lon_max, lat_min, lat_max)
    ax4.set_title('Annual MAE (2017)\nMean Absolute Error', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add MAE colorbar
    cbar4 = fig.colorbar(scatter4, ax=ax4, shrink=0.8, pad=0.02)
    cbar4.set_label('MAE (FWI units)', fontsize=10)
    
    # Add MAE statistics
    mae_stats = f"Mean: {metrics_clean['mae'].mean():.2f}\nMedian: {metrics_clean['mae'].median():.2f}\nStd: {metrics_clean['mae'].std():.2f}"
    ax4.text(0.02, 0.98, mae_stats, transform=ax4.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=9, verticalalignment='top')
    
    # 5. Bias Map - Bottom Center
    ax5 = fig.add_subplot(2, 3, 5, projection=ccrs.PlateCarree())
    
    # Bias colormap (centered at 0)
    bias_cmap = plt.cm.RdBu_r
    bias_max = np.max(np.abs(metrics_clean['bias']))
    bias_norm = colors.Normalize(vmin=-bias_max, vmax=bias_max)
    
    scatter5 = ax5.scatter(metrics_clean['longitude'], metrics_clean['latitude'],
                          c=metrics_clean['bias'], cmap=bias_cmap, norm=bias_norm,
                          s=100, alpha=0.8, edgecolors='black', linewidth=0.5,
                          transform=ccrs.PlateCarree())
    
    _setup_map(ax5, lon_min, lon_max, lat_min, lat_max)
    ax5.set_title('Annual Bias (2017)\nPredicted - Real FWI', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add bias colorbar
    cbar5 = fig.colorbar(scatter5, ax=ax5, shrink=0.8, pad=0.02)
    cbar5.set_label('Bias (FWI units)', fontsize=10)
    
    # Add bias statistics
    bias_stats = f"Mean: {metrics_clean['bias'].mean():.2f}\nMedian: {metrics_clean['bias'].median():.2f}\nStd: {metrics_clean['bias'].std():.2f}"
    ax5.text(0.02, 0.98, bias_stats, transform=ax5.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=9, verticalalignment='top')
    
    # 6. Performance Summary - Bottom Right
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Create scatter plot: R² vs RMSE
    scatter6 = ax6.scatter(metrics_clean['r_squared'], metrics_clean['rmse'],
                          c=metrics_clean['correlation'], cmap=plt.cm.viridis,
                          s=60, alpha=0.7, edgecolors='black', linewidth=0.3)
    
    ax6.set_xlabel('R² (Coefficient of Determination)', fontsize=11)
    ax6.set_ylabel('RMSE (FWI units)', fontsize=11)
    ax6.set_title('Performance Summary\nR² vs RMSE (colored by Correlation)', 
                 fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Add performance summary colorbar
    cbar6 = fig.colorbar(scatter6, ax=ax6, shrink=0.8, pad=0.02)
    cbar6.set_label('Correlation (R)', fontsize=10)
    
    # Add quadrant lines
    ax6.axhline(y=metrics_clean['rmse'].median(), color='red', linestyle='--', alpha=0.5, label=f'Median RMSE: {metrics_clean["rmse"].median():.2f}')
    ax6.axvline(x=metrics_clean['r_squared'].median(), color='blue', linestyle='--', alpha=0.5, label=f'Median R²: {metrics_clean["r_squared"].median():.3f}')
    ax6.legend(fontsize=9)
    
    # Add overall title
    plt.suptitle('Annual FWI Performance Metrics - 2017\nERA5 Real vs ML Predicted (Aggregated to 25km)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    filename = "fwi_annual_performance_metrics.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   Annual performance maps saved: {filename}")
    
    # Create summary statistics table
    create_performance_summary(metrics_clean)

def create_performance_summary(metrics_df):
    """Create and save performance summary statistics including R²"""
    
    print("   Creating performance summary...")
    
    # Include R² in summary statistics
    stats_summary = {
        'Metric': ['RMSE', 'Correlation', 'R²', 'MAE', 'Bias'],
        'Mean': [
            metrics_df['rmse'].mean(),
            metrics_df['correlation'].mean(),
            metrics_df['r_squared'].mean(),
            metrics_df['mae'].mean(),
            metrics_df['bias'].mean()
        ],
        'Median': [
            metrics_df['rmse'].median(),
            metrics_df['correlation'].median(),
            metrics_df['r_squared'].median(),
            metrics_df['mae'].median(),
            metrics_df['bias'].median()
        ],
        'Std': [
            metrics_df['rmse'].std(),
            metrics_df['correlation'].std(),
            metrics_df['r_squared'].std(),
            metrics_df['mae'].std(),
            metrics_df['bias'].std()
        ],
        'Min': [
            metrics_df['rmse'].min(),
            metrics_df['correlation'].min(),
            metrics_df['r_squared'].min(),
            metrics_df['mae'].min(),
            metrics_df['bias'].min()
        ],
        'Max': [
            metrics_df['rmse'].max(),
            metrics_df['correlation'].max(),
            metrics_df['r_squared'].max(),
            metrics_df['mae'].max(),
            metrics_df['bias'].max()
        ]
    }
    
    summary_df = pd.DataFrame(stats_summary)
    
    # Save to CSV
    summary_df.to_csv('fwi_annual_performance_summary.csv', index=False)
    print(f"   Performance summary saved: fwi_annual_performance_summary.csv")
    
    # Print summary
    print("\nANNUAL PERFORMANCE SUMMARY (2017):")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        if row['Metric'] in ['Correlation', 'R²']:
            print(f"{row['Metric']:12}: {row['Mean']:6.3f} ± {row['Std']:6.3f} (median: {row['Median']:6.3f})")
        else:
            print(f"{row['Metric']:12}: {row['Mean']:6.2f} ± {row['Std']:6.2f} (median: {row['Median']:6.2f})")
    
    print(f"\nTotal grid points analyzed: {len(metrics_df):,}")
    
    # Performance categories for both correlation and R²
    excellent_corr = (metrics_df['correlation'] > 0.8).sum()
    good_corr = ((metrics_df['correlation'] > 0.6) & (metrics_df['correlation'] <= 0.8)).sum()
    fair_corr = ((metrics_df['correlation'] > 0.4) & (metrics_df['correlation'] <= 0.6)).sum()
    poor_corr = (metrics_df['correlation'] <= 0.4).sum()
    
    # R² performance categories
    excellent_r2 = (metrics_df['r_squared'] > 0.64).sum()  # R² > 0.8²
    good_r2 = ((metrics_df['r_squared'] > 0.36) & (metrics_df['r_squared'] <= 0.64)).sum()  # 0.6² < R² ≤ 0.8²
    fair_r2 = ((metrics_df['r_squared'] > 0.16) & (metrics_df['r_squared'] <= 0.36)).sum()  # 0.4² < R² ≤ 0.6²
    poor_r2 = (metrics_df['r_squared'] <= 0.16).sum()  # R² ≤ 0.4²
    
    print(f"\nCORRELATION PERFORMANCE BREAKDOWN:")
    print(f"   Excellent (R > 0.8): {excellent_corr:3d} points ({excellent_corr/len(metrics_df)*100:.1f}%)")
    print(f"   Good (0.6 < R ≤ 0.8): {good_corr:3d} points ({good_corr/len(metrics_df)*100:.1f}%)")
    print(f"   Fair (0.4 < R ≤ 0.6): {fair_corr:3d} points ({fair_corr/len(metrics_df)*100:.1f}%)")
    print(f"   Poor (R ≤ 0.4):       {poor_corr:3d} points ({poor_corr/len(metrics_df)*100:.1f}%)")
    
    print(f"\nR² PERFORMANCE BREAKDOWN:")
    print(f"   Excellent (R² > 0.64): {excellent_r2:3d} points ({excellent_r2/len(metrics_df)*100:.1f}%)")
    print(f"   Good (0.36 < R² ≤ 0.64): {good_r2:3d} points ({good_r2/len(metrics_df)*100:.1f}%)")
    print(f"   Fair (0.16 < R² ≤ 0.36): {fair_r2:3d} points ({fair_r2/len(metrics_df)*100:.1f}%)")
    print(f"   Poor (R² ≤ 0.16):        {poor_r2:3d} points ({poor_r2/len(metrics_df)*100:.1f}%)")
    
    # Model performance insights
    high_performance = ((metrics_df['r_squared'] > 0.5) & (metrics_df['rmse'] < metrics_df['rmse'].median())).sum()
    print(f"\nHIGH PERFORMANCE POINTS:")
    print(f"   R² > 0.5 AND RMSE < median: {high_performance:3d} points ({high_performance/len(metrics_df)*100:.1f}%)")

def main():
    """Main function"""
    print("FWI Annual Performance Analysis")
    print("=" * 50)
    
    try:
        create_annual_performance_maps()
        print("\nAnnual performance analysis completed successfully!")
        print("Generated files:")
        print("   fwi_annual_performance_metrics.png - Performance maps")
        print("   fwi_annual_performance_summary.csv - Statistics summary")
        
    except Exception as e:
        print(f"ERROR: Error in performance analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()