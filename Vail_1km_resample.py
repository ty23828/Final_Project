import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

class FWIValidation1km:
    def __init__(self):
        print("FWI 1KM RESAMPLING VALIDATION")
        print("="*60)
        
    def load_1km_data(self, filepath='fwi_1km_resampled.csv'):
        """Load 1km resampled FWI data with memory optimization"""
        try:
            print(f"\nLoading 1km resampled data from: {filepath}")
            
            if not os.path.exists(filepath):
                print(f"ERROR: File not found: {filepath}")
                return None
            
            # First check file size and number of rows
            print("   Checking file size...")
            file_size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"   File size: {file_size_mb:.1f} MB")
            
            # Load with optimized data types to save memory
            dtype_dict = {
                'latitude': 'float32',
                'longitude': 'float32', 
                'fwi_1km': 'float32'
            }
            
            # Load in chunks if file is very large
            if file_size_mb > 500:  # If larger than 500MB
                print("   Large file detected - using chunked loading...")
                chunk_size = 100000
                chunks = []
                
                for chunk in pd.read_csv(filepath, chunksize=chunk_size, dtype=dtype_dict):
                    chunks.append(chunk)
                    if len(chunks) % 10 == 0:
                        print(f"      Loaded {len(chunks) * chunk_size:,} records...")
                
                data = pd.concat(chunks, ignore_index=True)
                del chunks  # Free memory
            else:
                data = pd.read_csv(filepath, dtype=dtype_dict)
            
            print(f"SUCCESS: Loaded {len(data):,} 1km records")
            
            # Basic info
            print(f"1km Data Overview:")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            print(f"   Lat range: [{data['latitude'].min():.3f}, {data['latitude'].max():.3f}]")
            print(f"   Lon range: [{data['longitude'].min():.3f}, {data['longitude'].max():.3f}]")
            print(f"   FWI range: [{data['fwi_1km'].min():.2f}, {data['fwi_1km'].max():.2f}]")
            
            return data
            
        except Exception as e:
            print(f"ERROR: Error loading 1km data: {e}")
            return None
    
    def load_original_25km_data(self, filepath='experiment/ERA5_reanalysis_fwi/era5_fwi_2017_portugal_3decimal.csv'):
        """Load original 25km ERA5 FWI data"""
        try:
            print(f"\nLoading original 25km ERA5 data from: {filepath}")
            
            if not os.path.exists(filepath):
                print(f"ERROR: File not found: {filepath}")
                return None
            
            # Use optimized data types
            dtype_dict = {
                'latitude': 'float32',
                'longitude': 'float32'
            }
            
            data = pd.read_csv(filepath, dtype=dtype_dict)
            print(f"SUCCESS: Loaded {len(data):,} original 25km records")
            
            # Basic info
            print(f"Original 25km Data Overview:")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            print(f"   Lat range: [{data['latitude'].min():.3f}, {data['latitude'].max():.3f}]")
            print(f"   Lon range: [{data['longitude'].min():.3f}, {data['longitude'].max():.3f}]")
            
            # Check FWI column name
            fwi_col = None
            for col in data.columns:
                if 'fwi' in col.lower():
                    fwi_col = col
                    break
            
            if fwi_col is None:
                print(f"ERROR: No FWI column found in original data")
                return None
            
            print(f"   FWI column: {fwi_col}")
            print(f"   FWI range: [{data[fwi_col].min():.2f}, {data[fwi_col].max():.2f}]")
            
            # Standardize column name and convert to float32
            data[fwi_col] = data[fwi_col].astype('float32')
            data = data.rename(columns={fwi_col: 'fwi_original'})
            
            return data
            
        except Exception as e:
            print(f"ERROR: Error loading original data: {e}")
            return None
    
    def aggregate_1km_to_25km_optimized(self, data_1km, target_25km_points):
        """Memory-optimized aggregation of 1km data back to 25km resolution"""
        try:
            print(f"\nAggregating 1km data back to 25km resolution (optimized)...")
            
            # Sample 1km data if it's too large
            if len(data_1km) > 1000000:  # More than 1M points
                sample_size = 500000  # Sample 500k points
                print(f"   Large dataset detected ({len(data_1km):,} points)")
                print(f"   Sampling {sample_size:,} points for validation...")
                data_1km_sample = data_1km.sample(n=sample_size, random_state=42).copy()
            else:
                data_1km_sample = data_1km.copy()
            
            print(f"   Using {len(data_1km_sample):,} 1km points for aggregation")
            
            # Extract coordinates and values (using float32 to save memory)
            coords_1km = data_1km_sample[['latitude', 'longitude']].values.astype(np.float32)
            fwi_1km = data_1km_sample['fwi_1km'].values.astype(np.float32)
            
            # Sample 25km points if too many
            if len(target_25km_points) > 1000:
                sample_25km_size = 500  # Sample 500 25km points
                print(f"   Sampling {sample_25km_size} 25km points for validation...")
                target_25km_sample = target_25km_points.sample(n=sample_25km_size, random_state=42).copy()
            else:
                target_25km_sample = target_25km_points.copy()
            
            coords_25km = target_25km_sample[['latitude', 'longitude']].values.astype(np.float32)
            
            print(f"   Processing {len(coords_25km)} 25km grid points...")
            
            aggregated_results = []
            
            # Process in smaller batches to manage memory
            batch_size = 50  # Process 50 25km points at a time
            n_batches = len(coords_25km) // batch_size + (1 if len(coords_25km) % batch_size else 0)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(coords_25km))
                
                print(f"      Processing batch {batch_idx + 1}/{n_batches} (points {start_idx+1}-{end_idx})")
                
                batch_coords = coords_25km[start_idx:end_idx]
                
                for i, target_coord in enumerate(batch_coords):
                    # Calculate distances (vectorized for efficiency)
                    distances = np.sqrt(np.sum((coords_1km - target_coord[np.newaxis, :])**2, axis=1))
                    
                    # Define aggregation radius
                    radius = 0.15  # ~15km radius
                    
                    # Find points within radius
                    within_radius = distances <= radius
                    nearby_fwi = fwi_1km[within_radius]
                    
                    if len(nearby_fwi) > 0:
                        # Calculate aggregation metrics
                        fwi_mean = float(np.mean(nearby_fwi))
                        fwi_median = float(np.median(nearby_fwi))
                        fwi_std = float(np.std(nearby_fwi)) if len(nearby_fwi) > 1 else 0.0
                        fwi_min = float(np.min(nearby_fwi))
                        fwi_max = float(np.max(nearby_fwi))
                        count_points = int(len(nearby_fwi))
                        
                        # Inverse distance weighting
                        if len(nearby_fwi) > 1:
                            nearby_distances = distances[within_radius]
                            weights = 1.0 / (nearby_distances + 1e-10)
                            weights = weights / np.sum(weights)
                            fwi_weighted = float(np.sum(nearby_fwi * weights))
                        else:
                            fwi_weighted = fwi_mean
                        
                    else:
                        # Use nearest point
                        nearest_idx = np.argmin(distances)
                        fwi_mean = float(fwi_1km[nearest_idx])
                        fwi_median = fwi_mean
                        fwi_weighted = fwi_mean
                        fwi_std = 0.0
                        fwi_min = fwi_mean
                        fwi_max = fwi_mean
                        count_points = 1
                    
                    aggregated_results.append({
                        'latitude': float(target_coord[0]),
                        'longitude': float(target_coord[1]),
                        'fwi_aggregated_mean': fwi_mean,
                        'fwi_aggregated_median': fwi_median,
                        'fwi_aggregated_weighted': fwi_weighted,
                        'fwi_std': fwi_std,
                        'fwi_min': fwi_min,
                        'fwi_max': fwi_max,
                        'count_1km_points': count_points
                    })
            
            aggregated_df = pd.DataFrame(aggregated_results)
            print(f"SUCCESS: Aggregation completed: {len(aggregated_df)} 25km points")
            
            return aggregated_df
            
        except Exception as e:
            print(f"ERROR: Error during aggregation: {e}")
            return None
    
    def compare_and_validate(self, original_25km, aggregated_25km):
        """Compare original vs aggregated 25km data with memory optimization"""
        try:
            print(f"\nVALIDATION ANALYSIS")
            print("="*50)
            
            # Merge datasets for comparison with tolerance for coordinate matching
            print("   Merging datasets with coordinate tolerance...")
            
            # Round coordinates to avoid floating point precision issues
            original_25km_rounded = original_25km.copy()
            aggregated_25km_rounded = aggregated_25km.copy()
            
            original_25km_rounded['lat_rounded'] = original_25km_rounded['latitude'].round(3)
            original_25km_rounded['lon_rounded'] = original_25km_rounded['longitude'].round(3)
            aggregated_25km_rounded['lat_rounded'] = aggregated_25km_rounded['latitude'].round(3)
            aggregated_25km_rounded['lon_rounded'] = aggregated_25km_rounded['longitude'].round(3)
            
            merged = pd.merge(
                original_25km_rounded[['lat_rounded', 'lon_rounded', 'latitude', 'longitude', 'fwi_original']],
                aggregated_25km_rounded[['lat_rounded', 'lon_rounded'] + 
                                      [col for col in aggregated_25km_rounded.columns if 'fwi_' in col]],
                on=['lat_rounded', 'lon_rounded'],
                how='inner'
            )
            
            print(f"Comparison Dataset:")
            print(f"   Matched points: {len(merged)}")
            print(f"   Original 25km points: {len(original_25km)}")
            print(f"   Aggregated points: {len(aggregated_25km)}")
            
            if len(merged) == 0:
                print("ERROR: No matching points found for comparison")
                print("   Trying alternative matching strategy...")
                
                # Alternative: find nearest neighbors
                from scipy.spatial import cKDTree
                
                # Build tree for original points
                original_coords = original_25km[['latitude', 'longitude']].values
                tree = cKDTree(original_coords)
                
                # Find nearest matches for aggregated points
                aggregated_coords = aggregated_25km[['latitude', 'longitude']].values
                distances, indices = tree.query(aggregated_coords, k=1, distance_upper_bound=0.1)
                
                # Keep only close matches
                valid_matches = distances < 0.05  # Within ~5km
                
                if np.any(valid_matches):
                    matched_indices = indices[valid_matches]
                    
                    merged = pd.DataFrame({
                        'latitude': aggregated_25km.iloc[valid_matches]['latitude'].values,
                        'longitude': aggregated_25km.iloc[valid_matches]['longitude'].values,
                        'fwi_original': original_25km.iloc[matched_indices]['fwi_original'].values,
                        'fwi_aggregated_mean': aggregated_25km.iloc[valid_matches]['fwi_aggregated_mean'].values,
                        'fwi_aggregated_median': aggregated_25km.iloc[valid_matches]['fwi_aggregated_median'].values,
                        'fwi_aggregated_weighted': aggregated_25km.iloc[valid_matches]['fwi_aggregated_weighted'].values
                    })
                    
                    print(f"   Alternative matching found: {len(merged)} points")
                else:
                    print("ERROR: No matching points found with alternative method")
                    return None, None
            
            # Calculate validation metrics for different aggregation methods
            methods = [
                ('Mean', 'fwi_aggregated_mean'),
                ('Median', 'fwi_aggregated_median'),
                ('Weighted', 'fwi_aggregated_weighted')
            ]
            
            results = {}
            
            for method_name, method_col in methods:
                if method_col not in merged.columns:
                    print(f"   Warning: Column {method_col} not found, skipping...")
                    continue
                    
                original = merged['fwi_original'].values
                predicted = merged[method_col].values
                
                # Remove any NaN values
                valid_mask = ~(np.isnan(original) | np.isnan(predicted))
                original_clean = original[valid_mask]
                predicted_clean = predicted[valid_mask]
                
                if len(original_clean) > 0:
                    # Calculate metrics
                    r2 = r2_score(original_clean, predicted_clean)
                    rmse = np.sqrt(mean_squared_error(original_clean, predicted_clean))
                    mae = mean_absolute_error(original_clean, predicted_clean)
                    bias = np.mean(predicted_clean - original_clean)
                    
                    # Correlation
                    correlation = np.corrcoef(original_clean, predicted_clean)[0, 1]
                    
                    results[method_name] = {
                        'R²': r2,
                        'RMSE': rmse,
                        'MAE': mae,
                        'Bias': bias,
                        'Correlation': correlation,
                        'N_points': len(original_clean)
                    }
                    
                    print(f"\n{method_name} Aggregation Results:")
                    print(f"   R² Score: {r2:.4f}")
                    print(f"   RMSE: {rmse:.3f}")
                    print(f"   MAE: {mae:.3f}")
                    print(f"   Bias: {bias:.3f}")
                    print(f"   Correlation: {correlation:.4f}")
                    print(f"   Valid points: {len(original_clean)}")
            
            # Find best method
            if results:
                best_method = max(results.keys(), key=lambda x: results[x]['R²'])
                print(f"\nBest Aggregation Method: {best_method} (R² = {results[best_method]['R²']:.4f})")
            
            return merged, results
            
        except Exception as e:
            print(f"ERROR: Error during validation: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def create_validation_plots(self, merged_data, results):
        """Create validation plots with memory optimization"""
        try:
            print(f"\nCreating validation plots...")
            
            # Limit data for plotting if too large
            if len(merged_data) > 10000:
                print(f"   Large dataset ({len(merged_data)} points) - sampling for visualization...")
                plot_data = merged_data.sample(n=5000, random_state=42)
            else:
                plot_data = merged_data
            
            # Set up the plotting style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'FWI 1km→25km Aggregation Validation\n({len(plot_data)} points shown)', 
                        fontsize=16, fontweight='bold')
            
            methods = [
                ('Mean', 'fwi_aggregated_mean'),
                ('Median', 'fwi_aggregated_median'),
                ('Weighted', 'fwi_aggregated_weighted')
            ]
            
            colors = ['blue', 'red', 'green']
            
            # Plot 1: Scatter plot comparison
            ax1 = axes[0, 0]
            for i, (method_name, method_col) in enumerate(methods):
                if method_col in plot_data.columns and method_name in results:
                    ax1.scatter(plot_data['fwi_original'], plot_data[method_col], 
                               alpha=0.6, s=20, color=colors[i], 
                               label=f'{method_name} (R²={results[method_name]["R²"]:.3f})')
            
            # Perfect correlation line
            min_val = plot_data['fwi_original'].min()
            max_val = plot_data['fwi_original'].max()
            
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect correlation')
            ax1.set_xlabel('Original 25km FWI')
            ax1.set_ylabel('Aggregated 25km FWI')
            ax1.set_title('Original vs Aggregated FWI')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Residuals plot (using best method)
            if results:
                best_method = max(results.keys(), key=lambda x: results[x]['R²'])
                best_col = next(col for name, col in methods if name == best_method)
                
                if best_col in plot_data.columns:
                    ax2 = axes[0, 1]
                    residuals = plot_data[best_col] - plot_data['fwi_original']
                    ax2.scatter(plot_data['fwi_original'], residuals, alpha=0.6, s=20)
                    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                    ax2.set_xlabel('Original 25km FWI')
                    ax2.set_ylabel('Residuals (Aggregated - Original)')
                    ax2.set_title(f'Residuals Plot ({best_method} Method)')
                    ax2.grid(True, alpha=0.3)
            
            # Plot 3: Distribution comparison
            ax3 = axes[1, 0]
            ax3.hist(plot_data['fwi_original'], bins=30, alpha=0.5, label='Original 25km', 
                    color='blue', density=True)
            
            if 'fwi_aggregated_mean' in plot_data.columns:
                ax3.hist(plot_data['fwi_aggregated_mean'], bins=30, alpha=0.5, 
                        label='Aggregated (Mean)', color='red', density=True)
            
            ax3.set_xlabel('FWI Value')
            ax3.set_ylabel('Density')
            ax3.set_title('FWI Distribution Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Metrics comparison
            ax4 = axes[1, 1]
            if results:
                methods_available = [(name, col) for name, col in methods if name in results]
                metrics_to_plot = ['R²', 'RMSE', 'MAE']
                
                x_pos = np.arange(len(methods_available))
                width = 0.25
                
                for i, metric in enumerate(metrics_to_plot):
                    values = [results[method][metric] for method, _ in methods_available]
                    ax4.bar(x_pos + i*width, values, width, label=metric, alpha=0.8)
                
                ax4.set_xlabel('Aggregation Method')
                ax4.set_ylabel('Metric Value')
                ax4.set_title('Validation Metrics Comparison')
                ax4.set_xticks(x_pos + width)
                ax4.set_xticklabels([method for method, _ in methods_available])
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = 'fwi_1km_validation_plots.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"SUCCESS: Validation plots saved to: {plot_filename}")
            
            plt.show()
            
            return True
            
        except Exception as e:
            print(f"ERROR: Error creating plots: {e}")
            return False
    
    def save_validation_results(self, merged_data, results):
        """Save validation results to files"""
        try:
            print(f"\nSaving validation results...")
            
            # Save comparison data (sample if too large)
            comparison_file = 'fwi_validation_comparison.csv'
            if len(merged_data) > 100000:
                print(f"   Large dataset - saving sample of {min(50000, len(merged_data))} points")
                sample_data = merged_data.sample(n=min(50000, len(merged_data)), random_state=42)
                sample_data.to_csv(comparison_file, index=False)
            else:
                merged_data.to_csv(comparison_file, index=False)
            print(f"   Comparison data saved to: {comparison_file}")
            
            # Save metrics summary
            if results:
                metrics_df = pd.DataFrame(results).T
                metrics_file = 'fwi_validation_metrics.csv'
                metrics_df.to_csv(metrics_file)
                print(f"   Validation metrics saved to: {metrics_file}")
                
                # Create summary report
                report_file = 'fwi_validation_report.txt'
                with open(report_file, 'w') as f:
                    f.write("FWI 1KM RESAMPLING VALIDATION REPORT\n")
                    f.write("="*50 + "\n\n")
                    
                    f.write(f"Dataset Information:\n")
                    f.write(f"- Comparison points: {len(merged_data)}\n")
                    f.write(f"- Original FWI range: [{merged_data['fwi_original'].min():.2f}, {merged_data['fwi_original'].max():.2f}]\n")
                    
                    if 'fwi_aggregated_mean' in merged_data.columns:
                        f.write(f"- Aggregated FWI range: [{merged_data['fwi_aggregated_mean'].min():.2f}, {merged_data['fwi_aggregated_mean'].max():.2f}]\n\n")
                    
                    f.write("Validation Metrics:\n")
                    for method, metrics in results.items():
                        f.write(f"\n{method} Aggregation:\n")
                        for metric_name, value in metrics.items():
                            f.write(f"  {metric_name}: {value:.4f}\n")
                    
                    # Best method
                    best_method = max(results.keys(), key=lambda x: results[x]['R²'])
                    f.write(f"\nBest Method: {best_method} (R² = {results[best_method]['R²']:.4f})\n")
                    
                    # Interpretation
                    r2_best = results[best_method]['R²']
                    if r2_best > 0.9:
                        quality = "Excellent"
                    elif r2_best > 0.8:
                        quality = "Good"
                    elif r2_best > 0.7:
                        quality = "Fair"
                    else:
                        quality = "Poor"
                    
                    f.write(f"\nValidation Quality: {quality}\n")
                    f.write(f"RMSE: {results[best_method]['RMSE']:.3f}\n")
                    f.write(f"Bias: {results[best_method]['Bias']:.3f}\n")
                
                print(f"   Validation report saved to: {report_file}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Error saving results: {e}")
            return False
    
    def run_validation(self, 
                      data_1km_file='fwi_1km_resampled.csv',
                      data_25km_file='experiment/ERA5_reanalysis_fwi/era5_fwi_2017_portugal_3decimal.csv'):
        """Main validation execution with memory optimization"""
        print(f"\nStarting FWI 1km Resampling Validation (Memory Optimized)...")
        
        try:
            # Step 1: Load data
            data_1km = self.load_1km_data(data_1km_file)
            if data_1km is None:
                return False
                
            data_25km_original = self.load_original_25km_data(data_25km_file)
            if data_25km_original is None:
                return False
            
            # Step 2: Aggregate 1km back to 25km (optimized)
            aggregated_25km = self.aggregate_1km_to_25km_optimized(data_1km, data_25km_original)
            if aggregated_25km is None:
                return False
            
            # Step 3: Compare and validate
            merged_data, results = self.compare_and_validate(data_25km_original, aggregated_25km)
            if merged_data is None or results is None:
                return False
            
            # Step 4: Create plots
            self.create_validation_plots(merged_data, results)
            
            # Step 5: Save results
            self.save_validation_results(merged_data, results)
            
            print(f"\nVALIDATION COMPLETED SUCCESSFULLY!")
            if results:
                best_method = max(results.keys(), key=lambda x: results[x]['R²'])
                print(f"Best method: {best_method} (R² = {results[best_method]['R²']:.4f})")
            print(f"Check output files for detailed results.")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main execution"""
    validator = FWIValidation1km()
    
    # Check input files
    file_1km = 'fwi_1km_resampled.csv'
    file_25km = 'experiment/ERA5_reanalysis_fwi/era5_fwi_2017_portugal_3decimal.csv'
    
    missing_files = []
    if not os.path.exists(file_1km):
        missing_files.append(file_1km)
    if not os.path.exists(file_25km):
        missing_files.append(file_25km)
    
    if missing_files:
        print(f"ERROR: Missing input files:")
        for file in missing_files:
            print(f"   - {file}")
        return
    
    # Run validation
    success = validator.run_validation(file_1km, file_25km)
    
    if success:
        print("\nValidation completed successfully!")
        print("Output files:")
        print("   - fwi_validation_comparison.csv (detailed comparison data)")
        print("   - fwi_validation_metrics.csv (validation metrics)")
        print("   - fwi_validation_report.txt (summary report)")
        print("   - fwi_1km_validation_plots.png (visualization)")
    else:
        print("\nValidation failed. Check error messages above.")

if __name__ == "__main__":
    main()