import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class FWIResampler1km:
    def __init__(self):
        print("FWI 1KM RESAMPLER - DIRECT INTERPOLATION")
        print("="*60)
        
    def load_era5_data(self, filepath='merged_era5_final.csv'):
        """Load ERA5 merged data"""
        try:
            print(f"Loading ERA5 data from: {filepath}")
            
            if not os.path.exists(filepath):
                print(f"ERROR: File not found: {filepath}")
                return None
                
            # Load data
            data = pd.read_csv(filepath)
            print(f"SUCCESS: Loaded {len(data):,} records")
            
            # Check required columns
            required_cols = ['latitude', 'longitude', 'fwi']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                print(f"ERROR: Missing required columns: {missing_cols}")
                return None
                
            # Basic info
            print(f"Data Overview:")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Date range: {data['date'].min()} to {data['date'].max()}" if 'date' in data.columns else "   No date column")
            print(f"   Lat range: [{data['latitude'].min():.3f}, {data['latitude'].max():.3f}]")
            print(f"   Lon range: [{data['longitude'].min():.3f}, {data['longitude'].max():.3f}]")
            print(f"   FWI range: [{data['fwi'].min():.2f}, {data['fwi'].max():.2f}]")
            
            return data
            
        except Exception as e:
            print(f"ERROR: Error loading data: {e}")
            return None
    
    def create_1km_grid(self, data):
        """Create 1km resolution grid"""
        try:
            print(f"\nCreating 1km resolution grid...")
            
            # Get data bounds
            lat_min, lat_max = data['latitude'].min(), data['latitude'].max()
            lon_min, lon_max = data['longitude'].min(), data['longitude'].max()
            
            print(f"   Source bounds: Lat[{lat_min:.3f}, {lat_max:.3f}], Lon[{lon_min:.3f}, {lon_max:.3f}]")
            
            # Create 1km grid (approximately 0.009 degrees ≈ 1km)
            resolution = 0.009  # ~1km at mid-latitudes
            
            # Create grid points
            lat_1km = np.arange(lat_min, lat_max + resolution, resolution)
            lon_1km = np.arange(lon_min, lon_max + resolution, resolution)
            
            # Create meshgrid
            lon_grid, lat_grid = np.meshgrid(lon_1km, lat_1km)
            
            # Flatten for interpolation
            target_points = np.column_stack([
                lat_grid.flatten(),
                lon_grid.flatten()
            ])
            
            print(f"   1km grid size: {len(lat_1km)} x {len(lon_1km)} = {len(target_points):,} points")
            print(f"   Grid resolution: ~{resolution*111:.1f}km")
            
            return target_points, lat_grid.shape
            
        except Exception as e:
            print(f"ERROR: Error creating 1km grid: {e}")
            return None, None
    
    def interpolate_fwi_to_1km(self, data, target_points, grid_shape):
        """Interpolate FWI values to 1km grid using various methods"""
        try:
            print(f"\nInterpolating FWI to 1km resolution...")
            
            results = []
            
            # Check if we have date column for daily processing
            if 'date' in data.columns:
                dates = data['date'].unique()
                print(f"   Processing {len(dates)} unique dates...")
                
                for i, date in enumerate(dates):
                    if i % 10 == 0:
                        print(f"      Processing date {i+1}/{len(dates)}: {date}")
                    
                    # Get data for this date
                    daily_data = data[data['date'] == date].copy()
                    
                    if len(daily_data) == 0:
                        continue
                    
                    # Source points and values
                    source_points = daily_data[['latitude', 'longitude']].values
                    fwi_values = daily_data['fwi'].values
                    
                    # Remove NaN values
                    valid_mask = ~np.isnan(fwi_values)
                    if not np.any(valid_mask):
                        continue
                        
                    source_points = source_points[valid_mask]
                    fwi_values = fwi_values[valid_mask]
                    
                    # Interpolate using different methods
                    try:
                        # Method 1: Linear interpolation (fastest, good quality)
                        fwi_interp = griddata(
                            source_points, fwi_values, target_points,
                            method='linear', fill_value=np.nan
                        )
                        
                        # Fill NaN values with nearest neighbor
                        nan_mask = np.isnan(fwi_interp)
                        if np.any(nan_mask):
                            fwi_nearest = griddata(
                                source_points, fwi_values, target_points[nan_mask],
                                method='nearest'
                            )
                            fwi_interp[nan_mask] = fwi_nearest
                        
                        # Create daily results
                        daily_result = pd.DataFrame({
                            'latitude': target_points[:, 0],
                            'longitude': target_points[:, 1],
                            'fwi_1km': fwi_interp,
                            'date': date
                        })
                        
                        results.append(daily_result)
                        
                    except Exception as e:
                        print(f"         WARNING: Error interpolating date {date}: {e}")
                        continue
                
            else:
                # Single time interpolation
                print("   Single time interpolation...")
                
                # Source points and values
                source_points = data[['latitude', 'longitude']].values
                fwi_values = data['fwi'].values
                
                # Remove NaN values
                valid_mask = ~np.isnan(fwi_values)
                source_points = source_points[valid_mask]
                fwi_values = fwi_values[valid_mask]
                
                # Interpolate
                fwi_interp = griddata(
                    source_points, fwi_values, target_points,
                    method='linear', fill_value=np.nan
                )
                
                # Fill NaN with nearest neighbor
                nan_mask = np.isnan(fwi_interp)
                if np.any(nan_mask):
                    fwi_nearest = griddata(
                        source_points, fwi_values, target_points[nan_mask],
                        method='nearest'
                    )
                    fwi_interp[nan_mask] = fwi_nearest
                
                # Create results
                result = pd.DataFrame({
                    'latitude': target_points[:, 0],
                    'longitude': target_points[:, 1],
                    'fwi_1km': fwi_interp
                })
                results.append(result)
            
            # Combine all results
            if results:
                final_result = pd.concat(results, ignore_index=True)
                print(f"SUCCESS: Interpolation completed: {len(final_result):,} 1km points")
                return final_result
            else:
                print("ERROR: No valid interpolation results")
                return None
                
        except Exception as e:
            print(f"ERROR: Error during interpolation: {e}")
            return None
    
    def save_results(self, results, output_filename='fwi_1km_resampled.csv'):
        """Save 1km results to single file without timestamp"""
        try:
            print(f"\nSaving 1km results...")
            
            # Calculate file size estimate
            estimated_size_mb = len(results) * len(results.columns) * 8 / (1024*1024)
            print(f"   Estimated file size: ~{estimated_size_mb:.0f}MB")
            
            # Save all data to single file
            print(f"   Saving all {len(results):,} records to: {output_filename}")
            results.to_csv(output_filename, index=False)
            
            # Check actual file size
            if os.path.exists(output_filename):
                actual_size_mb = os.path.getsize(output_filename) / (1024*1024)
                print(f"SUCCESS: Results saved successfully!")
                print(f"   File: {output_filename}")
                print(f"   Size: {actual_size_mb:.1f}MB")
                print(f"   Records: {len(results):,}")
                return [output_filename]
            else:
                print(f"ERROR: Failed to create output file")
                return []
                
        except Exception as e:
            print(f"ERROR: Error saving results: {e}")
            return []

    def generate_analysis(self, results):
        """Generate analysis of 1km results"""
        try:
            print(f"\n1KM RESAMPLING ANALYSIS")
            print("="*50)
            
            print(f"Dataset Statistics:")
            print(f"   Total points: {len(results):,}")
            print(f"   Unique coordinates: {len(results[['latitude', 'longitude']].drop_duplicates()):,}")
            
            if 'date' in results.columns:
                print(f"   Date range: {results['date'].min()} to {results['date'].max()}")
                print(f"   Number of dates: {results['date'].nunique()}")
            
            print(f"\nFWI Statistics:")
            fwi_stats = results['fwi_1km'].describe()
            print(f"   Mean: {fwi_stats['mean']:.2f}")
            print(f"   Std:  {fwi_stats['std']:.2f}")
            print(f"   Min:  {fwi_stats['min']:.2f}")
            print(f"   Max:  {fwi_stats['max']:.2f}")
            print(f"   25%:  {fwi_stats['25%']:.2f}")
            print(f"   50%:  {fwi_stats['50%']:.2f}")
            print(f"   75%:  {fwi_stats['75%']:.2f}")
            
            print(f"\nSpatial Coverage:")
            print(f"   Latitude range:  [{results['latitude'].min():.3f}, {results['latitude'].max():.3f}]")
            print(f"   Longitude range: [{results['longitude'].min():.3f}, {results['longitude'].max():.3f}]")
            
            # FWI distribution
            print(f"\nFWI Distribution:")
            fwi_bins = [0, 5, 10, 20, 30, 50, 100]
            fwi_labels = ['Very Low (0-5)', 'Low (5-10)', 'Moderate (10-20)', 
                         'High (20-30)', 'Very High (30-50)', 'Extreme (50+)']
            
            fwi_cut = pd.cut(results['fwi_1km'], bins=fwi_bins, labels=fwi_labels, include_lowest=True)
            fwi_dist = fwi_cut.value_counts().sort_index()
            
            for category, count in fwi_dist.items():
                percentage = (count / len(results)) * 100
                print(f"   {category}: {count:,} ({percentage:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Error generating analysis: {e}")
            return False
    
    def run_resampling(self, input_file='merged_era5_final.csv', output_file='fwi_1km_resampled.csv'):
        """Main execution function"""
        print(f"\nStarting FWI 1km Resampling Process...")
        
        try:
            # Step 1: Load data
            data = self.load_era5_data(input_file)
            if data is None:
                return False
            
            # Step 2: Create 1km grid
            target_points, grid_shape = self.create_1km_grid(data)
            if target_points is None:
                return False
            
            # Step 3: Interpolate to 1km
            results_1km = self.interpolate_fwi_to_1km(data, target_points, grid_shape)
            if results_1km is None:
                return False
            
            # Step 4: Save results (modified to use specified filename)
            output_files = self.save_results(results_1km, output_file)
            if not output_files:
                return False
            
            # Step 5: Generate analysis
            self.generate_analysis(results_1km)
            
            print(f"\n1KM RESAMPLING COMPLETED SUCCESSFULLY!")
            print(f"Output file: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Resampling failed: {e}")
            return False

def main():
    """Main execution"""
    resampler = FWIResampler1km()
    
    # Check if input file exists
    input_file = 'experiment/ERA5_reanalysis_FWI/era5_fwi_2017_portugal_3decimal.csv'
    output_file = 'fwi_1km_resampled.csv'  # Fixed filename without timestamp
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("   Please make sure the file is in the current directory.")
        return
    
    # Run resampling
    success = resampler.run_resampling(input_file, output_file)
    
    if success:
        print("\nProcess completed successfully!")
        print(f"Output saved to: {output_file}")
    else:
        print("\nProcess failed. Check error messages above.")

if __name__ == "__main__":
    main()