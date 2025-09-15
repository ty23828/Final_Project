import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

class FWISuperResolutionEDA:
    """
    FWI Super-Resolution - Data Loading & EDA
    """

    def __init__(self):
        self.temp_data = None
        self.atmospheric_data = None
        self.fwi_data = None
        self.merged_25km = None
        self.coarse_data = None
        self.training_pairs = None

        print("FWI SUPER-RESOLUTION SYSTEM (EDA)")
        print("="*60)
        print("Phase 1: Data loading → Preprocessing → Downsampling")
        print("="*60)

    def load_era5_data(self):
        """Load ERA5 data files"""
        print("\nLoading ERA5 Data...")
        
        files = {
            'temp': "experiment/ERA5_reanalysis_atmospheric_parameters/era5_daily_max_temp_2017_portugal.csv",
            'atmospheric': "experiment/ERA5_reanalysis_atmospheric_parameters/era5_daily_mean_2017_combined_3decimal.csv", 
            'fwi': "experiment/ERA5_reanalysis_fwi/era5_fwi_2017_portugal_3decimal.csv"
        }
        
        try:
            datasets = {}
            for name, file_path in files.items():
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    datasets[name] = df
                    print(f"   LOADED {name}: {df.shape}")
                else:
                    print(f"   ERROR File not found: {file_path}")
                    return False
            
            self.temp_data = datasets['temp']
            self.atmospheric_data = datasets['atmospheric']
            self.fwi_data = datasets['fwi']
            
            return True
            
        except Exception as e:
            print(f"ERROR loading data: {e}")
            return False

    def preprocess_and_merge(self):
        """Merge and preprocess all ERA5 datasets"""
        print("\nPreprocessing and Merging Data...")
        
        try:
            # Standardize datasets
            datasets = {
                'temp': self.temp_data,
                'atmospheric': self.atmospheric_data,
                'fwi': self.fwi_data
            }
            
            processed = {}
            
            for name, df in datasets.items():
                df_clean = df.copy()
                
                # First, let's examine what columns we have
                print(f"   {name} original columns: {list(df_clean.columns)}")
                
                # Standardize column names with priority handling
                column_mapping = {}
                
                # Handle coordinate columns first
                for col in df_clean.columns:
                    col_lower = col.lower()
                    if 'lat' in col_lower and 'latitude' not in col_lower and 'latitude' not in column_mapping.values():
                        column_mapping[col] = 'latitude'
                    elif 'lon' in col_lower and 'longitude' not in col_lower and 'longitude' not in column_mapping.values():
                        column_mapping[col] = 'longitude'
                
                # Handle time/date columns with priority (avoid duplicates)
                time_candidates = []
                for col in df_clean.columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in ['time', 'date']) and col not in column_mapping:
                        time_candidates.append(col)
                
                # Select the best time column
                if time_candidates:
                    # Prioritize 'date' over 'time', and shorter names over longer ones
                    time_col = None
                    for priority_name in ['date', 'time']:
                        for candidate in time_candidates:
                            if priority_name in candidate.lower():
                                time_col = candidate
                                break
                        if time_col:
                            break
                
                    # If no priority match, use the first candidate
                    if not time_col:
                        time_col = time_candidates[0]
                    
                    column_mapping[time_col] = 'date'
                    print(f"   Selected time column for {name}: {time_col}")
                
                # Apply column mapping
                if column_mapping:
                    df_clean = df_clean.rename(columns=column_mapping)
                    print(f"   Applied mapping for {name}: {column_mapping}")
                
                # Handle coordinates with proper precision
                if 'latitude' in df_clean.columns:
                    df_clean['latitude'] = np.round(df_clean['latitude'].astype(float), 3)
                if 'longitude' in df_clean.columns:
                    df_clean['longitude'] = np.round(df_clean['longitude'].astype(float), 3)
                
                # Handle time conversion more carefully
                if 'date' in df_clean.columns:
                    try:
                        # Check if it's already datetime
                        if df_clean['date'].dtype == 'object' or 'datetime' not in str(df_clean['date'].dtype):
                            # Try different parsing approaches
                            try:
                                df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
                            except ValueError as e:
                                if "duplicate keys" in str(e):
                                    # Handle the specific duplicate keys error
                                    print(f"   WARNING Handling duplicate keys in date parsing for {name}")
                                    # Convert to string first, then parse
                                    df_clean['date'] = df_clean['date'].astype(str)
                                    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
                                else:
                                    raise e
                        
                        # Convert to date only (remove time component)
                        df_clean['date'] = df_clean['date'].dt.date
                        
                        print(f"   Date conversion successful for {name}")
                        
                    except Exception as date_error:
                        print(f"   WARNING Date conversion failed for {name}: {date_error}")
                        # Try alternative approaches
                        try:
                            # If it's a string representation, try direct conversion
                            if df_clean['date'].dtype == 'object':
                                # Sample a few values to understand the format
                                sample_dates = df_clean['date'].dropna().head(3).tolist()
                                print(f"   Sample date values: {sample_dates}")
                                
                                # Try parsing with infer_datetime_format
                                df_clean['date'] = pd.to_datetime(df_clean['date'], 
                                                                infer_datetime_format=True, 
                                                                errors='coerce').dt.date
                            else:
                                # Already in datetime format, just extract date
                                df_clean['date'] = pd.to_datetime(df_clean['date']).dt.date
                                
                        except Exception as fallback_error:
                            print(f"   ERROR All date parsing attempts failed for {name}: {fallback_error}")
                            # Last resort: keep original if it looks like dates
                            if df_clean['date'].dtype == 'object':
                                sample_val = str(df_clean['date'].iloc[0])
                                if any(char.isdigit() for char in sample_val) and ('-' in sample_val or '/' in sample_val):
                                    print(f"   Keeping original date format for {name}")
                                else:
                                    print(f"   ERROR Invalid date format in {name}, skipping this dataset")
                                    continue
            
            processed[name] = df_clean
            print(f"   {name} processed: {df_clean.shape} - {list(df_clean.columns)}")
        
            if len(processed) == 0:
                print("ERROR No datasets successfully processed")
                return False
            
            # Merge datasets using FWI as base (if available)
            print("   Merging datasets...")
            
            if 'fwi' in processed:
                merged = processed['fwi'].copy()
                base_dataset = 'fwi'
            elif 'atmospheric' in processed:
                merged = processed['atmospheric'].copy()
                base_dataset = 'atmospheric'
            elif 'temp' in processed:
                merged = processed['temp'].copy()
                base_dataset = 'temp'
            else:
                print("ERROR No suitable base dataset found")
                return False
            
            print(f"   Using {base_dataset} as base dataset: {merged.shape}")
            
            # Filter to Portugal region
            if 'latitude' in merged.columns and 'longitude' in merged.columns:
                portugal_mask = (
                    (merged['latitude'] >= 36.0) & (merged['latitude'] <= 43.0) &
                    (merged['longitude'] >= -10.0) & (merged['longitude'] <= -6.0)
                )
                merged_filtered = merged[portugal_mask].copy()
                
                if len(merged_filtered) > 0:
                    merged = merged_filtered
                    print(f"   Portugal region filtered: {len(merged):,} points")
                else:
                    print("   WARNING No data in Portugal region, expanding search...")
                    # Expand to broader Iberian region
                    expanded_mask = (
                        (merged['latitude'] >= 35.0) & (merged['latitude'] <= 44.0) &
                        (merged['longitude'] >= -11.0) & (merged['longitude'] <= -5.0)
                    )
                    merged_expanded = merged[expanded_mask].copy()
                    
                    if len(merged_expanded) > 0:
                        merged = merged_expanded
                        print(f"   Expanded to broader Iberian region: {len(merged):,} points")
                    else:
                        print("   WARNING No data found in expanded region, using all available data")
                        print(f"   Using all available data: {len(merged):,} points")
            else:
                print("   WARNING No coordinate columns found, skipping regional filtering")
            
            # Merge with other datasets
            for dataset_name in processed.keys():
                if dataset_name == base_dataset:
                    continue
                    
                other_df = processed[dataset_name]
                print(f"   Attempting to merge with {dataset_name}...")
                
                # Check for common columns
                common_cols = ['date', 'latitude', 'longitude']
                available_common = [col for col in common_cols if col in merged.columns and col in other_df.columns]
                
                print(f"      Common columns for merge: {available_common}")
                
                if len(available_common) >= 2:  # Need at least 2 common columns
                    try:
                        # Try exact merge first
                        merged_with_other = pd.merge(
                            merged,
                            other_df,
                            on=available_common,
                            how='inner',
                            suffixes=('', f'_{dataset_name}')
                        )
                        
                        retention_rate = len(merged_with_other) / len(merged)
                        print(f"      Exact merge retention: {retention_rate:.1%}")
                        
                        if retention_rate > 0.1:  # Keep if >10% retained
                            merged = merged_with_other
                            print(f"   Successfully merged with {dataset_name}: {len(merged):,} points")
                        else:
                            print(f"   WARNING Poor retention with {dataset_name}, trying approximate merge...")
                            # Try approximate merge with coordinate tolerance
                            merged_approx = self._approximate_merge(merged, other_df, tolerance=0.1)
                            if len(merged_approx) > len(merged) * 0.8:  # If we retain >80% of base data
                                merged = merged_approx
                                print(f"   Approximate merge successful: {len(merged):,} points")
                            else:
                                print(f"   WARNING Skipping {dataset_name} due to poor spatial overlap")
                    
                    except Exception as merge_error:
                        print(f"   ERROR Merge with {dataset_name} failed: {merge_error}")
                        continue
                else:
                    print(f"   WARNING Insufficient common columns for {dataset_name}")
            
            # Clean and finalize
            merged = merged.dropna(subset=['latitude', 'longitude'])
            
            # Identify and standardize FWI column
            fwi_cols = [col for col in merged.columns if 'fwi' in col.lower() and 'index' not in col.lower()]
            if fwi_cols:
                # Use the first FWI column found
                merged['fwi'] = merged[fwi_cols[0]]
                print(f"   Using FWI column: {fwi_cols[0]}")
            else:
                print(f"   WARNING No FWI column found in merged data")
                print(f"   Available columns: {list(merged.columns)}")
            
            self.merged_25km = merged
            
            print(f"SUCCESS Preprocessing completed: {merged.shape}")
            if 'date' in merged.columns:
                date_range = merged['date'].agg(['min', 'max'])
                print(f"   Date range: {date_range['min']} to {date_range['max']}")
            
            if 'latitude' in merged.columns and 'longitude' in merged.columns:
                spatial_stats = merged[['latitude', 'longitude']].describe()
                print(f"   Spatial extent:")
                print(f"      Latitude: [{spatial_stats.loc['min', 'latitude']:.3f}, {spatial_stats.loc['max', 'latitude']:.3f}]")
                print(f"      Longitude: [{spatial_stats.loc['min', 'longitude']:.3f}, {spatial_stats.loc['max', 'longitude']:.3f}]")
            
            if 'fwi' in merged.columns:
                fwi_stats = merged['fwi'].describe()
                print(f"   FWI statistics: mean={fwi_stats['mean']:.2f}, std={fwi_stats['std']:.2f}, range=[{fwi_stats['min']:.1f}, {fwi_stats['max']:.1f}]")
            
            return True
        
        except Exception as e:
            print(f"ERROR in preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_downsampled_training_data(self, coarse_resolution='50km'):
        """Create training data by downsampling 25km to coarser resolution"""
        print(f"\nCreating Downsampled Training Data ({coarse_resolution})...")
        
        if self.merged_25km is None or len(self.merged_25km) == 0:
            print("ERROR No 25km data available for downsampling")
            return False
        
        try:
            # Define downsampling parameters
            if coarse_resolution == '50km':
                downsample_factor = 2.0  # 25km → 50km
                grid_spacing = 0.5  # ~50km grid
            elif coarse_resolution == '75km':
                downsample_factor = 3.0  # 25km → 75km  
                grid_spacing = 0.75  # ~75km grid
            else:
                downsample_factor = 2.0
                grid_spacing = 0.5
            
            print(f"   Downsampling factor: {downsample_factor}x")
            print(f"   Target grid spacing: {grid_spacing}°")
            
            # Get unique dates for temporal processing
            unique_dates = sorted(self.merged_25km['date'].unique())
            print(f"   Processing {len(unique_dates)} dates")
            
            all_training_pairs = []
            
            for i, date in enumerate(unique_dates):
                if (i + 1) % 20 == 0:
                    print(f"      Processing date {i+1}/{len(unique_dates)}: {date}")
                
                # Get daily data
                daily_data = self.merged_25km[self.merged_25km['date'] == date].copy()
                
                if len(daily_data) < 4:  # Need minimum points for interpolation
                    continue
                
                # Create training pairs for this date
                daily_pairs = self._create_daily_training_pairs(daily_data, grid_spacing, downsample_factor)
                
                if daily_pairs is not None and len(daily_pairs) > 0:
                    daily_pairs['date'] = date
                    all_training_pairs.append(daily_pairs)
            
            if not all_training_pairs:
                print("ERROR No training pairs created")
                return False
            
            # Combine all training pairs
            self.training_pairs = pd.concat(all_training_pairs, ignore_index=True)
            
            print(f"SUCCESS Training data created: {len(self.training_pairs):,} samples")
            print(f"   Features: {[col for col in self.training_pairs.columns if col.startswith('coarse_')]}")
            print(f"   Target: {[col for col in self.training_pairs.columns if col.startswith('fine_')]}")
            
            # Analyze training data quality
            self._analyze_training_data()
            
            return True
            
        except Exception as e:
            print(f"ERROR creating training data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_daily_training_pairs(self, daily_data, grid_spacing, downsample_factor):
        """Create training pairs for a single date"""
        try:
            if len(daily_data) < 4:
                return None
            
            # Define spatial extent
            lat_min, lat_max = daily_data['latitude'].min(), daily_data['latitude'].max()
            lon_min, lon_max = daily_data['longitude'].min(), daily_data['longitude'].max()
            
            # Create coarse grid
            coarse_lats = np.arange(
                np.floor(lat_min / grid_spacing) * grid_spacing,
                np.ceil(lat_max / grid_spacing) * grid_spacing + grid_spacing,
                grid_spacing
            )
            coarse_lons = np.arange(
                np.floor(lon_min / grid_spacing) * grid_spacing,
                np.ceil(lon_max / grid_spacing) * grid_spacing + grid_spacing,
                grid_spacing
            )
            
            coarse_lon_mesh, coarse_lat_mesh = np.meshgrid(coarse_lons, coarse_lats)
            coarse_points = np.column_stack([coarse_lat_mesh.flatten(), coarse_lon_mesh.flatten()])
            
            # Downsample original data to coarse grid
            coarse_data = self._downsample_to_grid(daily_data, coarse_points, grid_spacing)
            
            if coarse_data is None or len(coarse_data) == 0:
                return None
            
            # Create fine grid (same as original 25km)
            fine_points = daily_data[['latitude', 'longitude']].values
            
            # For each fine point, find corresponding coarse data and create training pair
            training_samples = []
            
            for fine_idx, fine_point in enumerate(fine_points):
                fine_row = daily_data.iloc[fine_idx]
                
                # Find nearest coarse points (for feature extraction)
                coarse_features = self._extract_coarse_features(fine_point, coarse_data, radius=grid_spacing*1.5)
                
                if coarse_features is not None:
                    # Create training sample
                    sample = {
                        'fine_lat': fine_point[0],
                        'fine_lon': fine_point[1],
                        'fine_fwi': fine_row.get('fwi', np.nan)
                    }
                    
                    # Add coarse features
                    sample.update(coarse_features)
                    
                    # Add other fine variables as targets
                    for col in fine_row.index:
                        if col not in ['latitude', 'longitude', 'date'] and pd.notna(fine_row[col]):
                            sample[f'fine_{col}'] = fine_row[col]
                    
                    training_samples.append(sample)
            
            if training_samples:
                return pd.DataFrame(training_samples)
            else:
                return None
                
        except Exception as e:
            return None

    def _downsample_to_grid(self, fine_data, coarse_points, grid_spacing):
        """Downsample fine data to coarse grid using spatial averaging"""
        try:
            coarse_samples = []
            
            for coarse_point in coarse_points:
                coarse_lat, coarse_lon = coarse_point
                
                # Define spatial window around coarse point
                lat_window = grid_spacing / 2
                lon_window = grid_spacing / 2
                
                # Find fine points within this window
                in_window = (
                    (np.abs(fine_data['latitude'] - coarse_lat) <= lat_window) &
                    (np.abs(fine_data['longitude'] - coarse_lon) <= lon_window)
                )
                
                window_data = fine_data[in_window]
                
                if len(window_data) > 0:
                    # Aggregate data in window (mean)
                    coarse_sample = {
                        'latitude': coarse_lat,
                        'longitude': coarse_lon
                    }
                    
                    # Aggregate numeric columns
                    numeric_cols = window_data.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if col not in ['latitude', 'longitude']:
                            coarse_sample[col] = window_data[col].mean()
                    
                    coarse_samples.append(coarse_sample)
            
            if coarse_samples:
                return pd.DataFrame(coarse_samples)
            else:
                return None
                
        except Exception as e:
            return None

    def _extract_coarse_features(self, fine_point, coarse_data, radius=0.75):
        """Extract features from coarse data around a fine point"""
        try:
            fine_lat, fine_lon = fine_point
            
            # Calculate distances to all coarse points
            distances = np.sqrt(
                (coarse_data['latitude'] - fine_lat)**2 + 
                (coarse_data['longitude'] - fine_lon)**2
            )
            
            # Find points within radius
            nearby_mask = distances <= radius
            nearby_coarse = coarse_data[nearby_mask]
            
            if len(nearby_coarse) == 0:
                # Use nearest point if none within radius
                nearest_idx = distances.idxmin()
                nearby_coarse = coarse_data.loc[[nearest_idx]]
            
            # Extract features
            features = {}
            
            # Basic features: closest point values
            closest_point = nearby_coarse.iloc[0]
            
            for col in nearby_coarse.columns:
                if col not in ['latitude', 'longitude'] and pd.notna(closest_point[col]):
                    features[f'coarse_{col}'] = closest_point[col]
            
            # Spatial features
            features['coarse_distance'] = distances.min()
            features['coarse_lat'] = closest_point['latitude']
            features['coarse_lon'] = closest_point['longitude']
            features['target_lat'] = fine_lat
            features['target_lon'] = fine_lon
            
            # Multi-point statistics if available
            if len(nearby_coarse) > 1:
                numeric_cols = nearby_coarse.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col not in ['latitude', 'longitude']]
                
                for col in numeric_cols:
                    if col in nearby_coarse.columns:
                        values = nearby_coarse[col].dropna()
                        if len(values) > 0:
                            features[f'coarse_{col}_mean'] = values.mean()
                            if len(values) > 1:
                                features[f'coarse_{col}_std'] = values.std()
            
            return features if features else None
            
        except Exception as e:
            return None

    def _analyze_training_data(self):
        """Analyze training data quality"""
        try:
            print(f"\n   TRAINING DATA ANALYSIS:")
            
            # Feature analysis
            coarse_features = [col for col in self.training_pairs.columns if col.startswith('coarse_')]
            fine_features = [col for col in self.training_pairs.columns if col.startswith('fine_')]
            
            print(f"      Coarse features: {len(coarse_features)}")
            print(f"      Fine targets: {len(fine_features)}")
            
            # FWI analysis
            if 'fine_fwi' in self.training_pairs.columns:
                fwi_fine = self.training_pairs['fine_fwi'].dropna()
                print(f"      Fine FWI range: [{fwi_fine.min():.2f}, {fwi_fine.max():.2f}]")
                print(f"      Fine FWI mean: {fwi_fine.mean():.2f} ± {fwi_fine.std():.2f}")
            
            if 'coarse_fwi' in self.training_pairs.columns:
                fwi_coarse = self.training_pairs['coarse_fwi'].dropna()
                print(f"      Coarse FWI range: [{fwi_coarse.min():.2f}, {fwi_coarse.max():.2f}]")
                print(f"      Coarse FWI mean: {fwi_coarse.mean():.2f} ± {fwi_coarse.std():.2f}")
            
            # Data completeness
            completeness = (1 - self.training_pairs.isnull().mean()) * 100
            print(f"      Data completeness: {completeness.mean():.1f}%")
            
        except Exception as e:
            print(f"      WARNING Analysis failed: {e}")
    
    def _approximate_merge(self, base_df, other_df, tolerance=0.05):
        """Approximate merge with spatial tolerance"""
        try:
            from scipy.spatial import cKDTree
            
            # Build spatial tree for other_df
            other_coords = other_df[['latitude', 'longitude']].values
            tree = cKDTree(other_coords)
            
            # Find nearest neighbors
            base_coords = base_df[['latitude', 'longitude']].values
            distances, indices = tree.query(base_coords)
            
            # Keep only close matches
            close_matches = distances <= tolerance
            
            if close_matches.any():
                result_df = base_df[close_matches].copy()
                matched_other = other_df.iloc[indices[close_matches]]
                
                # Add numerical columns from other dataset
                numeric_cols = matched_other.select_dtypes(include=[np.number]).columns
                exclude_cols = ['latitude', 'longitude']
                
                for col in numeric_cols:
                    if col not in exclude_cols and col not in result_df.columns:
                        result_df[col] = matched_other[col].values
                
                return result_df.reset_index(drop=True)
            else:
                return base_df
                
        except Exception as e:
            print(f"      WARNING Approximate merge failed: {e}")
            return base_df

    def save_results(self):
        """Save the processed data and analysis results"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save merged 25km data
            if self.merged_25km is not None:
                merged_file = f"merged_25km_data.csv"
                self.merged_25km.to_csv(merged_file, index=False)
                print(f"   Merged 25km data saved: {merged_file}")
            
            # Save training pairs
            if self.training_pairs is not None:
                training_file = f"training_pairs.csv"
                self.training_pairs.to_csv(training_file, index=False)
                print(f"   Training pairs saved: {training_file}")
            
            # Save summary report
            self._generate_summary_report(timestamp)
            
            return True
            
        except Exception as e:
            print(f"ERROR saving results: {e}")
            return False

    def _generate_summary_report(self, timestamp):
        """Generate a summary report of the EDA process"""
        try:
            report_file = f"eda_summary_report.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("FWI SUPER-RESOLUTION EDA SUMMARY REPORT\n")
                f.write("="*60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Data loading summary
                f.write("DATA LOADING SUMMARY\n")
                f.write("-"*30 + "\n")
                if self.temp_data is not None:
                    f.write(f"Temperature data: {self.temp_data.shape}\n")
                if self.atmospheric_data is not None:
                    f.write(f"Atmospheric data: {self.atmospheric_data.shape}\n")
                if self.fwi_data is not None:
                    f.write(f"FWI data: {self.fwi_data.shape}\n")
                f.write("\n")
                
                # Merged data summary
                if self.merged_25km is not None:
                    f.write("MERGED DATA SUMMARY\n")
                    f.write("-"*30 + "\n")
                    f.write(f"Final merged shape: {self.merged_25km.shape}\n")
                    f.write(f"Columns: {list(self.merged_25km.columns)}\n")
                    
                    if 'fwi' in self.merged_25km.columns:
                        fwi_stats = self.merged_25km['fwi'].describe()
                        f.write(f"FWI statistics:\n")
                        f.write(f"  Mean: {fwi_stats['mean']:.2f}\n")
                        f.write(f"  Std: {fwi_stats['std']:.2f}\n")
                        f.write(f"  Range: [{fwi_stats['min']:.2f}, {fwi_stats['max']:.2f}]\n")
                    f.write("\n")
                
                # Training data summary
                if self.training_pairs is not None:
                    f.write("TRAINING DATA SUMMARY\n")
                    f.write("-"*30 + "\n")
                    f.write(f"Training pairs: {len(self.training_pairs):,}\n")
                    
                    coarse_features = [col for col in self.training_pairs.columns if col.startswith('coarse_')]
                    fine_features = [col for col in self.training_pairs.columns if col.startswith('fine_')]
                    
                    f.write(f"Coarse features: {len(coarse_features)}\n")
                    f.write(f"Fine targets: {len(fine_features)}\n")
                    f.write(f"Features: {coarse_features}\n")
                    f.write(f"Targets: {fine_features}\n")
            
            print(f"   Summary report saved: {report_file}")
            
        except Exception as e:
            print(f"   WARNING Report generation failed: {e}")

def main():
    """Main function to execute the EDA process"""
    print("FWI SUPER-RESOLUTION EDA - COMPLETE WORKFLOW")
    print("="*65)
    
    # Initialize the EDA system
    eda_system = FWISuperResolutionEDA()
    
    # Step 1: Load ERA5 data
    print("\nStep 1: Loading ERA5 datasets...")
    if not eda_system.load_era5_data():
        print("ERROR Failed to load ERA5 data. Please check file paths.")
        return
    
    print("SUCCESS Data loading completed successfully")
    
    # Step 2: Preprocess and merge data
    print("\nStep 2: Preprocessing and merging datasets...")
    if not eda_system.preprocess_and_merge():
        print("ERROR Failed to preprocess and merge data")
        return
    
    print("SUCCESS Data preprocessing and merging completed successfully")
    print(f"   Final merged dataset: {eda_system.merged_25km.shape}")
    
    # Step 3: Create downsampled training data
    print("\nStep 3: Creating downsampled training data...")
    if not eda_system.create_downsampled_training_data('50km'):
        print("ERROR Failed to create training data")
        return
    
    print("SUCCESS Training data creation completed successfully")
    print(f"   Training pairs generated: {len(eda_system.training_pairs):,}")
    
    # Step 4: Save results
    print("\nStep 4: Saving results and generating reports...")
    if eda_system.save_results():
        print("SUCCESS Results saved successfully")
    else:
        print("WARNING Some issues occurred while saving results")
    
    # Final summary
    print("\nEDA PROCESS COMPLETED!")
    print("="*50)
    print("SUMMARY:")
    
    if eda_system.merged_25km is not None:
        print(f"   • Merged 25km data: {eda_system.merged_25km.shape}")
        
        # Show available variables
        met_vars = [col for col in eda_system.merged_25km.columns 
                   if col not in ['latitude', 'longitude', 'date', 'fwi']]
        print(f"   • Meteorological variables: {len(met_vars)}")
        print(f"     {met_vars}")
        
        # Show data coverage
        if 'date' in eda_system.merged_25km.columns:
            date_range = eda_system.merged_25km['date'].agg(['min', 'max'])
            unique_dates = eda_system.merged_25km['date'].nunique()
            print(f"   • Temporal coverage: {unique_dates} days ({date_range['min']} to {date_range['max']})")
        
        if 'latitude' in eda_system.merged_25km.columns:
            spatial_points = len(eda_system.merged_25km[['latitude', 'longitude']].drop_duplicates())
            print(f"   • Spatial coverage: {spatial_points:,} unique locations")
        
        if 'fwi' in eda_system.merged_25km.columns:
            fwi_count = eda_system.merged_25km['fwi'].count()
            print(f"   • FWI data points: {fwi_count:,}")
    
    if eda_system.training_pairs is not None:
        print(f"   • Training pairs: {len(eda_system.training_pairs):,}")
        
        coarse_features = [col for col in eda_system.training_pairs.columns if col.startswith('coarse_')]
        fine_features = [col for col in eda_system.training_pairs.columns if col.startswith('fine_')]
        
        print(f"   • Coarse features: {len(coarse_features)}")
        print(f"   • Fine targets: {len(fine_features)}")
    
    print("\nReady for super-resolution model training!")
    print("   Next step: Use the generated training data to train ML models")

if __name__ == "__main__":
    main()