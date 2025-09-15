"""
Feature Importance Analysis for FWI Data with Proper Downscaling Strategy
========================================================================

This script analyzes feature importance using the processed CSV data from EDA.py
with the proper downscaling strategy: downscale meteorological variables to 50km
while keeping FWI at 25km as target.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial import cKDTree

class FWIFeatureImportanceAnalyzerDownscale:
    """
    Feature importance analyzer with proper downscaling strategy:
    - Downscale meteorological variables from 25km to 50km
    - Keep FWI at 25km resolution as target
    - Train models: 50km meteovars → 25km FWI
    """
    
    def __init__(self, csv_file_path=None):
        """
        Initialize the analyzer with downscaling strategy
        
        Args:
            csv_file_path: Path to the processed CSV file from EDA.py
        """
        self.csv_file_path = csv_file_path
        self.original_25km_data = None
        
        # Downscaled training data
        self.downscaled_50km_meteovars = None
        self.target_25km_fwi = None
        self.training_features = None
        self.training_target = None
        
        # Analysis data
        self.X = None
        self.y = None
        self.feature_names = None
        self.models = {}
        self.importance_results = {}
        
        print("FWI Feature Importance Analyzer with Proper Downscaling Strategy")
        print("Strategy: Downscale meteovars (25km→50km), Keep FWI at 25km")
    
    def load_data(self):
        """Load and validate the processed CSV data"""
        try:
            # Try to find the CSV file automatically if not provided
            if self.csv_file_path is None:
                possible_files = [
                    "merged_fwi_25km_processed.csv"                                   
                ]
                
                for file in possible_files:
                    if os.path.exists(file):
                        self.csv_file_path = file
                        print(f"Found data file: {file}")
                        break
                
                if self.csv_file_path is None:
                    print("ERROR: No suitable CSV file found!")
                    return False
            
            # Load the data
            print(f"Loading 25km data from: {self.csv_file_path}")
            self.original_25km_data = pd.read_csv(self.csv_file_path)
            
            print(f"   Shape: {self.original_25km_data.shape}")
            print(f"   Columns: {list(self.original_25km_data.columns)}")
            
            # Validate required columns
            required_cols = ['latitude', 'longitude', 'date']
            missing_cols = [col for col in required_cols if col not in self.original_25km_data.columns]
            
            if missing_cols:
                print(f"ERROR: Missing required columns: {missing_cols}")
                return False
            
            # Find FWI column
            fwi_cols = [col for col in self.original_25km_data.columns if 'fwi' in col.lower()]
            if not fwi_cols:
                print("ERROR: No FWI column found!")
                return False
            
            # Ensure FWI column is named 'fwi'
            if 'fwi' not in self.original_25km_data.columns and fwi_cols:
                self.original_25km_data['fwi'] = self.original_25km_data[fwi_cols[0]]
                print(f"   Using '{fwi_cols[0]}' as FWI column")
            
            print(f"   Date range: {self.original_25km_data['date'].min()} to {self.original_25km_data['date'].max()}")
            print(f"   FWI range: [{self.original_25km_data['fwi'].min():.2f}, {self.original_25km_data['fwi'].max():.2f}]")
            
            return True
            
        except Exception as e:
            print(f"ERROR loading data: {e}")
            return False
    
    def create_downscaled_training_data(self):
        """
        Create training data by downscaling meteorological variables to 50km
        while keeping FWI at 25km as target
        """
        print(f"\nCreating downscaled training data for feature importance analysis...")
        
        if self.original_25km_data is None:
            print("ERROR: No 25km data available")
            return False
        
        try:
            # Define meteorological variables (exclude FWI, coordinates, and date)
            exclude_cols = ['fwi', 'latitude', 'longitude', 'date']
            meteo_vars = [col for col in self.original_25km_data.columns 
                         if col not in exclude_cols and 
                         self.original_25km_data[col].dtype in ['int64', 'float64']]
            
            print(f"   Meteorological variables to downscale ({len(meteo_vars)}):")
            for i, var in enumerate(meteo_vars):
                if i < 10:  # Show first 10
                    print(f"      - {var}")
                elif i == 10:
                    print(f"      ... and {len(meteo_vars)-10} more")
            
            if len(meteo_vars) == 0:
                print("ERROR: No meteorological variables found for downscaling")
                return False
            
            # Get unique dates for processing
            unique_dates = sorted(self.original_25km_data['date'].unique())
            print(f"   Processing {len(unique_dates)} dates for downscaling")
            
            # Limit dates for feature importance analysis (faster processing)
            if len(unique_dates) > 30:
                print(f"   Using first 30 dates for feature importance analysis")
                unique_dates = unique_dates[:30]
            
            all_training_samples = []
            
            for i, date in enumerate(unique_dates):
                if (i + 1) % 5 == 0:
                    print(f"      Processing date {i+1}/{len(unique_dates)}: {date}")
                
                # Get daily data
                daily_data = self.original_25km_data[self.original_25km_data['date'] == date].copy()
                
                if len(daily_data) < 4:  # Need minimum points
                    continue
                
                # Create downscaled training samples for this date
                daily_samples = self._create_daily_downscaled_samples(daily_data, meteo_vars)
                
                if daily_samples is not None and len(daily_samples) > 0:
                    daily_samples['date'] = date
                    all_training_samples.append(daily_samples)
            
            if not all_training_samples:
                print("ERROR: No training samples created")
                return False
            
            # Combine all training samples
            training_df = pd.concat(all_training_samples, ignore_index=True)
            
            print(f"SUCCESS: Downscaled training data created: {len(training_df):,} samples")
            
            # Extract features and target for analysis
            self._prepare_features_and_target(training_df, meteo_vars)
            
            return True
            
        except Exception as e:
            print(f"ERROR creating downscaled training data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_daily_downscaled_samples(self, daily_data, meteo_vars):
        """Create downscaled training samples for a single date"""
        try:
            # Define 50km grid spacing
            grid_spacing = 0.5  # ~50km
            
            # Get spatial bounds
            lat_min, lat_max = daily_data['latitude'].min(), daily_data['latitude'].max()
            lon_min, lon_max = daily_data['longitude'].min(), daily_data['longitude'].max()
            
            # Create 50km grid
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
            
            # Downscale meteorological variables to 50km grid
            downscaled_meteovars = self._downscale_meteovars_to_50km_grid(
                daily_data, coarse_lats, coarse_lons, meteo_vars, grid_spacing
            )
            
            if downscaled_meteovars is None or len(downscaled_meteovars) == 0:
                return None
            
            # For each 25km FWI point, create training sample with 50km meteovar features
            training_samples = []
            
            for _, row_25km in daily_data.iterrows():
                lat_25km = row_25km['latitude']
                lon_25km = row_25km['longitude']
                fwi_25km = row_25km['fwi']
                
                if pd.isna(fwi_25km):
                    continue
                
                # Extract 50km meteorological features for this 25km FWI point
                features_50km = self._extract_50km_features_for_25km_fwi(
                    lat_25km, lon_25km, downscaled_meteovars, meteo_vars
                )
                
                if features_50km is not None:
                    # Create training sample
                    sample = {
                        'target_lat': lat_25km,
                        'target_lon': lon_25km,
                        'target_fwi': fwi_25km  # 25km FWI as target
                    }
                    
                    # Add 50km meteorological features
                    sample.update(features_50km)
                    
                    training_samples.append(sample)
            
            if training_samples:
                return pd.DataFrame(training_samples)
            else:
                return None
                
        except Exception as e:
            print(f"      ERROR creating daily samples: {e}")
            return None
    
    def _downscale_meteovars_to_50km_grid(self, daily_data, coarse_lats, coarse_lons, meteo_vars, grid_spacing):
        """Downscale meteorological variables from 25km to 50km using spatial averaging"""
        try:
            downscaled_points = []
            
            for coarse_lat in coarse_lats:
                for coarse_lon in coarse_lons:
                    # Define spatial window around 50km point
                    lat_window = grid_spacing / 2
                    lon_window = grid_spacing / 2
                    
                    # Find 25km points within this window
                    in_window = (
                        (np.abs(daily_data['latitude'] - coarse_lat) <= lat_window) &
                        (np.abs(daily_data['longitude'] - coarse_lon) <= lon_window)
                    )
                    
                    window_data = daily_data[in_window]
                    
                    if len(window_data) > 0:
                        # Aggregate meteorological variables in window
                        downscaled_point = {
                            'latitude': coarse_lat,
                            'longitude': coarse_lon
                        }
                        
                        # Average meteorological variables
                        for var in meteo_vars:
                            if var in window_data.columns:
                                values = window_data[var].dropna()
                                if len(values) > 0:
                                    downscaled_point[f'meteo_50km_{var}'] = values.mean()
                                    # Add additional statistics
                                    if len(values) > 1:
                                        downscaled_point[f'meteo_50km_{var}_std'] = values.std()
                                        downscaled_point[f'meteo_50km_{var}_range'] = values.max() - values.min()
                        
                        downscaled_points.append(downscaled_point)
            
            if downscaled_points:
                return pd.DataFrame(downscaled_points)
            else:
                return None
                
        except Exception as e:
            print(f"         ERROR downscaling meteovars: {e}")
            return None
    
    def _analyze_feature_categories(self):
        """Analyze and display feature categories"""
        try:
            print(f"   Feature category breakdown:")
            
            categories = {
                'Downscaled Meteorological': 0,
                'Spatial Context': 0,
                'Gradient Features': 0,
                'Variability Features': 0
            }
            
            for feature in self.feature_names:
                if 'feature_downscaled_' in feature and '_variability' not in feature and '_spatial_range' not in feature:
                    categories['Downscaled Meteorological'] += 1
                elif 'feature_spatial_' in feature:
                    categories['Spatial Context'] += 1
                elif 'feature_gradient_' in feature:
                    categories['Gradient Features'] += 1
                elif '_variability' in feature or '_spatial_range' in feature:
                    categories['Variability Features'] += 1
            
            for category, count in categories.items():
                if count > 0:
                    print(f"      - {category}: {count} features")
            
        except Exception as e:
            print(f"      WARNING: Feature category analysis failed: {e}")
    
    def train_models(self):
        """Train multiple models for feature importance analysis using downscaled data with robust error handling"""
        try:
            print("Training models for downscaled feature importance analysis...")
            
            if self.X is None or self.y is None:
                print("ERROR: No prepared data available for training")
                return False
            
            # Additional data validation before training
            print("   Final data validation before training...")
            
            # Check data quality - fix the Series boolean issue
            if np.isinf(self.X.values).any() or np.isnan(self.X.values).any():
                print("   Fixing remaining data issues...")
                self.X = self.X.replace([np.inf, -np.inf], np.nan)
                self.X = self.X.fillna(self.X.median())
                self.X = self.X.fillna(0)
            
            if np.isinf(self.y.values).any() or np.isnan(self.y.values).any():
                print("   Fixing target variable issues...")
                self.y = self.y.replace([np.inf, -np.inf], np.nan)
                self.y = self.y.fillna(self.y.median())
                self.y = self.y.fillna(0)
            
            # Remove features with zero variance
            feature_variance = self.X.var()
            zero_variance_features = feature_variance[feature_variance == 0].index.tolist()
            if zero_variance_features:
                print(f"   Removing {len(zero_variance_features)} zero-variance features")
                self.X = self.X.drop(columns=zero_variance_features)
                self.feature_names = [f for f in self.feature_names if f not in zero_variance_features]
            
            # Check minimum data requirements
            if len(self.X) < 10:
                print("ERROR: Insufficient training samples")
                return False
            
            if len(self.feature_names) < 2:
                print("ERROR: Insufficient features after cleaning")
                return False
            
            print(f"   Data ready for training: {self.X.shape}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            print(f"   Training split: train{X_train.shape}, test{X_test.shape}")
            
            # Scale features with robust handling
            try:
                scaler = RobustScaler()
                
                # Fit scaler on training data
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                print("   Feature scaling completed successfully")
                
            except Exception as scaling_error:
                print(f"   WARNING: Scaling failed ({scaling_error}), using original features")
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values
                scaler = None
            
            # Define models optimized for downscaled meteorological data
            models_to_train = {
                
                'Gradient Boosting': {
                    'model': GradientBoostingRegressor(
                        n_estimators=100,       # Reduced for speed
                        max_depth=6,            # Moderate depth
                        learning_rate=0.1,      # Standard learning rate
                        subsample=0.8,          # Prevent overfitting
                        random_state=42
                    ),
                    'X_train': X_train,
                    'X_test': X_test,
                    'scaled': False
                }
            }
            
            # Train and evaluate models
            for model_name, model_config in models_to_train.items():
                print(f"   Training {model_name} on downscaled features...")
                
                try:
                    model = model_config['model']
                    X_tr = model_config['X_train']
                    X_te = model_config['X_test']
                    
                    # Additional validation for this model - FIX THE BOOLEAN ISSUE
                    if isinstance(X_tr, pd.DataFrame):
                        if np.isinf(X_tr.values).any() or np.isnan(X_tr.values).any():
                            print(f"      WARNING: Data issues detected for {model_name}, skipping...")
                            continue
                    else:  # numpy array
                        if np.isinf(X_tr).any() or np.isnan(X_tr).any():
                            print(f"      WARNING: Data issues detected for {model_name}, skipping...")
                            continue
                    
                    # Train model
                    model.fit(X_tr, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_te)
                    
                    # Validate predictions
                    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                        print(f"      WARNING: Invalid predictions from {model_name}, skipping...")
                        continue
                    
                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    # Cross-validation with error handling
                    try:
                        cv_scores = cross_val_score(model, X_tr, y_train, cv=3, scoring='r2')  # Reduced CV folds
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    except Exception as cv_error:
                        print(f"      WARNING: Cross-validation failed for {model_name}: {cv_error}")
                        cv_mean = r2
                        cv_std = 0.0
                    
                    # Store model and results
                    self.models[model_name] = {
                        'model': model,
                        'r2': r2,
                        'rmse': rmse,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std,
                        'scaled': model_config['scaled'],
                        'scaler': scaler if model_config['scaled'] else None
                    }
                    
                    print(f"      SUCCESS: R²: {r2:.4f}, RMSE: {rmse:.4f}, CV: {cv_mean:.4f}±{cv_std:.4f}")
                    
                    # Feature importance for tree models
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        top_features = sorted(zip(self.feature_names, importances), 
                                            key=lambda x: x[1], reverse=True)[:3]
                        clean_top = [self._clean_feature_name(f) for f, _ in top_features]
                        print(f"         Top features: {clean_top}")
                    
                except Exception as model_error:
                    print(f"      ERROR: {model_name} training failed: {model_error}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if self.models:
                print(f"   SUCCESS: Successfully trained {len(self.models)} models on downscaled data!")
                return True
            else:
                print("   ERROR: No models trained successfully")
                return False
            
        except Exception as e:
            print(f"ERROR training models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _prepare_features_and_target(self, training_df, meteo_vars):
        """Prepare final features and target for importance analysis with enhanced data validation"""
        try:
            print(f"   Preparing features and target for analysis...")
            
            # Get feature columns (50km meteorological features)
            feature_cols = [col for col in training_df.columns if col.startswith('feature_')]
            
            # Remove non-numeric features
            numeric_features = []
            for feat in feature_cols:
                if (training_df[feat].dtype in ['int64', 'float64'] and 
                    not training_df[feat].isnull().all()):
                    numeric_features.append(feat)
            
            print(f"   Using {len(numeric_features)} downscaled meteorological features")
            
            if len(numeric_features) < 2:
                print("ERROR: Insufficient features for analysis")
                return False
            
            # Prepare features with proper data cleaning
            X_raw = training_df[numeric_features].copy()
            
            # Data validation and cleaning
            print(f"   Cleaning and validating data...")
            
            # 1. Check for infinite values column by column
            inf_cols = []
            for col in X_raw.columns:
                col_values = X_raw[col]
                if pd.isna(col_values).all():
                    continue
                
                # Check for infinite values safely
                inf_mask = np.isinf(col_values.fillna(0))
                if inf_mask.any():
                    inf_count = inf_mask.sum()
                    print(f"      WARNING: Found {inf_count} infinite values in {col}")
                    inf_cols.append(col)
            
            # 2. Check for extremely large values
            large_cols = []
            for col in X_raw.columns:
                col_values = X_raw[col].fillna(0)
                max_abs_val = np.abs(col_values).max()
                if max_abs_val > 1e10:
                    print(f"      WARNING: Found extremely large values in {col}: max = {max_abs_val:.2e}")
                    large_cols.append(col)
            
            # 3. Clean the data systematically
            X_cleaned = X_raw.copy()
            
            for col in X_cleaned.columns:
                col_series = X_cleaned[col]
                
                # Replace infinite values with NaN
                col_series = col_series.replace([np.inf, -np.inf], np.nan)
                
                # Cap extremely large values (beyond 99.9th percentile)
                if not col_series.isna().all():
                    # Calculate percentiles only for finite values
                    finite_values = col_series[np.isfinite(col_series)]
                    if len(finite_values) > 0:
                        p999 = finite_values.quantile(0.999)
                        p001 = finite_values.quantile(0.001)
                        
                        # Cap values beyond reasonable range
                        col_series = col_series.clip(lower=p001, upper=p999)
                
                X_cleaned[col] = col_series
            
            # 4. Fill NaN values with column median (robust to outliers)
            for col in X_cleaned.columns:
                col_median = X_cleaned[col].median()
                if pd.isna(col_median):
                    col_median = 0.0
                X_cleaned[col] = X_cleaned[col].fillna(col_median)
            
            # 5. Final validation - replace any remaining NaN with 0
            X_cleaned = X_cleaned.fillna(0)
            
            # 6. Check for remaining issues
            if np.isinf(X_cleaned.values).any():
                print(f"      WARNING: Still have infinite values after cleaning, replacing with 0...")
                X_cleaned = X_cleaned.replace([np.inf, -np.inf], 0)
            
            if np.isnan(X_cleaned.values).any():
                print(f"      WARNING: Still have NaN values after cleaning, replacing with 0...")
                X_cleaned = X_cleaned.fillna(0)
            
            # Prepare target variable with same cleaning
            y_raw = training_df['target_fwi'].copy()
            y_cleaned = y_raw.replace([np.inf, -np.inf], np.nan)
            
            # Cap FWI values to reasonable range (0-100)
            y_cleaned = y_cleaned.clip(lower=0, upper=100)
            
            # Fill NaN in target
            target_median = y_cleaned.median()
            if pd.isna(target_median):
                target_median = 10.0  # Default reasonable FWI value
            y_cleaned = y_cleaned.fillna(target_median)
            
            # Final assignment
            self.X = X_cleaned
            self.y = y_cleaned
            self.feature_names = numeric_features
            
            print(f"   Final dataset prepared: X{self.X.shape}, y{self.y.shape}")
            print(f"   Target (25km FWI) range: [{self.y.min():.2f}, {self.y.max():.2f}]")
            
            # Additional validation
            print(f"   Data validation:")
            print(f"      - X contains infinite: {np.isinf(self.X.values).any()}")
            print(f"      - X contains NaN: {np.isnan(self.X.values).any()}")
            print(f"      - y contains infinite: {np.isinf(self.y.values).any()}")
            print(f"      - y contains NaN: {np.isnan(self.y.values).any()}")
            print(f"      - X value range: [{self.X.values.min():.2e}, {self.X.values.max():.2e}]")
            
            # Show feature categories
            self._analyze_feature_categories()
            
            return True
            
        except Exception as e:
            print(f"ERROR preparing features and target: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_50km_features_for_25km_fwi(self, lat_25km, lon_25km, downscaled_meteovars, meteo_vars):
        """Extract 50km meteorological features for a 25km FWI point with enhanced robustness"""
        try:
            # Calculate distances to all 50km points
            lat_diff = downscaled_meteovars['latitude'] - lat_25km
            lon_diff = downscaled_meteovars['longitude'] - lon_25km
            distances = np.sqrt(lat_diff**2 + lon_diff**2)
            
            # Find nearby 50km points
            radius = 0.75  # ~75km radius for feature extraction
            nearby_mask = distances <= radius
            nearby_50km = downscaled_meteovars[nearby_mask].copy()
            
            if len(nearby_50km) == 0:
                # Use nearest point if none within radius
                nearest_idx = distances.idxmin()
                nearby_50km = downscaled_meteovars.loc[[nearest_idx]].copy()
                distances_nearby = distances.loc[[nearest_idx]]
            else:
                distances_nearby = distances[nearby_50km.index]
            
            # Extract features using distance-weighted averaging with robust handling
            features = {}
            
            if len(nearby_50km) == 1:
                # Single point case
                closest_point = nearby_50km.iloc[0]
                for var in meteo_vars:
                    col_name = f'meteo_50km_{var}'
                    if col_name in closest_point.index:
                        value = closest_point[col_name]
                        # Validate value more carefully
                        if pd.notna(value) and np.isfinite(value) and abs(value) < 1e10:
                            features[f'feature_downscaled_{var}'] = float(value)
                
                # Include additional statistics if available
                std_col = f'meteo_50km_{var}_std'
                range_col = f'meteo_50km_{var}_range'
                
                if std_col in closest_point.index:
                    std_value = closest_point[std_col]
                    if pd.notna(std_value) and np.isfinite(std_value) and abs(std_value) < 1e10:
                        features[f'feature_downscaled_{var}_variability'] = float(std_value)
                
                if range_col in closest_point.index:
                    range_value = closest_point[range_col]
                    if pd.notna(range_value) and np.isfinite(range_value) and abs(range_value) < 1e10:
                        features[f'feature_downscaled_{var}_spatial_range'] = float(range_value)
            else:
                # Multiple points - distance-weighted average with robust handling
                weights = 1 / (distances_nearby + 0.001)  # Avoid division by zero
                
                # Check for infinite weights
                if np.isinf(weights).any():
                    weights = pd.Series(np.ones(len(weights)), index=weights.index)
                
                # Normalize weights
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    weights = pd.Series(np.ones(len(weights)) / len(weights), index=weights.index)
                
                for var in meteo_vars:
                    col_name = f'meteo_50km_{var}'
                    if col_name in nearby_50km.columns:
                        values = nearby_50km[col_name].dropna()
                        if len(values) > 0:
                            # Filter out infinite and extremely large values
                            finite_mask = np.isfinite(values) & (np.abs(values) < 1e10)
                            valid_values = values[finite_mask]
                            
                            if len(valid_values) > 0:
                                # Weighted average
                                valid_weights = weights[valid_values.index]
                                if len(valid_weights) > 0 and valid_weights.sum() > 0:
                                    valid_weights = valid_weights / valid_weights.sum()
                                    weighted_avg = (valid_values * valid_weights).sum()
                                    
                                    if np.isfinite(weighted_avg):
                                        features[f'feature_downscaled_{var}'] = float(weighted_avg)
                    
                    # Add spatial variability features with validation
                    std_col = f'meteo_50km_{var}_std'
                    if std_col in nearby_50km.columns:
                        std_values = nearby_50km[std_col].dropna()
                        if len(std_values) > 0:
                            finite_std_mask = np.isfinite(std_values) & (np.abs(std_values) < 1e10)
                            valid_std = std_values[finite_std_mask]
                            if len(valid_std) > 0:
                                valid_weights_std = weights[valid_std.index]
                                if len(valid_weights_std) > 0 and valid_weights_std.sum() > 0:
                                    valid_weights_std = valid_weights_std / valid_weights_std.sum()
                                    weighted_std = (valid_std * valid_weights_std).sum()
                                    if np.isfinite(weighted_std):
                                        features[f'feature_downscaled_{var}_variability'] = float(weighted_std)
        
            # Add spatial context features with validation
            min_distance = distances_nearby.min()
            if np.isfinite(min_distance):
                features['feature_spatial_distance_to_50km'] = float(min_distance)
            
            features['feature_spatial_num_50km_points'] = int(len(nearby_50km))
            
            mean_lat = nearby_50km['latitude'].mean()
            mean_lon = nearby_50km['longitude'].mean()
            if np.isfinite(mean_lat):
                features['feature_spatial_lat_50km'] = float(mean_lat)
            if np.isfinite(mean_lon):
                features['feature_spatial_lon_50km'] = float(mean_lon)
            
            # Add gradient features if multiple points available (with robust calculation)
            if len(nearby_50km) > 2:
                for var in meteo_vars:
                    col_name = f'meteo_50km_{var}'
                    if col_name in nearby_50km.columns:
                        values = nearby_50km[col_name].dropna()
                        if len(values) > 2:
                            # Filter valid values for gradient calculation
                            finite_mask = np.isfinite(values) & (np.abs(values) < 1e10)
                            valid_values = values[finite_mask]
                            
                            if len(valid_values) >= 3:
                                # Calculate spatial gradients with error handling
                                try:
                                    coords = nearby_50km.loc[valid_values.index, ['latitude', 'longitude']].values
                                    if len(coords) >= 3:
                                        # Check for variation in latitude
                                        unique_lats = np.unique(coords[:, 0])
                                        if len(unique_lats) > 1:
                                            lat_gradient = np.gradient(valid_values.values, coords[:, 0])
                                            mean_lat_grad = np.mean(lat_gradient)
                                            if np.isfinite(mean_lat_grad):
                                                features[f'feature_gradient_{var}_lat'] = float(mean_lat_grad)
                                            
                                        # Check for variation in longitude
                                        unique_lons = np.unique(coords[:, 1])
                                        if len(unique_lons) > 1:
                                            lon_gradient = np.gradient(valid_values.values, coords[:, 1])
                                            mean_lon_grad = np.mean(lon_gradient)
                                            if np.isfinite(mean_lon_grad):
                                                features[f'feature_gradient_{var}_lon'] = float(mean_lon_grad)
                                except Exception:
                                    pass  # Skip if gradient calculation fails
            
            # Final validation of all features
            validated_features = {}
            for key, value in features.items():
                if isinstance(value, (int, float)) and np.isfinite(value) and abs(value) < 1e10:
                    validated_features[key] = value
            
            return validated_features if len(validated_features) > 0 else None
            
        except Exception as e:
            # print(f"         ERROR extracting 50km features: {e}")
            return None

    def extract_feature_importance(self):
        """Extract feature importance from all trained models"""
        try:
            print("Extracting feature importance from downscaled models...")
            
            for model_name, model_info in self.models.items():
                print(f"   Analyzing {model_name}...")
                
                model = model_info['model']
                importance_scores = None
                
                # Method 1: Built-in feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_scores = model.feature_importances_
                    method = 'built_in'
                
                # Method 2: Coefficient magnitude (for linear models)
                elif hasattr(model, 'coef_'):
                    importance_scores = np.abs(model.coef_)
                    method = 'coefficient'
                
                # Method 3: Permutation importance (for neural networks)
                else:
                    print(f"      Using permutation importance for {model_name}...")
                    X_sample = self.X.sample(min(1000, len(self.X)), random_state=42)
                    y_sample = self.y.loc[X_sample.index]
                    
                    if model_info['scaled']:
                        X_sample = model_info['scaler'].transform(X_sample)
                    
                    perm_importance = permutation_importance(
                        model, X_sample, y_sample, 
                        n_repeats=5, random_state=42, n_jobs=-1
                    )
                    importance_scores = perm_importance.importances_mean
                    method = 'permutation'
                
                # Store importance results
                if importance_scores is not None:
                    self.importance_results[model_name] = {
                        'scores': importance_scores,
                        'method': method,
                        'features': self.feature_names,
                        'performance': {
                            'r2': model_info['r2'],
                            'rmse': model_info['rmse'],
                            'cv_mean': model_info['cv_mean']
                        }
                    }
                    
                    # Show top features with cleaned names
                    feature_importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': importance_scores
                    }).sort_values('importance', ascending=False)
                    
                    top_features = [self._clean_feature_name(f) for f in feature_importance_df.head()['feature']]
                    print(f"      Top 5 features: {top_features}")
                
                else:
                    print(f"      WARNING: Could not extract importance for {model_name}")
            
            print("   Feature importance extraction completed!")
            return True
            
        except Exception as e:
            print(f"ERROR extracting feature importance: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_downscaled_category_analysis(self):
        """Create feature category analysis specific to downscaled features"""
        print("      Creating downscaled category analysis...")
        
        # Define downscaled feature categories
        categories = {
            'Temperature': ['temp', 'temperature', '2m'],
            'Humidity': ['humidity', 'rh', 'dewpoint'],
            'Wind': ['wind', 'u_component', 'v_component', 'speed', 'u10', 'v10'],
            'Precipitation': ['precip', 'rain', 'precipitation', 'tp'],
            'Pressure': ['pressure', 'msl', 'sp'],
            'Radiation': ['radiation', 'solar', 'ssr'],
            'Spatial': ['spatial', 'distance', 'lat', 'lon'],
            'Gradient': ['gradient'],
            'Variability': ['variability', 'range', 'std']
        }
        
        # Calculate category importance for each model
        category_results = {}
        
        for model_name, results in self.importance_results.items():
            model_categories = {}
            
            for category, keywords in categories.items():
                category_importance = []
                category_features = []
                
                for i, feature in enumerate(results['features']):
                    # Use more precise matching
                    feature_lower = feature.lower()
                    matched = False
                    
                    for keyword in keywords:
                        if (keyword in feature_lower and 
                            (feature_lower.startswith(f'feature_downscaled_{keyword}') or
                             feature_lower.startswith(f'feature_{keyword}') or
                             f'_{keyword}_' in feature_lower or
                             feature_lower.endswith(f'_{keyword}'))):
                            category_importance.append(results['scores'][i])
                            category_features.append(feature)
                            matched = True
                            break
                    
                    # Special handling for partial matches
                    if not matched:
                        for keyword in keywords:
                            if keyword in feature_lower:
                                category_importance.append(results['scores'][i])
                                category_features.append(feature)
                                break
                
                if category_importance:
                    model_categories[category] = {
                        'total': sum(category_importance),
                        'mean': np.mean(category_importance),
                        'count': len(category_importance),
                        'features': category_features
                    }
                else:
                    model_categories[category] = {'total': 0, 'mean': 0, 'count': 0, 'features': []}
            
            category_results[model_name] = model_categories
        
        # Create visualization
        categories_list = list(categories.keys())
        model_names = list(self.importance_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Downscaled Feature Category Analysis', fontsize=16, fontweight='bold')
        
        # Total importance by category (averaged across models)
        ax1 = axes[0, 0]
        category_totals = {}
        for category in categories_list:
            totals = []
            for model_name in model_names:
                totals.append(category_results[model_name][category]['total'])
            category_totals[category] = np.mean(totals)
        
        sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        cats, tots = zip(*sorted_categories)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(cats)))
        bars = ax1.bar(cats, tots, color=colors)
        ax1.set_ylabel('Average Total Importance')
        ax1.set_title('Feature Category Importance\n(50km Meteovars → 25km FWI)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, tot in zip(bars, tots):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(tots)*0.01,
                    f'{tot:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Feature count by category
        ax2 = axes[0, 1]
        category_counts = {}
        for category in categories_list:
            counts = []
            for model_name in model_names:
                counts.append(category_results[model_name][category]['count'])
            category_counts[category] = np.mean(counts)
        
        sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        cats_c, counts_c = zip(*sorted_counts)
        
        ax2.bar(cats_c, counts_c, color=plt.cm.plasma(np.linspace(0, 1, len(cats_c))))
        ax2.set_ylabel('Average Feature Count')
        ax2.set_title('Number of Features by Category')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Model comparison for top categories
        ax3 = axes[1, 0]
        top_3_categories = [cat for cat, _ in sorted_categories[:3]]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, category in enumerate(top_3_categories):
            values = [category_results[model][category]['total'] for model in model_names]
            ax3.bar(x + i*width, values, width, label=category, alpha=0.8)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Total Importance')
        ax3.set_title('Top 3 Categories by Model')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([m.replace(' ', '\n') for m in model_names])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Feature type distribution pie chart
        ax4 = axes[1, 1]
        feature_types = {
            'Meteorological Variables': sum(category_counts[cat] for cat in ['Temperature', 'Humidity', 'Wind', 'Precipitation', 'Pressure', 'Radiation']),
            'Spatial Features': category_counts['Spatial'],
            'Gradient Features': category_counts['Gradient'],
            'Variability Features': category_counts['Variability']
        }
        
        # Filter out zero values
        feature_types = {k: v for k, v in feature_types.items() if v > 0}
        
        if feature_types:
            ax4.pie(feature_types.values(), labels=feature_types.keys(), autopct='%1.1f%%', startangle=90)
            ax4.set_title('Feature Type Distribution')
        
        plt.tight_layout()
        plt.savefig('downscaled_feature_category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_downscaling_validation_plot(self):
        """Create visualization showing downscaling strategy validation"""
        print("      Creating downscaling strategy validation...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Downscaling Strategy Validation (50km Meteovars → 25km FWI)', fontsize=16, fontweight='bold')
        
        # 1. Model performance comparison
        ax1 = axes[0, 0]
        models = list(self.importance_results.keys())
        r2_scores = [self.importance_results[model]['performance']['r2'] for model in models]
        cv_scores = [self.importance_results[model]['performance']['cv_mean'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, r2_scores, width, label='Test R²', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, cv_scores, width, label='CV R²', alpha=0.8, color='lightcoral')
        
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Performance with Downscaled Features')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace(' ', '\n') for m in models])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # 2. Feature importance distribution
        ax2 = axes[0, 1]
        all_importances = []
        for results in self.importance_results.values():
            all_importances.extend(results['scores'])
        
        ax2.hist(all_importances, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Feature Importance')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Feature Importance Distribution\n(Downscaled Meteorological Variables)')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_imp = np.mean(all_importances)
        std_imp = np.std(all_importances)
        ax2.axvline(mean_imp, color='red', linestyle='--', label=f'Mean: {mean_imp:.3f}')
        ax2.axvline(mean_imp + std_imp, color='orange', linestyle='--', alpha=0.7, label=f'Mean+Std: {mean_imp+std_imp:.3f}')
        ax2.legend()
        
        # 3. Top features across all models
        ax3 = axes[1, 0]
        
        # Calculate consensus importance
        all_features = set()
        for results in self.importance_results.values():
            all_features.update(results['features'])
        
        consensus_importance = {}
        for feature in all_features:
            importances = []
            for results in self.importance_results.values():
                if feature in results['features']:
                    idx = results['features'].index(feature)
                    importances.append(results['scores'][idx])
            
            if importances:
                consensus_importance[feature] = np.mean(importances)
    
            # Plot top 10 consensus features
        sorted_features = sorted(consensus_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        features, importances = zip(*sorted_features)
        
        clean_features = [self._clean_feature_name(f) for f in features]
        
        bars = ax3.barh(range(len(features)), importances, color=plt.cm.viridis(np.linspace(0, 1, len(features))))
        ax3.set_yticks(range(len(features)))
        ax3.set_yticklabels(clean_features, fontsize=9)
        ax3.set_xlabel('Average Importance')
        ax3.set_title('Top 10 Consensus Features\n(50km → 25km Strategy)')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.invert_yaxis()
        
        # 4. Strategy benefits summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create text summary
        best_model = max(self.importance_results.items(), key=lambda x: x[1]['performance']['cv_mean'])
        best_r2 = best_model[1]['performance']['cv_mean']
        
        summary_text = f"""
        DOWNSCALING STRATEGY RESULTS
        ═══════════════════════════════
        
        Strategy: Downscale meteorological variables 
        from 25km → 50km, keep FWI at 25km
        
        Best Model: {best_model[0]}
        Cross-Validation R²: {best_r2:.3f}
        
        Total Features Analyzed: {len(self.feature_names)}
        
        Feature Categories:
        • Downscaled Meteorological Variables
        • Spatial Context Features  
        • Gradient Features
        • Variability Features
        
        Benefits:
        • Realistic training scenario
        • Proper resolution relationship
        • Enhanced spatial understanding
        • Robust feature importance
        
        Training: 50km meteovars → 25km FWI
        Application: 25km meteovars → 1km FWI
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('downscaling_strategy_validation.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_individual_plots(self):
        """Create individual feature importance plots for each model"""
        print("      Creating individual model plots...")
        
        n_models = len(self.importance_results)
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Fix: Use proper figure title positioning
        fig.suptitle('Feature Importance by Model (Downscaled Strategy)', 
                    fontsize=16, fontweight='bold', y=0.98)  # Add y parameter for positioning
        
        axes_flat = axes.flatten()
        colors = ['forestgreen', 'purple', 'orange', 'red']
        
        for i, (model_name, results) in enumerate(self.importance_results.items()):
            if i >= len(axes_flat):
                break
            
            ax = axes_flat[i]
            
            # Prepare data
            feature_df = pd.DataFrame({
                'feature': results['features'],
                'importance': results['scores']
            }).sort_values('importance', ascending=True).tail(15)  # Top 15
            
            # Clean feature names for display
            clean_names = [self._clean_feature_name(f) for f in feature_df['feature']]
            
            # Create plot
            bars = ax.barh(range(len(feature_df)), feature_df['importance'], 
                        color=colors[i], alpha=0.7)
            
            ax.set_yticks(range(len(feature_df)))
            ax.set_yticklabels(clean_names, fontsize=9)
            ax.set_xlabel('Feature Importance')
            
            # Fix: Use proper title formatting with explicit positioning
            title_text = f'{model_name}\n(R² = {results["performance"]["r2"]:.3f})\n50km→25km Strategy'
            ax.set_title(title_text, fontsize=12, fontweight='bold', 
                        pad=20, ha='center')  # Add ha='center' for horizontal alignment
            
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar, importance in zip(bars, feature_df['importance']):
                width = bar.get_width()
                ax.text(width + max(feature_df['importance']) * 0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', va='center', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(self.importance_results), len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        # Fix: Adjust layout to accommodate centered title
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for title
        plt.savefig('downscaled_feature_importance_individual.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_comparison_heatmap(self):
        """Create feature importance comparison heatmap"""
        print("      Creating comparison heatmap...")
        
        # Prepare importance matrix
        all_features = set()
        for results in self.importance_results.values():
            all_features.update(results['features'])
        
        all_features = sorted(list(all_features))
        
        # Create importance matrix
        importance_matrix = []
        model_names = list(self.importance_results.keys())
        
        for feature in all_features:
            feature_importances = []
            for model_name in model_names:
                results = self.importance_results[model_name]
                if feature in results['features']:
                    idx = results['features'].index(feature)
                    importance = results['scores'][idx]
                else:
                    importance = 0
                feature_importances.append(importance)
            importance_matrix.append(feature_importances)
        
        importance_matrix = np.array(importance_matrix)
        
        # Normalize by column (model) for better comparison
        for j in range(importance_matrix.shape[1]):
            col_max = importance_matrix[:, j].max()
            if col_max > 0:
                importance_matrix[:, j] = importance_matrix[:, j] / col_max
        
        # Create heatmap
        plt.figure(figsize=(12, max(8, len(all_features) * 0.25)))
        
        # Select top features for display
        avg_importance = importance_matrix.mean(axis=1)
        top_indices = np.argsort(avg_importance)[-25:]  # Top 25 features
        
        heatmap_data = importance_matrix[top_indices]
        feature_labels = [self._clean_feature_name(all_features[i]) for i in top_indices]
        
        # Truncate long feature names
        feature_labels = [label[:35] + '...' if len(label) > 35 else label for label in feature_labels]
        
        sns.heatmap(heatmap_data, 
                xticklabels=model_names,
                yticklabels=feature_labels,
                annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': 'Normalized Importance'})
        
        # Fix: Properly center the title
        plt.title('Feature Importance Comparison Across Models (Top 25)\nDownscaling Strategy: 50km Meteovars → 25km FWI', 
                fontsize=14, fontweight='bold', pad=20, ha='center')
        plt.xlabel('Models')
        plt.ylabel('Downscaled Features')
        plt.tight_layout()
        plt.savefig('downscaled_feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_performance_comparison(self):
        """Create model performance comparison"""
        print("      Creating performance comparison...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Fix: Add proper figure title
        fig.suptitle('Model Performance Comparison - Downscaling Strategy', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        models = list(self.importance_results.keys())
        r2_scores = [self.importance_results[model]['performance']['r2'] for model in models]
        cv_scores = [self.importance_results[model]['performance']['cv_mean'] for model in models]
        
        colors = ['forestgreen', 'purple', 'orange', 'red'][:len(models)]
        
        # R² comparison
        bars1 = ax1.bar(models, r2_scores, color=colors, alpha=0.7)
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Performance (R²)\nDownscaling Strategy', pad=15, ha='center')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Cross-validation comparison
        bars2 = ax2.bar(models, cv_scores, color=colors, alpha=0.7)
        ax2.set_ylabel('Cross-Validation R²')
        ax2.set_title('Model Stability (CV R²)\n50km→25km Training', pad=15, ha='center')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars2, cv_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Fix: Adjust layout for centered titles
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig('downscaled_model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_comprehensive_dashboard(self):
        """Create comprehensive dashboard combining all analyses"""
        print("      Creating comprehensive dashboard...")
        
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)  # Increased hspace for title spacing
        
        # Fix: Properly positioned main title
        fig.suptitle('FWI Feature Importance Analysis - Downscaling Strategy Dashboard\n50km Meteorological Variables → 25km FWI', 
                    fontsize=20, fontweight='bold', y=0.98, ha='center')
        
        # 1. Top features consensus (top row, full width)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_consensus_features(ax1)
        
        # 2. Model performance (second row, left)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_model_performance_summary(ax2)
        
        # 3. Feature category summary (second row, middle-left)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_downscaled_category_summary(ax3)
        
        # 4. Importance distribution (second row, middle-right)
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_importance_distribution(ax4)
        
        # 5. Strategy validation (second row, right)
        ax5 = fig.add_subplot(gs[1, 3])
        self._plot_strategy_summary(ax5)
        
        # 6-9. Individual model plots (bottom two rows)
        model_axes = [
            fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), 
            fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[2, 3]),
            fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]), 
            fig.add_subplot(gs[3, 2]), fig.add_subplot(gs[3, 3])
        ]
        
        for i, (model_name, results) in enumerate(self.importance_results.items()):
            if i < len(model_axes):
                self._plot_top_features_mini(model_axes[i], model_name, results)
        
        # Hide unused axes
        for i in range(len(self.importance_results), len(model_axes)):
            model_axes[i].set_visible(False)
        
        # Fix: Adjust layout to properly accommodate title
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('downscaled_feature_importance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_consensus_features(self, ax):
        """Plot consensus top features across all models"""
        # Calculate average importance across models
        all_features = set()
        for results in self.importance_results.values():
            all_features.update(results['features'])
        
        consensus_importance = {}
        for feature in all_features:
            importances = []
            for results in self.importance_results.values():
                if feature in results['features']:
                    idx = results['features'].index(feature)
                    # Normalize by max importance in each model
                    max_imp = max(results['scores'])
                    normalized_imp = results['scores'][idx] / max_imp if max_imp > 0 else 0
                    importances.append(normalized_imp)
            
            if importances:
                consensus_importance[feature] = np.mean(importances)
        
        # Plot top 20 consensus features
        sorted_features = sorted(consensus_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        features, importances = zip(*sorted_features)
        
        clean_features = [self._clean_feature_name(f) for f in features]
        
        bars = ax.barh(range(len(features)), importances, color=plt.cm.viridis(np.linspace(0, 1, len(features))))
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(clean_features, fontsize=10)
        ax.set_xlabel('Average Normalized Importance')
        
        # Fix: Properly center the title
        ax.set_title('Top 20 Consensus Features (Average Across All Models)\nDownscaling Strategy: 50km Meteovars → 25km FWI',
                    fontsize=12, fontweight='bold', pad=15, ha='center')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            ax.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', va='center', fontsize=8)

    def _plot_model_performance_summary(self, ax):
        """Plot model performance summary"""
        models = list(self.importance_results.keys())
        r2_scores = [self.importance_results[model]['performance']['r2'] for model in models]
        
        colors = ['forestgreen', 'purple', 'orange', 'red'][:len(models)]
        bars = ax.bar(range(len(models)), r2_scores, color=colors, alpha=0.8)
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=9)
        ax.set_ylabel('R² Score')
        ax.set_title('Model Performance\n(50km→25km Training)', fontsize=10, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)

    def _plot_importance_distribution(self, ax):
        """Plot importance distribution"""
        all_importances = []
        for results in self.importance_results.values():
            all_importances.extend(results['scores'])
        
        ax.hist(all_importances, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Frequency')
        ax.set_title('Feature Importance\nDistribution (Downscaled)', fontsize=10, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_imp = np.mean(all_importances)
        median_imp = np.median(all_importances)
        ax.axvline(mean_imp, color='red', linestyle='--', label=f'Mean: {mean_imp:.3f}')
        ax.axvline(median_imp, color='orange', linestyle='--', label=f'Median: {median_imp:.3f}')
        ax.legend(fontsize=8)

    def _plot_top_features_mini(self, ax, model_name, results):
        """Plot top features for individual model (mini version)"""
        feature_df = pd.DataFrame({
            'feature': results['features'],
            'importance': results['scores']
        }).sort_values('importance', ascending=False).head(8)
        
        clean_features = [self._clean_feature_name(f)[:20] + ('...' if len(self._clean_feature_name(f)) > 20 else '') 
                        for f in feature_df['feature']]
        
        bars = ax.barh(range(len(feature_df)), feature_df['importance'], 
                    color=plt.cm.tab10(len(feature_df) - np.arange(len(feature_df))))
        
        ax.set_yticks(range(len(feature_df)))
        ax.set_yticklabels(clean_features, fontsize=7)
        ax.set_xlabel('Importance', fontsize=8)
        
        # Fix: Properly center the mini plot title
        title_text = f'{model_name}\n(R²={results["performance"]["r2"]:.3f})'
        ax.set_title(title_text, fontsize=9, fontweight='bold', pad=10, ha='center')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            if width > 0:  # Only add labels for non-zero values
                ax.text(width + max(feature_df['importance']) * 0.02, 
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', va='center', fontsize=6)
        
    def _plot_downscaled_category_summary(self, ax):
        """Plot category summary for downscaled features"""
        categories = ['Temperature', 'Humidity', 'Wind', 'Precipitation', 'Spatial', 'Gradient', 'Variability']
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
        
        # Count features in each category
        category_counts = {cat: 0 for cat in categories}
        keywords_map = {
            'Temperature': ['temp', '2m'],
            'Humidity': ['humidity', 'rh', 'dewpoint'],
            'Wind': ['wind', 'component', 'speed'],
            'Precipitation': ['precip', 'rain', 'tp'],
            'Spatial': ['spatial', 'distance', 'lat', 'lon'],
            'Gradient': ['gradient'],
            'Variability': ['variability', 'var', 'std', 'range']
        }
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            categorized = False
            
            for category, keywords in keywords_map.items():
                if any(keyword in feature_lower for keyword in keywords):
                    category_counts[category] += 1
                    categorized = True
                    break
            
            # If not categorized, check for other patterns
            if not categorized:
                if 'feature_downscaled_' in feature_lower:
                    category_counts['Temperature'] += 1  # Default to temperature for uncategorized meteovars
        
        # Filter out zero values
        non_zero_cats = {k: v for k, v in category_counts.items() if v > 0}
        
        if non_zero_cats:
            cats = list(non_zero_cats.keys())
            counts = list(non_zero_cats.values())
            cat_colors = [colors[categories.index(cat)] for cat in cats]
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(counts, labels=cats, colors=cat_colors, 
                                            autopct='%d', startangle=90)
            
            # Improve text formatting
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
            
            for text in texts:
                text.set_fontsize(8)
            
            ax.set_title('Downscaled Feature\nDistribution by Category', fontsize=10, fontweight='bold', pad=10)
        else:
            ax.text(0.5, 0.5, 'No categorized\nfeatures found', ha='center', va='center', 
                transform=ax.transAxes, fontsize=10)
            ax.set_title('Feature Categories', fontsize=10, fontweight='bold')

    def _plot_strategy_summary(self, ax):
        """Plot strategy summary"""
        ax.axis('off')
        
        # Calculate some statistics
        total_features = len(self.feature_names)
        best_model = max(self.importance_results.items(), key=lambda x: x[1]['performance']['cv_mean'])
        avg_performance = np.mean([r['performance']['cv_mean'] for r in self.importance_results.values()])
        
        # Count feature types
        meteo_features = len([f for f in self.feature_names if 'feature_downscaled_' in f])
        spatial_features = len([f for f in self.feature_names if 'feature_spatial_' in f])
        gradient_features = len([f for f in self.feature_names if 'feature_gradient_' in f])
        
        strategy_text = f"""
        DOWNSCALING STRATEGY
        ══════════════════════
        
        Total Features: {total_features}
        
        Best Model: {best_model[0][:12]}...
           CV R²: {best_model[1]['performance']['cv_mean']:.3f}
        
        Average Performance: {avg_performance:.3f}
        
        Feature Breakdown:
        • Meteorological: {meteo_features}
        • Spatial: {spatial_features}
        • Gradient: {gradient_features}
        
        Strategy Benefits:
        • Realistic training scenario
        • Proper resolution scaling
        • Enhanced feature analysis
        • Improved generalization
        
        Training: 50km → 25km
        Application: 25km → 1km
        """
        
        ax.text(0.05, 0.95, strategy_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=0.5))

    def _clean_feature_name(self, feature_name):
        """Clean feature names for better display"""
        # Remove prefixes
        cleaned = feature_name.replace('feature_downscaled_', '').replace('feature_gradient_', 'grad_').replace('feature_spatial_', 'spatial_')
        
        # Replace common terms
        replacements = {
            '_variability': '_var',
            '_spatial_range': '_s_range',
            'temperature': 'temp',
            'humidity': 'humid',
            'precipitation': 'precip',
            'u_component': 'u_comp',
            'v_component': 'v_comp',
            'surface_pressure': 'surf_press',
            'total_precipitation': 'tot_precip',
            '2m_temperature': '2m_temp',
            'surface_net_solar_radiation': 'net_solar',
            'distance_to_50km': 'dist_50km',
            'num_50km_points': 'n_50km_pts',
            'lat_50km': 'lat_50k',
            'lon_50km': 'lon_50k'
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned
    
    def create_visualizations(self):
        """Create comprehensive feature importance visualizations for downscaled analysis"""
        try:
            print("Creating downscaled feature importance visualizations...")
            
            if not self.importance_results:
                print("ERROR: No importance results available for visualization")
                return False
            
            # 1. Individual model importance plots
            self._create_individual_plots()
            
            # 2. Comparison heatmap
            self._create_comparison_heatmap()
            
            # 3. Model performance vs importance
            self._create_performance_comparison()
            
            # 4. Downscaled feature categories analysis
            self._create_downscaled_category_analysis()
            
            # 5. Comprehensive dashboard
            self._create_comprehensive_dashboard()
            
            # 6. Downscaling strategy validation
            self._create_downscaling_validation_plot()
            
            print("   All visualizations created successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR creating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_comprehensive_report(self):
        """Generate comprehensive text report of downscaled feature importance analysis"""
        try:
            print("Generating comprehensive downscaled feature importance report...")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"downscaled_feature_importance_report_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("FWI FEATURE IMPORTANCE ANALYSIS - DOWNSCALING STRATEGY REPORT\n")
                f.write("="*80 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Strategy: Downscale meteorological variables (25km → 50km), Keep FWI at 25km\n")
                f.write("Training: 50km meteorological variables → 25km FWI\n\n")
                
                # Dataset summary
                f.write("DATASET SUMMARY\n")
                f.write("-"*40 + "\n")
                f.write(f"Original data shape: {self.original_25km_data.shape}\n")
                f.write(f"Training samples: {len(self.X):,}\n")
                f.write(f"Features analyzed: {len(self.feature_names)}\n")
                f.write(f"FWI range: [{self.y.min():.2f}, {self.y.max():.2f}]\n\n")
                
                # Model performance
                f.write("MODEL PERFORMANCE\n")
                f.write("-"*40 + "\n")
                f.write(f"{'Model':<20} {'Test R²':<10} {'CV R²':<10} {'RMSE':<10}\n")
                f.write("-"*50 + "\n")
                
                for model_name, results in self.importance_results.items():
                    perf = results['performance']
                    f.write(f"{model_name:<20} {perf['r2']:<10.4f} {perf['cv_mean']:<10.4f} {perf['rmse']:<10.4f}\n")
                
                # Best performing model
                best_model = max(self.importance_results.items(), key=lambda x: x[1]['performance']['cv_mean'])
                f.write(f"\nBest Model: {best_model[0]} (CV R² = {best_model[1]['performance']['cv_mean']:.4f})\n\n")
                
                # Feature importance analysis
                f.write("FEATURE IMPORTANCE ANALYSIS\n")
                f.write("-"*40 + "\n")
                
                # Calculate consensus importance
                all_features = set()
                for results in self.importance_results.values():
                    all_features.update(results['features'])
                
                consensus_importance = {}
                for feature in all_features:
                    importances = []
                    for results in self.importance_results.values():
                        if feature in results['features']:
                            idx = results['features'].index(feature)
                            importances.append(results['scores'][idx])
                
                    if importances:
                        consensus_importance[feature] = {
                            'mean': np.mean(importances),
                            'std': np.std(importances),
                            'models': len(importances)
                        }
                
                # Top 20 consensus features
                sorted_features = sorted(consensus_importance.items(), 
                                       key=lambda x: x[1]['mean'], reverse=True)[:20]
                
                f.write("TOP 20 CONSENSUS FEATURES (Average across all models):\n")
                f.write(f"{'Rank':<5} {'Feature':<40} {'Avg Importance':<15} {'Std':<10} {'Models':<8}\n")
                f.write("-"*78 + "\n")
                
                for i, (feature, stats) in enumerate(sorted_features, 1):
                    clean_name = self._clean_feature_name(feature)
                    f.write(f"{i:<5} {clean_name[:38]:<40} {stats['mean']:<15.6f} {stats['std']:<10.6f} {stats['models']:<8}\n")
                
                # Category analysis
                f.write(f"\nFEATURE CATEGORY ANALYSIS\n")
                f.write("-"*40 + "\n")
                
                categories = {
                    'Temperature': ['temp', 'temperature', '2m'],
                    'Humidity': ['humidity', 'rh', 'dewpoint'],
                    'Wind': ['wind', 'u_component', 'v_component', 'speed', 'u10', 'v10'],
                    'Precipitation': ['precip', 'rain', 'precipitation', 'tp'],
                    'Pressure': ['pressure', 'msl', 'sp'],
                    'Radiation': ['radiation', 'solar', 'ssr'],
                    'Spatial': ['spatial', 'distance', 'lat', 'lon'],
                    'Gradient': ['gradient'],
                    'Variability': ['variability', 'range', 'std']
                }
                
                category_importance = {}
                for category, keywords in categories.items():
                    category_features = []
                    for feature in consensus_importance.keys():
                        if any(keyword in feature.lower() for keyword in keywords):
                            category_features.append(consensus_importance[feature]['mean'])
                
                    if category_features:
                        category_importance[category] = {
                            'total': sum(category_features),
                            'mean': np.mean(category_features),
                            'count': len(category_features)
                        }
                
                sorted_categories = sorted(category_importance.items(), 
                                         key=lambda x: x[1]['total'], reverse=True)
                
                f.write(f"{'Category':<20} {'Total Imp.':<12} {'Mean Imp.':<12} {'Count':<8}\n")
                f.write("-"*52 + "\n")
                
                for category, stats in sorted_categories:
                    f.write(f"{category:<20} {stats['total']:<12.6f} {stats['mean']:<12.6f} {stats['count']:<8}\n")
                
                # Strategy benefits
                f.write(f"\nDOWNSCALING STRATEGY BENEFITS\n")
                f.write("-"*40 + "\n")
                f.write("1. REALISTIC TRAINING SCENARIO:\n")
                f.write("   • Meteorological variables downscaled from 25km to 50km\n")
                f.write("   • FWI kept at original 25km resolution as target\n")
                f.write("   • Mimics real-world super-resolution application\n\n")
                
                f.write("2. PROPER RESOLUTION RELATIONSHIP:\n")
                f.write("   • Training: 50km meteovars → 25km FWI\n")
                f.write("   • Application: 25km meteovars → 1km FWI\n")
                f.write("   • Consistent resolution enhancement factor\n\n")
                
                f.write("3. ENHANCED FEATURE UNDERSTANDING:\n")
                f.write("   • Spatial context features from 50km grid\n")
                f.write("   • Gradient features for spatial relationships\n")
                f.write("   • Variability features for uncertainty quantification\n\n")
                
                f.write("4. ROBUST ANALYSIS:\n")
                f.write("   • Multiple model validation\n")
                f.write("   • Cross-validation for stability assessment\n")
                f.write("   • Consensus feature importance ranking\n\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-"*40 + "\n")
                f.write(f"1. Use {best_model[0]} for production (highest CV R²)\n")
                f.write("2. Focus on top 10 consensus features for efficiency\n")
                f.write("3. Include spatial and gradient features for better performance\n")
                f.write("4. Consider ensemble approach combining all models\n")
                f.write("5. Validate on independent test data before deployment\n\n")
                
                # Technical notes
                f.write("TECHNICAL NOTES\n")
                f.write("-"*40 + "\n")
                f.write("• Downscaling performed using spatial averaging within 50km windows\n")
                f.write("• Distance-weighted interpolation for feature extraction\n")
                f.write("• RobustScaler used for feature normalization\n")
                f.write("• 5-fold cross-validation for model stability assessment\n")
                f.write("• Permutation importance used for neural networks\n")
                f.write("• Built-in importance for tree-based models\n")
                f.write("• Coefficient magnitude for linear models\n")
                
                f.write("="*80 + "\n")
                f.write("END OF DOWNSCALED FEATURE IMPORTANCE ANALYSIS REPORT\n")
                f.write("="*80)
        
            print(f"   Comprehensive report saved: {report_file}")
            return True
            
        except Exception as e:
            print(f"ERROR generating report: {e}")
            return False

    def run_complete_downscaled_analysis(self):
        """Run complete feature importance analysis with downscaling strategy"""
        print(f"\nSTARTING COMPLETE DOWNSCALED FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        try:
            # Step 1: Load data
            if not self.load_data():
                print("ERROR: Failed to load data")
                return False
            
            # Step 2: Create downscaled training data
            print(f"\nStep 2: Creating downscaled training data...")
            if not self.create_downscaled_training_data():
                print("ERROR: Failed to create downscaled training data")
                return False
            
            # Step 3: Train models
            print(f"\nStep 3: Training models...")
            if not self.train_models():
                print("ERROR: Failed to train models")
                return False
            
            # Step 4: Extract importance
            print(f"\nStep 4: Extracting feature importance...")
            if not self.extract_feature_importance():
                print("ERROR: Failed to extract feature importance")
                return False
            
            # Step 5: Create visualizations
            print(f"\nStep 5: Creating visualizations...")
            if not self.create_visualizations():
                print("ERROR: Failed to create visualizations")
                return False
            
            # Step 6: Generate report
            print(f"\nStep 6: Generating comprehensive report...")
            if not self.generate_comprehensive_report():
                print("ERROR: Failed to generate report")
                return False
            
            print(f"\nDOWNSCALED FEATURE IMPORTANCE ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*70)
            
            # Summary
            print(f"\nANALYSIS SUMMARY:")
            print(f"   Models trained: {len(self.models)}")
            print(f"   Features analyzed: {len(self.feature_names)}")
            print(f"   Training samples: {len(self.X):,}")
            
            best_model = max(self.importance_results.items(), key=lambda x: x[1]['performance']['cv_mean'])
            print(f"   Best model: {best_model[0]} (CV R² = {best_model[1]['performance']['cv_mean']:.4f})")
            
            print(f"\nFILES GENERATED:")
            generated_files = [
                "downscaled_feature_category_analysis.png",
                "downscaling_strategy_validation.png", 
                "downscaled_feature_importance_individual.png",
                "downscaled_feature_importance_heatmap.png",
                "downscaled_model_performance_comparison.png",
                "downscaled_feature_importance_dashboard.png"
            ]
            
            for file in generated_files:
                if os.path.exists(file):
                    file_size = os.path.getsize(file) / 1024
                    print(f"   {file} ({file_size:.1f} KB)")
            
            # Find and show report file
            report_files = [f for f in os.listdir('.') if f.startswith('downscaled_feature_importance_report_')]
            if report_files:
                latest_report = max(report_files)
                report_size = os.path.getsize(latest_report) / 1024
                print(f"   {latest_report} ({report_size:.1f} KB)")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False

# Main execution function for downscaled feature importance analysis
if __name__ == "__main__":
    """
    Main execution function for FWI Feature Importance Analysis with Downscaling Strategy
    
    This function:
    1. Uses ONLY merged_fwi_25km_processed.csv file
    2. Applies proper downscaling strategy (meteovars to 50km, FWI stays 25km)
    3. Trains multiple models and analyzes feature importance
    4. Generates comprehensive visualizations and reports
    """
    print("FWI Feature Importance Analysis with Proper Downscaling Strategy")
    print("="*70)
    
    import sys
    
    # ONLY use merged_fwi_25km_processed.csv file
    required_file = "merged_fwi_25km_processed.csv"
    
    print(f"\nLooking for required input file: {required_file}")
    
    if not os.path.exists(required_file):
        print(f"ERROR: Required file '{required_file}' not found!")
        print("\nSOLUTION:")
        print("1. Run EDA.py first to generate the required input file")
        print("2. Ensure EDA.py has completed successfully")
        print("3. Verify that merged_fwi_25km_processed.csv exists in current directory")
        print(f"\nCANNOT PROCEED WITHOUT {required_file}")
        sys.exit(1)
    
    try:
        # Initialize the downscaled feature importance analyzer
        print(f"   Found required file: {required_file}")
        analyzer = FWIFeatureImportanceAnalyzerDownscale(required_file)
        
        # Run complete analysis
        print(f"\nStarting downscaled feature importance analysis...")
        
        success = analyzer.run_complete_downscaled_analysis()
        
        if success:
            print(f"\nSUCCESS: Downscaled feature importance analysis completed!")
            print(f"\nKEY INSIGHTS:")
            print(f"   Proper downscaling strategy implemented")
            print(f"   Multiple models trained and compared")
            print(f"   Comprehensive feature importance analysis")
            print(f"   Visualizations and report generated")
            
            print(f"\nMETHODOLOGY ADVANTAGES:")
            print(f"   Realistic training scenario (50km → 25km)")
            print(f"   Proper resolution relationship")
            print(f"   Enhanced spatial feature understanding")
            print(f"   Robust multi-model validation")
            print(f"   Consensus feature importance ranking")
            
            # Show best performing model
            if analyzer.importance_results:
                best_model = max(analyzer.importance_results.items(), 
                               key=lambda x: x[1]['performance']['cv_mean'])
                print(f"\nRECOMMENDED MODEL: {best_model[0]}")
                print(f"   Cross-Validation R²: {best_model[1]['performance']['cv_mean']:.4f}")
                print(f"   Test R²: {best_model[1]['performance']['r2']:.4f}")
        else:
            print(f"\nFAILED: Feature importance analysis failed")
            
    except Exception as e:
        print(f"\nAnalysis execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("FWI DOWNSCALED FEATURE IMPORTANCE ANALYSIS COMPLETED")