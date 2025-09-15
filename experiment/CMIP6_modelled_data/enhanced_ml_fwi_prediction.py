#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhanced_ml_fwi_prediction.py

Enhanced ML-based FWI prediction without ground truth data
Uses improved FWI calculation and advanced feature engineering
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Enhanced ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Enhanced_ML_FWI")

class ImprovedFWICalculator:
    """Improved FWI calculator integrated into the main class"""
    
    def __init__(self):
        self.previous_ffmc = 85.0
        self.previous_dmc = 6.0
        self.previous_dc = 15.0
    
    def calculate_relative_humidity(self, temp_k, huss):
        """Calculate relative humidity from temperature and specific humidity"""
        temp_c = temp_k - 273.15
        
        # Simplified RH calculation from specific humidity
        # More accurate would require pressure, but this is a reasonable approximation
        rh = huss * 100  # Convert to percentage (simplified)
        rh = np.clip(rh, 0, 100)
        
        return rh
    
    def calculate_fwi_simplified(self, temp_c, rh, wind_kmh, rain_mm):
        """Calculate simplified FWI directly from weather variables"""
        # Direct FWI calculation without intermediate codes
        
        # Temperature factor (normalized)
        temp_factor = np.maximum(temp_c, 0) / 40.0  # Normalize to 0-1 range
        
        # Humidity factor (inverted - lower humidity = higher fire risk)
        humidity_factor = (100 - rh) / 100.0
        
        # Wind factor
        wind_factor = wind_kmh / 50.0  # Normalize to reasonable range
        
        # Rain factor (inverted - more rain = lower fire risk)
        rain_factor = np.maximum(1 - rain_mm / 10.0, 0)  # Normalize rain
        
        # Combine factors
        fwi = temp_factor * humidity_factor * wind_factor * rain_factor * 50
        
        return np.maximum(fwi, 0)
    
    def calculate_daily_fwi(self, df):
        """Calculate FWI for daily data"""
        results = []
        
        for _, row in df.iterrows():
            # Convert units
            temp_c = row['tasmax'] - 273.15
            wind_kmh = row['sfcWind'] * 3.6
            rain_mm = row['pr'] * 86400  # Convert m/s to mm/day
            
            # Calculate relative humidity
            rh = self.calculate_relative_humidity(row['tasmax'], row['huss'])
            
            # Calculate simplified FWI
            fwi = self.calculate_fwi_simplified(temp_c, rh, wind_kmh, rain_mm)
            
            results.append({
                'time': row['time'],
                'lat': row['lat'],
                'lon': row['lon'],
                'fwi_improved': fwi,
                'temp_c': temp_c,
                'rh': rh,
                'wind_kmh': wind_kmh,
                'rain_mm': rain_mm
            })
        
        return pd.DataFrame(results)

class EnhancedFWIPredictor:
    """Enhanced FWI predictor with improved features and models"""
    
    def __init__(self, target_resolution=25):
        self.target_resolution = target_resolution
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
        # FWI calculator
        self.fwi_calculator = ImprovedFWICalculator()
        
        logger.info(f"Initialized Enhanced FWI Predictor")
        logger.info(f"Target resolution: {target_resolution} km")
    
    def parse_time_robust(self, df):
        """Robust time parsing with multiple format attempts"""
        logger.info("Parsing time column...")
        
        # Try different time parsing methods
        time_col = 'time'
        
        # Check current format
        sample_times = df[time_col].head(10)
        print(f"Sample time values: {sample_times.tolist()}")
        
        try:
            # Try ISO8601 format first
            df[time_col] = pd.to_datetime(df[time_col], format='ISO8601')
            logger.info("Successfully parsed using ISO8601 format")
        except:
            try:
                # Try mixed format
                df[time_col] = pd.to_datetime(df[time_col], format='mixed')
                logger.info("Successfully parsed using mixed format")
            except:
                try:
                    # Try infer_datetime_format
                    df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True)
                    logger.info("Successfully parsed using infer_datetime_format")
                except:
                    try:
                        # Try manual format detection
                        if 'T' in str(df[time_col].iloc[0]):
                            # ISO format with T separator
                            df[time_col] = pd.to_datetime(df[time_col], format='%Y-%m-%dT%H:%M:%S')
                        else:
                            # Space separator
                            df[time_col] = pd.to_datetime(df[time_col], format='%Y-%m-%d %H:%M:%S')
                        logger.info("Successfully parsed using manual format detection")
                    except Exception as e:
                        logger.error(f"All time parsing methods failed: {e}")
                        # Create dummy dates if all else fails
                        logger.info("Creating dummy sequential dates...")
                        start_date = pd.Timestamp('2013-01-01')
                        df[time_col] = pd.date_range(start=start_date, periods=len(df), freq='D')
        
        # Remove any remaining NaT values
        before_count = len(df)
        df = df.dropna(subset=[time_col])
        after_count = len(df)
        
        if before_count != after_count:
            logger.warning(f"Removed {before_count - after_count} rows with invalid dates")
        
        logger.info(f"Time range: {df[time_col].min()} to {df[time_col].max()}")
        return df
    
    def create_advanced_features(self, df):
        """Create advanced meteorological and fire weather features"""
        logger.info("Creating advanced features...")
        
        df_features = df.copy()
        
        # Parse time first
        df_features = self.parse_time_robust(df_features)
        
        # 1. Basic conversions
        df_features['temp_celsius'] = df_features['tasmax'] - 273.15
        df_features['wind_kmh'] = df_features['sfcWind'] * 3.6
        df_features['precip_mm'] = df_features['pr'] * 86400
        df_features['humidity_percent'] = df_features['huss'] * 100
        
        # 2. Fire weather components
        # Vapor Pressure Deficit (more accurate)
        es = 6.112 * np.exp(17.67 * df_features['temp_celsius'] / (df_features['temp_celsius'] + 243.5))
        ea = df_features['humidity_percent'] / 100.0 * es
        df_features['vpd'] = es - ea
        
        # Atmospheric pressure (simplified from temperature)
        df_features['pressure_hpa'] = 1013.25 * np.exp(-0.0001184 * df_features['temp_celsius'])
        
        # 3. Fire danger indices
        # Heat index
        df_features['heat_index'] = self.calculate_heat_index(df_features['temp_celsius'], df_features['humidity_percent'])
        
        # Wind chill (for winter conditions)
        df_features['wind_chill'] = self.calculate_wind_chill(df_features['temp_celsius'], df_features['wind_kmh'])
        
        # Burning index components
        df_features['drying_factor'] = (df_features['temp_celsius'] + 10) * (100 - df_features['humidity_percent']) / 100
        df_features['wind_factor'] = df_features['wind_kmh'] ** 0.5
        df_features['fire_load'] = df_features['drying_factor'] * df_features['wind_factor']
        
        # 4. Temporal features
        df_features['year'] = df_features['time'].dt.year
        df_features['month'] = df_features['time'].dt.month
        df_features['day'] = df_features['time'].dt.day
        df_features['day_of_year'] = df_features['time'].dt.dayofyear
        df_features['week'] = df_features['time'].dt.isocalendar().week
        
        # Seasonal features
        df_features['sin_month'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['cos_month'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['sin_day'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
        df_features['cos_day'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
        
        # Fire season indicator
        df_features['fire_season'] = ((df_features['month'] >= 6) & (df_features['month'] <= 9)).astype(int)
        
        # 5. Lag features (simplified - avoid complex groupby operations)
        df_features = df_features.sort_values(['lat', 'lon', 'time'])
        
        # Simple lag features without groupby
        lag_variables = ['temp_celsius', 'humidity_percent', 'wind_kmh', 'precip_mm']
        for var in lag_variables:
            # Simple 1-day lag
            df_features[f'{var}_lag1'] = df_features[var].shift(1)
            
            # Simple 7-day rolling mean
            df_features[f'{var}_7d_mean'] = df_features[var].rolling(window=7, min_periods=1).mean()
        
        # 6. Drought indicators
        # Cumulative precipitation
        df_features['precip_7d_sum'] = df_features['precip_mm'].rolling(window=7, min_periods=1).sum()
        df_features['precip_30d_sum'] = df_features['precip_mm'].rolling(window=30, min_periods=1).sum()
        
        # Simple days since rain
        df_features['rain_day'] = (df_features['precip_mm'] > 1.0).astype(int)
        df_features['days_since_rain'] = df_features['rain_day'].cumsum() - df_features['rain_day'].cumsum().where(df_features['rain_day'] == 1).ffill()
        df_features['days_since_rain'] = df_features['days_since_rain'].fillna(0)
        
        # 7. Spatial features
        # Elevation proxy (based on latitude for Portugal)
        df_features['elevation_proxy'] = (df_features['lat'] - df_features['lat'].min()) * 500
        
        # Distance from coast (simplified)
        df_features['coast_distance'] = np.minimum(
            np.abs(df_features['lon'] - df_features['lon'].min()),
            np.abs(df_features['lon'] - df_features['lon'].max())
        ) * 111  # Convert to km
        
        # Terrain roughness (simplified)
        df_features['terrain_roughness'] = np.abs(df_features['lat'] - df_features['lat'].mean()) * \
                                          np.abs(df_features['lon'] - df_features['lon'].mean())
        
        # 8. Interaction features
        # Temperature-humidity interactions
        df_features['temp_humidity_interaction'] = df_features['temp_celsius'] * (100 - df_features['humidity_percent'])
        df_features['temp_wind_interaction'] = df_features['temp_celsius'] * df_features['wind_kmh']
        df_features['humidity_wind_interaction'] = df_features['humidity_percent'] * df_features['wind_kmh']
        
        # Drought-fire danger interaction
        df_features['drought_fire_interaction'] = df_features['days_since_rain'] * df_features['fire_load']
        
        logger.info(f"Created {df_features.shape[1] - df.shape[1]} new features")
        return df_features
    
    def calculate_heat_index(self, temp_c, rh):
        """Calculate heat index"""
        temp_f = temp_c * 9/5 + 32
        
        # Simplified heat index calculation
        hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (rh * 0.094))
        
        # More complex formula for higher temperatures
        mask = hi > 80
        if hasattr(mask, 'any') and mask.any():
            hi_complex = (-42.379 + 2.04901523 * temp_f + 10.14333127 * rh -
                         0.22475541 * temp_f * rh - 0.00683783 * temp_f**2 -
                         0.05481717 * rh**2 + 0.00122874 * temp_f**2 * rh +
                         0.00085282 * temp_f * rh**2 - 0.00000199 * temp_f**2 * rh**2)
            hi = np.where(mask, hi_complex, hi)
        
        return (hi - 32) * 5/9  # Convert back to Celsius
    
    def calculate_wind_chill(self, temp_c, wind_kmh):
        """Calculate wind chill"""
        temp_f = temp_c * 9/5 + 32
        wind_mph = wind_kmh * 0.621371
        
        if isinstance(temp_f, (int, float)):
            if temp_f <= 50 and wind_mph > 3:
                wc = 35.74 + 0.6215 * temp_f - 35.75 * (wind_mph ** 0.16) + 0.4275 * temp_f * (wind_mph ** 0.16)
            else:
                wc = temp_f
        else:
            wc = np.where((temp_f <= 50) & (wind_mph > 3),
                         35.74 + 0.6215 * temp_f - 35.75 * (wind_mph ** 0.16) + 0.4275 * temp_f * (wind_mph ** 0.16),
                         temp_f)
        
        return (wc - 32) * 5/9  # Convert back to Celsius
    
    def prepare_training_data(self, df):
        """Prepare training data with improved FWI targets"""
        logger.info("Preparing training data...")
        
        # Create advanced features
        df_features = self.create_advanced_features(df)
        
        # Calculate improved FWI as target
        logger.info("Calculating improved FWI targets...")
        fwi_results = self.fwi_calculator.calculate_daily_fwi(df_features)
        
        # Merge FWI results with features
        df_final = df_features.merge(fwi_results[['time', 'lat', 'lon', 'fwi_improved']], 
                                   on=['time', 'lat', 'lon'], how='left')
        
        # Remove rows without FWI values
        df_final = df_final.dropna(subset=['fwi_improved'])
        
        # Select feature columns - exclude all non-numeric columns
        exclude_cols = [
            'time', 'lat', 'lon', 'tasmax', 'huss', 'sfcWind', 'pr', 'fwi_improved',
            'bnds', 'time_bnds', 'lat_bnds', 'lon_bnds', 'height'  # Additional columns to exclude
        ]
        
        # Get all columns first
        all_cols = df_final.columns.tolist()
        
        # Select only numeric columns
        numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns from numeric columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Remove columns with too many NaN values
        nan_threshold = 0.5  # Increased threshold
        feature_cols = [col for col in feature_cols 
                       if df_final[col].isna().sum() / len(df_final) < nan_threshold]
        
        # Remove columns with zero variance
        feature_cols = [col for col in feature_cols 
                       if df_final[col].var() > 1e-10]
        
        self.feature_names = feature_cols
        
        logger.info(f"Selected features: {feature_cols}")
        
        # Prepare matrices
        X = df_final[feature_cols].fillna(0).values
        y = df_final['fwi_improved'].values
        
        # Validate that X contains only numeric data
        if not np.issubdtype(X.dtype, np.number):
            logger.error("Feature matrix contains non-numeric data")
            # Print problematic columns
            for i, col in enumerate(feature_cols):
                col_data = df_final[col].values
                if not np.issubdtype(col_data.dtype, np.number):
                    logger.error(f"Column {col} has non-numeric dtype: {col_data.dtype}")
            return None, None, None
        
        # Check for infinite values
        if np.any(np.isinf(X)):
            logger.warning("Found infinite values in feature matrix, replacing with 0")
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Target range: {y.min():.2f} to {y.max():.2f}")
        logger.info(f"Features: {len(feature_cols)}")
        
        return X, y, df_final
    
    def train_ensemble_models(self, X, y):
        """Train ensemble of models"""
        logger.info("Training ensemble models...")
        
        # Validate input data
        if X is None or y is None:
            logger.error("Invalid training data")
            return {}
        
        # Final check for data types
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        
        # 1. Random Forest (simplified parameters)
        logger.info("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X, y)
        self.models['random_forest'] = rf_model
        
        # 2. Gradient Boosting
        logger.info("Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb_model.fit(X, y)
        self.models['gradient_boosting'] = gb_model
        
        # 3. XGBoost
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0  # Reduce XGBoost output
        )
        xgb_model.fit(X, y)
        self.models['xgboost'] = xgb_model
        
        # Evaluate models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_scores = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            model_scores[name] = {'r2': r2, 'rmse': rmse, 'mae': mae}
            
            logger.info(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Feature importance (from best model)
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['r2'])
        best_model = self.models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = list(zip(self.feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 15 Most Important Features ({best_model_name}):")
            for i, (feature, importance) in enumerate(feature_importance[:15]):
                print(f"  {i+1:2d}. {feature}: {importance:.4f}")
        
        return model_scores
    
    def predict_ensemble(self, X):
        """Make ensemble predictions"""
        predictions = {}
        
        # Ensure X is float64
        X = X.astype(np.float64)
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions[name] = pred
        
        # Equal weight ensemble
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += pred / len(predictions)
        
        return ensemble_pred, predictions
    
    def predict_fwi(self, df):
        """Predict FWI for new data"""
        logger.info("Predicting FWI...")
        
        # Create features
        df_features = self.create_advanced_features(df)
        
        # Prepare feature matrix
        X = df_features[self.feature_names].fillna(0).values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)  # Handle any remaining issues
        
        # Make predictions
        ensemble_pred, individual_preds = self.predict_ensemble(X)
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df = self.parse_time_robust(result_df)  # Parse time for result
        result_df['fwi_predicted'] = ensemble_pred
        
        for name, pred in individual_preds.items():
            result_df[f'fwi_{name}'] = pred
        
        # Ensure positive predictions
        result_df['fwi_predicted'] = np.maximum(result_df['fwi_predicted'], 0.01)
        
        logger.info(f"FWI predictions range: {ensemble_pred.min():.2f} to {ensemble_pred.max():.2f}")
        
        return result_df

def main():
    """Main function"""
    print("Enhanced ML-based FWI Prediction")
    print("=" * 60)
    
    # Configuration
    target_resolution = 25
    
    # Find input file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(script_dir) if f.startswith('cmip6_merged_') and f.endswith('.csv')]
    
    if not csv_files:
        print("No merged CMIP6 CSV files found!")
        return
    
    csv_file = csv_files[0]
    csv_path = os.path.join(script_dir, csv_file)
    
    print(f"Input file: {csv_file}")
    print(f"Target resolution: {target_resolution} km")
    
    # Load data
    print("\nLoading CMIP6 data...")
    df = pd.read_csv(csv_path)
    
    logger.info(f"Loaded data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Initialize predictor
    predictor = EnhancedFWIPredictor(target_resolution=target_resolution)
    
    # Prepare training data
    print("\nPreparing training data...")
    X, y, df_processed = predictor.prepare_training_data(df)
    
    if X is None or len(X) == 0:
        print("❌ No training data prepared")
        return
    
    # Train models
    print("\nTraining ensemble models...")
    model_scores = predictor.train_ensemble_models(X, y)
    
    if not model_scores:
        print("❌ No models trained successfully")
        return
    
    # Make predictions
    print("\nGenerating FWI predictions...")
    results = predictor.predict_fwi(df)
    
    # Save results
    output_file = f"enhanced_fwi_predictions_{target_resolution}km.csv"
    output_path = os.path.join(script_dir, output_file)
    
    # Select output columns
    output_cols = ['time', 'lat', 'lon', 'tasmax', 'huss', 'sfcWind', 'pr', 'fwi_predicted']
    
    # Add individual model predictions if available
    for name in ['random_forest', 'gradient_boosting', 'xgboost']:
        if f'fwi_{name}' in results.columns:
            output_cols.append(f'fwi_{name}')
    
    results[output_cols].to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Output shape: {results.shape}")
    
    # Statistics
    fwi_pred = results['fwi_predicted']
    print(f"\nEnhanced FWI Prediction Statistics:")
    print(f"  Mean: {fwi_pred.mean():.2f}")
    print(f"  Std: {fwi_pred.std():.2f}")
    print(f"  Min: {fwi_pred.min():.2f}")
    print(f"  Max: {fwi_pred.max():.2f}")
    print(f"  Median: {fwi_pred.median():.2f}")
    
    # Fire danger categories
    low_fwi = results[results['fwi_predicted'] < 10]
    moderate_fwi = results[(results['fwi_predicted'] >= 10) & (results['fwi_predicted'] < 20)]
    high_fwi = results[(results['fwi_predicted'] >= 20) & (results['fwi_predicted'] < 30)]
    very_high_fwi = results[results['fwi_predicted'] >= 30]
    
    print(f"\nFire Danger Categories:")
    print(f"  Low (0-10): {len(low_fwi)} ({len(low_fwi)/len(results)*100:.1f}%)")
    print(f"  Moderate (10-20): {len(moderate_fwi)} ({len(moderate_fwi)/len(results)*100:.1f}%)")
    print(f"  High (20-30): {len(high_fwi)} ({len(high_fwi)/len(results)*100:.1f}%)")
    print(f"  Very High (30+): {len(very_high_fwi)} ({len(very_high_fwi)/len(results)*100:.1f}%)")
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # Model performance summary
    print(f"\nModel Performance Summary:")
    for name, scores in model_scores.items():
        print(f"  {name}: R² = {scores['r2']:.4f}, RMSE = {scores['rmse']:.4f}")
    
    print("\n" + "=" * 60)
    print("Enhanced FWI Prediction completed!")

if __name__ == "__main__":
    main()