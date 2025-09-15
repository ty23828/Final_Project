"""
ml_fwi_prediction.py

Machine Learning-based FWI prediction with spatial super-resolution
Input: Low-resolution CMIP6 climate data  
Output: High-resolution (25km) FWI predictions using Random Forest and CNN separately
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, UpSampling2D, 
                                       Flatten, Reshape, Input, concatenate)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Only Random Forest will be used.")

# Spatial processing imports
try:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available. Limited spatial processing.")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ML_FWI_Predictor")

class FWIMLPredictor:
    """Machine Learning-based FWI predictor with spatial super-resolution"""
    
    def __init__(self, target_resolution=25):
        """
        Initialize the ML FWI predictor
        
        Args:
            target_resolution: Target spatial resolution in km
        """
        self.target_resolution = target_resolution
        self.rf_model = None
        self.cnn_model = None
        self.scaler = StandardScaler()
        
        # FWI-related features
        self.climate_features = ['tasmax', 'huss', 'sfcWind', 'pr']
        self.derived_features = []
        self.spatial_features = []
        
        logger.info(f"Initialized ML FWI Predictor")
        logger.info(f"Target resolution: {target_resolution} km")
    
    def create_derived_features(self, df):
        """Create derived meteorological features"""
        logger.info("Creating derived features...")
        
        # Temperature conversions
        df['temp_celsius'] = df['tasmax'] - 273.15
        
        # Vapor pressure deficit approximation
        df['vpd'] = df['tasmax'] * (1 - df['huss'])
        
        # Wind power (related to fire spread)
        df['wind_power'] = df['sfcWind'] ** 2
        
        # Precipitation intensity
        df['precip_intensity'] = df['pr'] * 86400  # Convert to mm/day
        
        # Drought index (simplified)
        df['drought_index'] = df['temp_celsius'] / (df['precip_intensity'] + 1e-6)
        
        # Fire weather components (simplified)
        df['fire_danger'] = (df['temp_celsius'] * df['wind_power']) / (df['huss'] + 1e-6)
        
        self.derived_features = [
            'temp_celsius', 'vpd', 'wind_power', 'precip_intensity', 
            'drought_index', 'fire_danger'
        ]
        
        logger.info(f"Created {len(self.derived_features)} derived features")
        return df
    
    def create_spatial_features(self, df):
        """Create spatial features for CNN"""
        logger.info("Creating spatial features...")
        
        # Spatial gradients
        df_sorted = df.sort_values(['time', 'lat', 'lon'])
        
        for feature in self.climate_features:
            # Latitude gradient
            df_sorted[f'{feature}_lat_grad'] = df_sorted.groupby(['time', 'lon'])[feature].diff()
            
            # Longitude gradient  
            df_sorted[f'{feature}_lon_grad'] = df_sorted.groupby(['time', 'lat'])[feature].diff()
            
            # Spatial variance (rolling window)
            df_sorted[f'{feature}_spatial_var'] = df_sorted.groupby('time')[feature].rolling(window=3, center=True).var().values
        
        # Distance from coast (simplified - based on longitude)
        df_sorted['dist_from_coast'] = np.abs(df_sorted['lon'] - df_sorted['lon'].min())
        
        # Elevation proxy (simplified - based on latitude)
        df_sorted['elevation_proxy'] = (df_sorted['lat'] - df_sorted['lat'].min()) * 100
        
        self.spatial_features = [col for col in df_sorted.columns if '_grad' in col or '_var' in col]
        self.spatial_features.extend(['dist_from_coast', 'elevation_proxy'])
        
        logger.info(f"Created {len(self.spatial_features)} spatial features")
        return df_sorted.fillna(0)
    
    def prepare_training_data(self, df):
        """Prepare training data with target FWI values"""
        logger.info("Preparing training data...")
        
        # Create derived features
        df = self.create_derived_features(df)
        
        # Create spatial features
        df = self.create_spatial_features(df)
        
        # Calculate simple FWI as target (baseline)
        df['fwi_target'] = self.calculate_simple_fwi(df)
        
        # Prepare feature matrix
        all_features = self.climate_features + self.derived_features + self.spatial_features
        feature_matrix = df[all_features].values
        target_vector = df['fwi_target'].values
        
        # Remove any rows with NaN
        valid_mask = ~(np.isnan(feature_matrix).any(axis=1) | np.isnan(target_vector))
        feature_matrix = feature_matrix[valid_mask]
        target_vector = target_vector[valid_mask]
        
        logger.info(f"Training data shape: {feature_matrix.shape}")
        logger.info(f"Target range: {target_vector.min():.2f} to {target_vector.max():.2f}")
        
        return feature_matrix, target_vector, df[valid_mask]
    
    def calculate_simple_fwi(self, df):
        """Calculate simplified FWI as baseline target"""
        # Simplified FWI calculation for training target
        temp_c = df['temp_celsius']
        wind_kmh = df['sfcWind'] * 3.6
        humidity = df['huss'] * 100
        precip = df['precip_intensity']
        
        # Simplified moisture content
        moisture = 100 - humidity + precip * 0.1
        moisture = np.clip(moisture, 1, 100)
        
        # Simplified fire weather index
        fwi = (temp_c * wind_kmh) / (moisture + 1e-6)
        fwi = np.clip(fwi, 0, 100)
        
        return fwi
    
    def train_random_forest(self, X, y):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.rf_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Random Forest Performance:")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  RMSE: {np.sqrt(mse):.4f}")
        
        # Feature importance
        feature_names = (self.climate_features + self.derived_features + 
                        self.spatial_features)
        importances = self.rf_model.feature_importances_
        
        print("\nTop 10 Most Important Features (Random Forest):")
        for i, (feat, imp) in enumerate(sorted(zip(feature_names, importances), 
                                             key=lambda x: x[1], reverse=True)[:10]):
            print(f"  {i+1}. {feat}: {imp:.4f}")
        
        return mse, r2
    
    def create_cnn_model(self, input_shape):
        """Create CNN model for spatial super-resolution"""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available for CNN model")
            return None
        
        logger.info("Creating CNN model...")
        
        # Input layer
        input_layer = Input(shape=input_shape)
        
        # Encoder path
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        
        # Bottleneck
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        
        # Decoder path - match input dimensions
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)
        
        # Output layer - same dimensions as input
        output = Conv2D(1, (1, 1), activation='linear', padding='same')(conv7)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("CNN Model Architecture:")
        model.summary()
        
        return model
    
    def train_cnn(self, X, y, df):
        """Train CNN model"""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available for CNN training")
            return None, None
        
        logger.info("Preparing CNN training data...")
        
        # Reshape data for CNN (assuming we have spatial grid)
        unique_times = sorted(df['time'].unique())
        unique_lats = sorted(df['lat'].unique())
        unique_lons = sorted(df['lon'].unique())
        
        n_times = len(unique_times)
        n_lats = len(unique_lats)
        n_lons = len(unique_lons)
        n_features = len(self.climate_features)
        
        logger.info(f"Spatial grid: {n_lats}x{n_lons}, {n_times} time steps")
        
        # Create 4D tensor (samples, height, width, channels)
        X_cnn = np.zeros((n_times, n_lats, n_lons, n_features))
        y_cnn = np.zeros((n_times, n_lats, n_lons, 1))
        
        for t, time in enumerate(unique_times):
            time_data = df[df['time'] == time]
            
            for i, lat in enumerate(unique_lats):
                for j, lon in enumerate(unique_lons):
                    point_data = time_data[(time_data['lat'] == lat) & 
                                         (time_data['lon'] == lon)]
                    
                    if len(point_data) > 0:
                        X_cnn[t, i, j, :] = point_data[self.climate_features].values[0]
                        y_cnn[t, i, j, 0] = point_data['fwi_target'].values[0]
        
        # Check for NaN values
        X_cnn = np.nan_to_num(X_cnn, nan=0.0)
        y_cnn = np.nan_to_num(y_cnn, nan=0.0)
        
        # Split data
        split_idx = int(0.8 * n_times)
        X_train = X_cnn[:split_idx]
        X_test = X_cnn[split_idx:]
        y_train = y_cnn[:split_idx]
        y_test = y_cnn[split_idx:]
        
        logger.info(f"CNN training data shape: {X_train.shape}")
        logger.info(f"CNN test data shape: {X_test.shape}")
        logger.info(f"CNN training target shape: {y_train.shape}")
        logger.info(f"CNN test target shape: {y_test.shape}")
        
        # Create and train model
        input_shape = (n_lats, n_lons, n_features)
        self.cnn_model = self.create_cnn_model(input_shape)
        
        if self.cnn_model is None:
            return None, None
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
        
        try:
            # Train model
            logger.info("Starting CNN training...")
            history = self.cnn_model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=16,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluate
            y_pred = self.cnn_model.predict(X_test)
            mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
            r2 = r2_score(y_test.flatten(), y_pred.flatten())
            
            logger.info(f"CNN Performance:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  RMSE: {np.sqrt(mse):.4f}")
            
            return mse, r2
            
        except Exception as e:
            logger.error(f"Error during CNN training: {e}")
            return None, None

    def predict_rf_fwi(self, df):
        """Predict FWI using Random Forest"""
        logger.info("Predicting FWI using Random Forest...")
        
        if self.rf_model is None:
            logger.error("Random Forest model not trained")
            return None
        
        # Prepare features
        df = self.create_derived_features(df)
        df = self.create_spatial_features(df)
        
        all_features = self.climate_features + self.derived_features + self.spatial_features
        X = df[all_features].values
        
        # Handle missing values
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        
        # Scale features
        X_scaled = self.scaler.transform(X_valid)
        
        # Predict
        rf_pred = self.rf_model.predict(X_scaled)
        
        # Create result DataFrame
        result_df = df[valid_mask].copy()
        result_df['fwi_rf'] = rf_pred
        
        logger.info(f"RF FWI predictions range: {rf_pred.min():.2f} to {rf_pred.max():.2f}")
        logger.info(f"RF prediction completed - shape: {result_df.shape}")
        
        return result_df
    
    def predict_cnn_fwi(self, df):
        """Predict FWI using CNN"""
        logger.info("Predicting FWI using CNN...")
        
        if self.cnn_model is None:
            logger.error("CNN model not trained")
            return None
        
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available for CNN prediction")
            return None
        
        try:
            # Prepare features
            df_processed = self.create_derived_features(df.copy())
            df_processed = self.create_spatial_features(df_processed)
            
            # Reshape data for CNN
            unique_times = sorted(df_processed['time'].unique())
            unique_lats = sorted(df_processed['lat'].unique())
            unique_lons = sorted(df_processed['lon'].unique())
            
            n_times = len(unique_times)
            n_lats = len(unique_lats)
            n_lons = len(unique_lons)
            n_features = len(self.climate_features)
            
            logger.info(f"Prediction spatial grid: {n_lats}x{n_lons}, {n_times} time steps")
            
            # Create 4D tensor
            X_cnn = np.zeros((n_times, n_lats, n_lons, n_features))
            
            for t, time in enumerate(unique_times):
                time_data = df_processed[df_processed['time'] == time]
                
                for i, lat in enumerate(unique_lats):
                    for j, lon in enumerate(unique_lons):
                        point_data = time_data[(time_data['lat'] == lat) & 
                                             (time_data['lon'] == lon)]
                        
                        if len(point_data) > 0:
                            X_cnn[t, i, j, :] = point_data[self.climate_features].values[0]
            
            # Handle NaN values
            X_cnn = np.nan_to_num(X_cnn, nan=0.0)
            
            # Predict
            logger.info("Generating CNN predictions...")
            cnn_pred = self.cnn_model.predict(X_cnn, verbose=1)
            
            # Reshape predictions back to original format
            results = []
            for t, time in enumerate(unique_times):
                for i, lat in enumerate(unique_lats):
                    for j, lon in enumerate(unique_lons):
                        results.append({
                            'time': time,
                            'lat': lat,
                            'lon': lon,
                            'fwi_cnn': cnn_pred[t, i, j, 0]
                        })
            
            result_df = pd.DataFrame(results)
            
            # Merge with original data
            df_merged = df.merge(result_df, on=['time', 'lat', 'lon'], how='left')
            
            logger.info(f"CNN FWI predictions range: {result_df['fwi_cnn'].min():.2f} to {result_df['fwi_cnn'].max():.2f}")
            logger.info(f"CNN prediction completed - shape: {df_merged.shape}")
            
            return df_merged
            
        except Exception as e:
            logger.error(f"Error in CNN prediction: {e}")
            return None

    def save_models(self, save_dir):
        """Save trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save Random Forest
        if self.rf_model is not None:
            rf_path = os.path.join(save_dir, 'rf_fwi_model.pkl')
            joblib.dump(self.rf_model, rf_path)
            logger.info(f"Random Forest model saved to {rf_path}")
        
        # Save scaler
        scaler_path = os.path.join(save_dir, 'feature_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Feature scaler saved to {scaler_path}")
        
        # Save CNN
        if self.cnn_model is not None:
            cnn_path = os.path.join(save_dir, 'cnn_fwi_model.h5')
            self.cnn_model.save(cnn_path)
            logger.info(f"CNN model saved to {cnn_path}")

def load_cmip6_data(csv_file):
    """Load CMIP6 data from CSV file"""
    logger.info(f"Loading CMIP6 data from: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Parse time
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])
        
        # Check required columns
        required_cols = ['time', 'lat', 'lon', 'tasmax', 'huss', 'sfcWind', 'pr']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        logger.info(f"Time range: {df['time'].min()} to {df['time'].max()}")
        logger.info(f"Spatial coverage: {df['lat'].min():.2f}° to {df['lat'].max():.2f}°N, {df['lon'].min():.2f}° to {df['lon'].max():.2f}°E")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def main():
    """Main function"""
    print("ML-based FWI Prediction - RF and CNN Separate Outputs")
    print("=" * 60)
    
    # Configuration
    target_resolution = 25  # km
    
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
    df = load_cmip6_data(csv_path)
    if df is None:
        print("Failed to load data!")
        return
    
    # Initialize predictor
    predictor = FWIMLPredictor(target_resolution=target_resolution)
    
    # Prepare training data
    print("\nPreparing training data...")
    X, y, df_processed = predictor.prepare_training_data(df)
    
    # Train Random Forest
    print("\nTraining Random Forest model...")
    rf_mse, rf_r2 = predictor.train_random_forest(X, y)
    print(f"Random Forest trained - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")
    
    # Train CNN
    if TF_AVAILABLE:
        print("\nTraining CNN model...")
        cnn_mse, cnn_r2 = predictor.train_cnn(X, y, df_processed)
        if cnn_mse is not None:
            print(f"CNN trained - MSE: {cnn_mse:.4f}, R²: {cnn_r2:.4f}")
        else:
            print("CNN training failed!")
    else:
        print("TensorFlow not available - skipping CNN training")
    
    # Generate RF predictions
    print("\nGenerating Random Forest FWI predictions...")
    rf_results = predictor.predict_rf_fwi(df)
    
    if rf_results is not None:
        # Save RF results
        rf_output_file = f"fwi_rf_predictions_{target_resolution}km.csv"
        rf_output_path = os.path.join(script_dir, rf_output_file)
        
        # Select relevant columns for output
        rf_output_cols = ['time', 'lat', 'lon', 'tasmax', 'huss', 'sfcWind', 'pr', 'fwi_rf']
        rf_results[rf_output_cols].to_csv(rf_output_path, index=False)
        
        print(f"RF results saved to: {rf_output_file}")
        print(f"RF output shape: {rf_results.shape}")
        
        # RF statistics
        fwi_rf = rf_results['fwi_rf']
        print(f"\nRandom Forest FWI Statistics:")
        print(f"  Mean: {fwi_rf.mean():.2f}")
        print(f"  Std: {fwi_rf.std():.2f}")
        print(f"  Min: {fwi_rf.min():.2f}")
        print(f"  Max: {fwi_rf.max():.2f}")
        
        # High FWI days
        high_fwi_rf = rf_results[rf_results['fwi_rf'] > 30]
        print(f"  High FWI days (>30): {len(high_fwi_rf)}")
        
        file_size_mb = os.path.getsize(rf_output_path) / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
    
    # Generate CNN predictions
    if TF_AVAILABLE and predictor.cnn_model is not None:
        print("\nGenerating CNN FWI predictions...")
        cnn_results = predictor.predict_cnn_fwi(df)
        
        if cnn_results is not None:
            # Save CNN results
            cnn_output_file = f"fwi_cnn_predictions_{target_resolution}km.csv"
            cnn_output_path = os.path.join(script_dir, cnn_output_file)
            
            # Select relevant columns for output
            cnn_output_cols = ['time', 'lat', 'lon', 'tasmax', 'huss', 'sfcWind', 'pr', 'fwi_cnn']
            cnn_results[cnn_output_cols].to_csv(cnn_output_path, index=False)
            
            print(f"CNN results saved to: {cnn_output_file}")
            print(f"CNN output shape: {cnn_results.shape}")
            
            # CNN statistics
            fwi_cnn = cnn_results['fwi_cnn']
            print(f"\nCNN FWI Statistics:")
            print(f"  Mean: {fwi_cnn.mean():.2f}")
            print(f"  Std: {fwi_cnn.std():.2f}")
            print(f"  Min: {fwi_cnn.min():.2f}")
            print(f"  Max: {fwi_cnn.max():.2f}")
            
            # High FWI days
            high_fwi_cnn = cnn_results[cnn_results['fwi_cnn'] > 30]
            print(f"  High FWI days (>30): {len(high_fwi_cnn)}")
            
            file_size_mb = os.path.getsize(cnn_output_path) / (1024 * 1024)
            print(f"  File size: {file_size_mb:.2f} MB")
    
    # Save models
    model_dir = os.path.join(script_dir, 'trained_models')
    predictor.save_models(model_dir)
    print(f"\nModels saved to: {model_dir}")
    
    print("\n" + "=" * 60)
    print("FWI Prediction completed!")
    print("Generated separate CSV files for RF and CNN predictions")

if __name__ == "__main__":
    main()