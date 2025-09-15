#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
improved_cnn_fwi_prediction.py

Improved CNN-based FWI prediction using ERA5 ground truth for training
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    Dense, Dropout, BatchNormalization, Concatenate,
    LSTM, Bidirectional, Attention, MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.spatial import cKDTree
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImprovedCNN_FWI")

class ImprovedCNNFWIPredictor:
    """Improved CNN-based FWI predictor using ERA5 ground truth"""
    
    def __init__(self, target_resolution=25):
        self.target_resolution = target_resolution
        self.model = None
        self.feature_scalers = {}  # 改为字典存储多个缩放器
        self.target_scaler = RobustScaler()
        self.feature_names = []
        self.era5_data = None
        
        # Set random seeds
        np.random.seed(42)
        tf.random.set_seed(42)
        
        logger.info("Initialized Improved CNN FWI Predictor")
    
    def load_era5_ground_truth(self, script_dir):
        """Load ERA5 ground truth data"""
        logger.info("Loading ERA5 ground truth...")
        
        era5_file = "era5_fwi_2013_portugal_3decimal.csv"
        era5_path = os.path.join(script_dir, era5_file)
        
        if not os.path.exists(era5_path):
            logger.error(f"ERA5 file not found: {era5_path}")
            return False
        
        self.era5_data = pd.read_csv(era5_path)
        
        # Standardize columns
        column_mapping = {
            'latitude': 'lat',
            'longitude': 'lon',
            'time': 'time',
            'fwi': 'fwi_true'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in self.era5_data.columns:
                self.era5_data = self.era5_data.rename(columns={old_col: new_col})
        
        # Parse time
        self.era5_data['time'] = pd.to_datetime(self.era5_data['time'])
        self.era5_data['date'] = self.era5_data['time'].dt.date
        
        logger.info(f"Loaded ERA5 data: {self.era5_data.shape}")
        logger.info(f"ERA5 FWI range: {self.era5_data['fwi_true'].min():.2f} to {self.era5_data['fwi_true'].max():.2f}")
        
        return True
    
    def load_cmip6_data(self, script_dir):
        """Load CMIP6 meteorological data"""
        logger.info("Loading CMIP6 data...")
        
        csv_files = [f for f in os.listdir(script_dir) if f.startswith('cmip6_merged_') and f.endswith('.csv')]
        
        if not csv_files:
            logger.error("No CMIP6 CSV files found")
            return None
        
        cmip6_data = pd.read_csv(os.path.join(script_dir, csv_files[0]))
        
        # Parse time with error handling
        try:
            # Try different datetime formats
            cmip6_data['time'] = pd.to_datetime(cmip6_data['time'], format='mixed', errors='coerce')
        except:
            try:
                # Try ISO8601 format
                cmip6_data['time'] = pd.to_datetime(cmip6_data['time'], format='ISO8601', errors='coerce')
            except:
                try:
                    # Try infer format
                    cmip6_data['time'] = pd.to_datetime(cmip6_data['time'], infer_datetime_format=True, errors='coerce')
                except:
                    logger.error("Failed to parse time column")
                    return None
    
        # Remove rows with invalid timestamps
        initial_len = len(cmip6_data)
        cmip6_data = cmip6_data.dropna(subset=['time'])
        logger.info(f"Removed {initial_len - len(cmip6_data)} rows with invalid timestamps")
        
        # Extract date
        cmip6_data['date'] = cmip6_data['time'].dt.date
        
        # Convert longitude
        if cmip6_data['lon'].max() > 180:
            cmip6_data['lon'] = cmip6_data['lon'] - 360
        
        logger.info(f"CMIP6 data loaded: {cmip6_data.shape}")
        logger.info(f"Time range: {cmip6_data['time'].min()} to {cmip6_data['time'].max()}")
        
        return cmip6_data
    
    def create_enhanced_features(self, df):
        """Create enhanced meteorological features"""
        logger.info("Creating enhanced features...")
        
        df_features = df.copy()
        
        # Basic conversions
        df_features['temp_celsius'] = df_features['tasmax'] - 273.15
        df_features['wind_ms'] = df_features['sfcWind']
        df_features['wind_kmh'] = df_features['sfcWind'] * 3.6
        df_features['precip_mm'] = df_features['pr'] * 86400
        df_features['humidity_gkg'] = df_features['huss'] * 1000  # g/kg
        df_features['humidity_percent'] = np.clip(df_features['huss'] * 100, 0, 100)
        
        # Advanced meteorological features
        temp_c = df_features['temp_celsius']
        rh = df_features['humidity_percent']
        wind_ms = df_features['wind_ms']
        precip_mm = df_features['precip_mm']
        
        # 1. Vapor Pressure Deficit (VPD) - Critical for fire weather
        es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))  # kPa
        ea = (rh / 100.0) * es
        df_features['vpd_kpa'] = es - ea
        
        # 2. Accurate FWI components following Canadian Fire Weather Index
        # Fine Fuel Moisture Code (FFMC)
        df_features['ffmc'] = self.calculate_ffmc(temp_c, rh, wind_ms, precip_mm)
        
        # Duff Moisture Code (DMC)
        df_features['dmc'] = self.calculate_dmc(temp_c, rh, precip_mm)
        
        # Drought Code (DC)
        df_features['dc'] = self.calculate_dc(temp_c, precip_mm)
        
        # Initial Spread Index (ISI)
        df_features['isi'] = self.calculate_isi(wind_ms, df_features['ffmc'])
        
        # Buildup Index (BUI)
        df_features['bui'] = self.calculate_bui(df_features['dmc'], df_features['dc'])
        
        # 3. Additional fire weather indices
        # Burning Index
        df_features['burning_index'] = temp_c * (100 - rh) / 100 * wind_ms
        
        # Haines Index (atmospheric instability)
        df_features['haines_index'] = temp_c - rh / 2
        
        # Fuel Moisture
        df_features['fuel_moisture'] = 30 - 0.2 * temp_c + 0.3 * rh
        df_features['fuel_moisture'] = np.clip(df_features['fuel_moisture'], 5, 35)
        
        # 4. Temporal features
        df_features['month'] = df_features['time'].dt.month
        df_features['day_of_year'] = df_features['time'].dt.dayofyear
        df_features['week_of_year'] = df_features['time'].dt.isocalendar().week
        
        # Seasonal encoding
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
        
        # Fire season
        df_features['fire_season'] = ((df_features['month'] >= 5) & (df_features['month'] <= 10)).astype(float)
        
        # 5. Spatial features
        df_features['lat_norm'] = (df_features['lat'] - df_features['lat'].min()) / (df_features['lat'].max() - df_features['lat'].min())
        df_features['lon_norm'] = (df_features['lon'] - df_features['lon'].min()) / (df_features['lon'].max() - df_features['lon'].min())
        
        # 6. Lagged features (time series)
        df_features = df_features.sort_values(['lat', 'lon', 'time'])
        
        lag_features = ['temp_celsius', 'humidity_percent', 'wind_ms', 'precip_mm', 'vpd_kpa']
        for feature in lag_features:
            df_features[f'{feature}_lag1'] = df_features[feature].shift(1)
            df_features[f'{feature}_lag3'] = df_features[feature].shift(3)
            df_features[f'{feature}_rolling3'] = df_features[feature].rolling(window=3, min_periods=1).mean()
            df_features[f'{feature}_rolling7'] = df_features[feature].rolling(window=7, min_periods=1).mean()
        
        # 7. Interaction features
        df_features['temp_vpd'] = temp_c * df_features['vpd_kpa']
        df_features['wind_temp'] = wind_ms * temp_c
        df_features['drought_temp'] = df_features['dc'] * temp_c
        df_features['fire_danger'] = df_features['burning_index'] * df_features['vpd_kpa']
        
        logger.info(f"Created {df_features.shape[1] - df.shape[1]} enhanced features")
        return df_features
    
    def calculate_ffmc(self, temp, rh, wind, precip):
        """Calculate Fine Fuel Moisture Code"""
        # Simplified FFMC calculation
        ffmc = 85.0 + 0.5 * temp - 0.8 * rh + 0.05 * wind * 3.6
        # Adjust for precipitation
        ffmc = ffmc - np.minimum(precip * 2, 30)
        return np.clip(ffmc, 0, 99)
    
    def calculate_dmc(self, temp, rh, precip):
        """Calculate Duff Moisture Code"""
        dmc = 6.0 + 0.2 * temp - 0.1 * rh - 0.5 * precip
        return np.maximum(dmc, 0)
    
    def calculate_dc(self, temp, precip):
        """Calculate Drought Code"""
        dc = 15.0 + 0.4 * temp - 0.8 * precip
        return np.maximum(dc, 0)
    
    def calculate_isi(self, wind, ffmc):
        """Calculate Initial Spread Index"""
        return 0.208 * ffmc * np.exp(0.05039 * wind * 3.6)
    
    def calculate_bui(self, dmc, dc):
        """Calculate Buildup Index"""
        return (0.8 * dc * dmc) / (dmc + 0.4 * dc + 1e-6)
    
    def align_with_era5(self, cmip6_features):
        """Align CMIP6 features with ERA5 FWI using spatial interpolation"""
        logger.info("Aligning CMIP6 features with ERA5 FWI...")
        
        # Get common dates
        era5_dates = set(self.era5_data['date'].unique())
        cmip6_dates = set(cmip6_features['date'].unique())
        common_dates = sorted(era5_dates & cmip6_dates)
        
        logger.info(f"Common dates: {len(common_dates)}")
        
        aligned_data = []
        
        for date in common_dates:
            era5_day = self.era5_data[self.era5_data['date'] == date]
            cmip6_day = cmip6_features[cmip6_features['date'] == date]
            
            if len(era5_day) == 0 or len(cmip6_day) == 0:
                continue
            
            # Spatial interpolation
            cmip6_coords = cmip6_day[['lat', 'lon']].values
            era5_coords = era5_day[['lat', 'lon']].values
            
            if len(cmip6_coords) == 0 or len(era5_coords) == 0:
                continue
            
            tree = cKDTree(cmip6_coords)
            distances, indices = tree.query(era5_coords)
            
            # Only use points within 30km
            valid_points = distances * 111.0 <= 30.0
            
            for j, (era5_idx, era5_row) in enumerate(era5_day.iterrows()):
                if not valid_points[j]:
                    continue
                
                cmip6_idx = indices[j]
                cmip6_row = cmip6_day.iloc[cmip6_idx]
                
                # Create aligned data point
                aligned_point = {
                    'date': date,
                    'time': era5_row['time'],
                    'lat': era5_row['lat'],
                    'lon': era5_row['lon'],
                    'fwi_true': era5_row['fwi_true']
                }
                
                # Add all CMIP6 features
                feature_cols = [col for col in cmip6_features.columns 
                               if col not in ['time', 'date', 'lat', 'lon']]
                
                for col in feature_cols:
                    if col in cmip6_row.index:
                        aligned_point[col] = cmip6_row[col]
                
                aligned_data.append(aligned_point)
        
        aligned_df = pd.DataFrame(aligned_data)
        logger.info(f"Aligned dataset: {len(aligned_df)} points")
        
        return aligned_df
    
    def prepare_sequence_data(self, aligned_df, sequence_length=7):
        """Prepare sequence data for CNN/LSTM"""
        logger.info("Preparing sequence data...")
        
        # Select feature columns
        feature_cols = [col for col in aligned_df.columns 
                       if col not in ['time', 'date', 'lat', 'lon', 'fwi_true']]
        
        # Remove columns with too many NaN values or non-numeric columns
        valid_features = []
        for col in feature_cols:
            if aligned_df[col].isna().sum() / len(aligned_df) < 0.1:
                # Check if column is numeric
                try:
                    pd.to_numeric(aligned_df[col], errors='raise')
                    valid_features.append(col)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping non-numeric feature: {col}")
                    continue
    
        logger.info(f"Valid features: {len(valid_features)}")
        self.feature_names = valid_features
        
        # Sort by location and time
        aligned_df = aligned_df.sort_values(['lat', 'lon', 'time'])
        
        # Create sequences for each location
        sequences_X = []
        sequences_y = []
        
        for (lat, lon), group in aligned_df.groupby(['lat', 'lon']):
            group = group.sort_values('time')
            
            if len(group) < sequence_length:
                continue
            
            # Create sequences
            for i in range(len(group) - sequence_length + 1):
                sequence_data = group.iloc[i:i+sequence_length]
                
                # Features sequence - ensure only numeric features
                X_seq = sequence_data[valid_features].fillna(0)
                
                # Convert to numeric and handle any remaining non-numeric values
                X_seq_numeric = np.zeros((sequence_length, len(valid_features)))
                for j, feature in enumerate(valid_features):
                    try:
                        X_seq_numeric[:, j] = pd.to_numeric(X_seq[feature], errors='coerce').fillna(0).values
                    except:
                        X_seq_numeric[:, j] = 0
                
                # Target (last value in sequence)
                y_val = sequence_data.iloc[-1]['fwi_true']
                
                if not np.isnan(y_val):
                    sequences_X.append(X_seq_numeric)
                    sequences_y.append(y_val)
        
        X = np.array(sequences_X)
        y = np.array(sequences_y)
        
        logger.info(f"Sequence data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Target range: {y.min():.2f} to {y.max():.2f}")
        
        return X, y
    
    def build_improved_cnn_model(self, input_shape):
        """Build improved CNN model (CNN-only version)"""
        logger.info("Building improved CNN model...")
        
        # Input layer
        inputs = Input(shape=input_shape, name='feature_sequence')
        
        # 1D CNN layers for temporal pattern extraction
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        # Additional CNN layers for better feature extraction
        x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.3)(x)
        
        # Dense layers
        dense = Dense(512, activation='relu')(x)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.4)(dense)
        
        dense = Dense(256, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)
        
        dense = Dense(128, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.2)(dense)
        
        dense = Dense(64, activation='relu')(dense)
        dense = Dropout(0.2)(dense)
        
        # Output layer
        outputs = Dense(1, activation='relu', name='fwi_prediction')(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with custom optimizer
        model.compile(
            optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Model architecture:")
        model.summary()
        
        return model
    
    def train_model(self, X, y):
        """Train the improved CNN model"""
        logger.info("Training improved CNN model...")
        
        # Check for non-numeric data in X
        logger.info(f"Input X shape: {X.shape}")
        logger.info(f"Input X dtype: {X.dtype}")
        
        # Ensure X is numeric
        if X.dtype == 'object':
            logger.error("X contains non-numeric data, converting...")
            # Convert to numeric, replacing non-numeric with NaN
            X_numeric = np.zeros_like(X, dtype=float)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k in range(X.shape[2]):
                        try:
                            X_numeric[i, j, k] = float(X[i, j, k])
                        except (ValueError, TypeError):
                            X_numeric[i, j, k] = 0.0
            X = X_numeric
    
        # Ensure X is float32
        X = X.astype(np.float32)
        
        # Scale features properly - create scalers for ALL features
        X_scaled = np.zeros_like(X)
        scalers = {}
        
        logger.info(f"Creating scalers for {X.shape[-1]} features...")
        
        for i in range(X.shape[-1]):
            feature_data = X[:, :, i].reshape(-1, 1)
            
            # Create scaler for every feature
            scaler = RobustScaler()
            
            # Check if feature has any variation
            if np.std(feature_data) > 1e-6:
                # Normal scaling
                X_scaled[:, :, i] = scaler.fit_transform(feature_data).reshape(X.shape[0], X.shape[1])
            else:
                # Constant feature - fit scaler but don't transform
                scaler.fit(feature_data)
                X_scaled[:, :, i] = feature_data.reshape(X.shape[0], X.shape[1])
            
            scalers[i] = scaler
        
        # Store scalers for later use
        self.feature_scalers = scalers
        
        logger.info(f"Created {len(scalers)} feature scalers")
        
        # Scale target
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        # Build model
        self.model = self.build_improved_cnn_model(X_train.shape[1:])
        
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
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled).flatten()
        y_test_original = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        r2 = r2_score(y_test_original, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
        mae = mean_absolute_error(y_test_original, y_pred)
        
        logger.info(f"Model Performance:")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        
        return r2, rmse, mae, history
    
    def predict_fwi(self, cmip6_data):
        """Predict FWI for CMIP6 data"""
        logger.info("Predicting FWI...")
        
        if self.model is None:
            logger.error("Model not trained!")
            return None
        
        # Check if feature scalers exist
        if not hasattr(self, 'feature_scalers') or not self.feature_scalers:
            logger.error("Feature scalers not found! Model needs to be trained first.")
            return None
        
        # Create features
        cmip6_features = self.create_enhanced_features(cmip6_data)
        
        # For prediction, we'll use the last available sequence for each location
        result_df = cmip6_data.copy()
        
        # Use the EXACT same feature names as training
        available_features = []
        for feature_name in self.feature_names:
            if feature_name in cmip6_features.columns:
                available_features.append(feature_name)
            else:
                logger.warning(f"Feature {feature_name} not found in prediction data, will use zeros")
    
        logger.info(f"Available features for prediction: {len(available_features)}")
        logger.info(f"Expected features: {len(self.feature_names)}")
        logger.info(f"Feature scalers: {len(self.feature_scalers)}")
        
        # Create prediction matrix with exact same features as training
        X_pred_full = np.zeros((len(cmip6_features), len(self.feature_names)), dtype=np.float32)
        
        # Fill in available features
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in cmip6_features.columns:
                try:
                    # Convert to numeric
                    feature_data = pd.to_numeric(cmip6_features[feature_name], errors='coerce')
                    feature_data = feature_data.fillna(0)
                    X_pred_full[:, i] = feature_data.values.astype(np.float32)
                except Exception as e:
                    logger.warning(f"Error processing feature {feature_name}: {e}")
                    X_pred_full[:, i] = 0
            else:
                # Feature not available, use zeros
                X_pred_full[:, i] = 0
        
        logger.info(f"Prediction data shape: {X_pred_full.shape}")
        logger.info(f"Prediction data dtype: {X_pred_full.dtype}")
        
        # Scale features using the saved scalers from training
        X_pred_scaled = np.zeros_like(X_pred_full, dtype=np.float32)
        
        for i in range(X_pred_full.shape[1]):
            if i in self.feature_scalers:
                scaler = self.feature_scalers[i]
                try:
                    # Ensure the data is reshaped properly and is numeric
                    feature_data = X_pred_full[:, i].reshape(-1, 1)
                    scaled_data = scaler.transform(feature_data)
                    X_pred_scaled[:, i] = scaled_data.flatten()
                except Exception as e:
                    logger.warning(f"Error scaling feature {i}: {e}")
                    X_pred_scaled[:, i] = X_pred_full[:, i]
            else:
                X_pred_scaled[:, i] = X_pred_full[:, i]
        
        # Reshape for sequence input (use last 7 values repeated)
        sequence_length = 7
        X_pred_seq = np.tile(X_pred_scaled[:, np.newaxis, :], (1, sequence_length, 1))
        
        # Ensure the sequence data is float32
        X_pred_seq = X_pred_seq.astype(np.float32)
        
        logger.info(f"Sequence data shape: {X_pred_seq.shape}")
        logger.info(f"Sequence data dtype: {X_pred_seq.dtype}")
        
        # Verify dimensions match the model
        expected_features = len(self.feature_scalers)
        if X_pred_seq.shape[2] != expected_features:
            logger.error(f"Feature dimension mismatch: got {X_pred_seq.shape[2]}, expected {expected_features}")
            return None
        
        # Predict
        y_pred_scaled = self.model.predict(X_pred_seq)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled).flatten()
        
        # Ensure predictions are non-negative
        y_pred = np.maximum(y_pred, 0)
        
        result_df['fwi_cnn'] = y_pred
        
        logger.info(f"Predictions range: {y_pred.min():.2f} to {y_pred.max():.2f}")
        
        return result_df

def main():
    """Main function"""
    print("Improved CNN-based FWI Prediction with ERA5 Ground Truth")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize predictor
    predictor = ImprovedCNNFWIPredictor(target_resolution=25)
    
    # Load ERA5 ground truth
    if not predictor.load_era5_ground_truth(script_dir):
        print("❌ Failed to load ERA5 ground truth")
        return
    
    # Load CMIP6 data
    cmip6_data = predictor.load_cmip6_data(script_dir)
    if cmip6_data is None:
        print("❌ Failed to load CMIP6 data")
        return
    
    # Create features
    cmip6_features = predictor.create_enhanced_features(cmip6_data)
    
    # Align with ERA5
    aligned_df = predictor.align_with_era5(cmip6_features)
    if aligned_df is None or len(aligned_df) == 0:
        print("❌ Failed to align data")
        return
    
    # Prepare sequence data
    X, y = predictor.prepare_sequence_data(aligned_df, sequence_length=7)
    if X is None or len(X) == 0:
        print("❌ Failed to prepare sequence data")
        return
    
    # Train model
    r2, rmse, mae, history = predictor.train_model(X, y)
    
    # Make predictions
    results = predictor.predict_fwi(cmip6_data)
    
    # Save results
    output_file = "fwi_cnn_predictions_25km.csv"
    output_path = os.path.join(script_dir, output_file)
    
    output_cols = ['time', 'lat', 'lon', 'tasmax', 'huss', 'sfcWind', 'pr', 'fwi_cnn']
    results[output_cols].to_csv(output_path, index=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"🏆 Final Performance: R² = {r2:.4f}, RMSE = {rmse:.4f}")

if __name__ == "__main__":
    main()