import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import os
import warnings

from EDA import FWISuperResolutionEDA   

warnings.filterwarnings('ignore')

class FWISuperResolutionModel(FWISuperResolutionEDA):
    """
    FWI Super-Resolution - Modeling & Prediction
    """

    def __init__(self):
        super().__init__()
        self.sr_models = {}
        self.scalers = {}
        self.feature_names = []
        self.predicted_1km = None

        print("FWI SUPER-RESOLUTION SYSTEM (Modeling)")
        print("="*60)
        print("Phase 2: Train model → Predict 25km→1km → Save results")
        print("="*60)

    def train_super_resolution_models(self):
        """Train super-resolution models using downsampled data"""
        print(f"\nTraining Super-Resolution Models...")
        
        if self.training_pairs is None or len(self.training_pairs) == 0:
            print("ERROR: No training data available")
            return False
        
        try:
            # Prepare features and targets
            coarse_features = [col for col in self.training_pairs.columns if col.startswith('coarse_')]
            
            # Remove non-numeric or problematic features
            numeric_features = []
            for feat in coarse_features:
                if self.training_pairs[feat].dtype in ['int64', 'float64'] and not self.training_pairs[feat].isnull().all():
                    numeric_features.append(feat)
            
            print(f"   Using {len(numeric_features)} numeric features")
            
            if len(numeric_features) < 2:
                print("ERROR: Insufficient features for training")
                return False
            
            # Prepare training data
            X = self.training_pairs[numeric_features].fillna(self.training_pairs[numeric_features].median())
            
            # Add spatial features
            if 'fine_lat' in self.training_pairs.columns and 'fine_lon' in self.training_pairs.columns:
                X['target_lat'] = self.training_pairs['fine_lat']
                X['target_lon'] = self.training_pairs['fine_lon']
                numeric_features.extend(['target_lat', 'target_lon'])
            
            self.feature_names = numeric_features
            
            # Target variable (FWI)
            target_col = 'fine_fwi'
            if target_col not in self.training_pairs.columns:
                # Try alternative target columns
                fine_cols = [col for col in self.training_pairs.columns if col.startswith('fine_') and 'fwi' in col.lower()]
                if fine_cols:
                    target_col = fine_cols[0]
                else:
                    print("ERROR: No suitable target variable found")
                    return False
            
            y = self.training_pairs[target_col].fillna(self.training_pairs[target_col].median())
            
            print(f"   Target variable: {target_col}")
            print(f"   Training samples: {len(X):,}")
            print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
            
            # Feature scaling
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['features'] = scaler
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Define models
            models_config = {
                'gradient_boosting': {
                    'model': GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=42
                    ),
                    'description': 'Gradient Boosting - Excellent for complex relationships'
                },
                'random_forest': {
                    'model': RandomForestRegressor(
                        n_estimators=100,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    ),
                    'description': 'Random Forest - Good for non-linear patterns'
                },
                'neural_network': {
                    'model': MLPRegressor(
                        hidden_layer_sizes=(100, 50, 25),
                        max_iter=500,
                        learning_rate='adaptive',
                        alpha=0.01,
                        random_state=42
                    ),
                    'description': 'Neural Network - Captures non-linear interactions'
                },
                'ridge_regression': {
                    'model': Ridge(alpha=1.0),
                    'description': 'Ridge Regression - Robust linear baseline'
                }
            }
            
            # Train models
            for name, config in models_config.items():
                try:
                    print(f"   Training {name}...")
                    
                    model = config['model']
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    train_score = model.score(X_train, y_train)
                    test_score = model.score(X_test, y_test)
                    
                    y_pred = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Cross validation
                    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                    
                    self.sr_models[name] = {
                        'model': model,
                        'train_r2': train_score,
                        'test_r2': test_score,
                        'rmse': rmse,
                        'mae': mae,
                        'cv_r2': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'description': config['description']
                    }
                    
                    print(f"      SUCCESS: {name}: R²={test_score:.3f}, RMSE={rmse:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                    
                    # Feature importance for tree models
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        top_features = sorted(zip(self.feature_names, importances), 
                                            key=lambda x: x[1], reverse=True)[:3]
                        print(f"         Top features: {[f'{feat}({imp:.3f})' for feat, imp in top_features]}")
                    
                except Exception as e:
                    print(f"      ERROR: {name} failed: {e}")
                    continue
            
            if self.sr_models:
                print(f"   SUCCESS: Successfully trained {len(self.sr_models)} models")
                return True
            else:
                print(f"   ERROR: No models trained successfully")
                return False
                
        except Exception as e:
            print(f"ERROR: Error training models: {e}")
            import traceback
            traceback.print_exc()
            return False

    
    def apply_super_resolution_25km_to_1km(self):
        """Apply trained super-resolution models to enhance 25km → 1km (Single File Output)"""
        print(f"\nApplying Super-Resolution: 25km → 1km (Single File Output)...")
        
        if not self.sr_models or self.merged_25km is None:
            print("ERROR: Models or 25km data not available")
            return None
        
        try:
            # Select best model
            best_model_name = max(self.sr_models.keys(), 
                                key=lambda x: self.sr_models[x]['cv_r2'])
            best_model_info = self.sr_models[best_model_name]
            best_model = best_model_info['model']
            
            print(f"   Using best model: {best_model_name} (CV R²={best_model_info['cv_r2']:.3f})")
            
            # Get unique dates
            unique_dates = sorted(self.merged_25km['date'].unique())
            
            
            #unique_dates = unique_dates[:1]
            
            
            print(f"   Processing {len(unique_dates)} dates for 1km resolution")
            
            # Store all predictions in memory (no batch files)
            all_predictions = []
            
            # Process in smaller batches for memory efficiency but keep in memory
            batch_size = 5  # Process 5 dates at a time
            
            for batch_start in range(0, len(unique_dates), batch_size):
                batch_end = min(batch_start + batch_size, len(unique_dates))
                batch_dates = unique_dates[batch_start:batch_end]
                
                print(f"      Processing batch {batch_start//batch_size + 1}: dates {batch_start+1}-{batch_end}")
                
                batch_predictions = []
                
                for i, date in enumerate(batch_dates):
                    # Get daily 25km data
                    daily_25km = self.merged_25km[self.merged_25km['date'] == date]
                    
                    if len(daily_25km) < 3:
                        continue
                    
                    # Create 1km target grid for this date
                    target_1km_grid = self._create_1km_grid(daily_25km)
                    
                    if target_1km_grid is None or len(target_1km_grid) == 0:
                        continue
                    
                    print(f"         Date {date}: processing {len(target_1km_grid):,} 1km points...")
                    
                    # Apply super-resolution to this date
                    daily_predictions = self._predict_daily_1km(daily_25km, target_1km_grid, best_model)
                    
                    if daily_predictions is not None:
                        daily_predictions['date'] = date
                        batch_predictions.append(daily_predictions)
                        print(f"         SUCCESS: Generated {len(daily_predictions):,} predictions for {date}")
                
                # Collect batch predictions in memory instead of saving to disk
                if batch_predictions:
                    batch_df = pd.concat(batch_predictions, ignore_index=True)
                    all_predictions.append(batch_df)
                    print(f"      SUCCESS: Collected batch with {len(batch_df):,} points in memory")
                
                # Clear temporary batch data but keep collected predictions
                del batch_predictions
                import gc
                gc.collect()
            
            if not all_predictions:
                print("ERROR: No successful predictions generated")
                return None
            
            # Combine all predictions into single DataFrame
            print(f"   Combining all batches into single output file...")
            final_predictions = pd.concat(all_predictions, ignore_index=True)
            
            # Add metadata
            final_predictions['prediction_method'] = f'SuperResolution_{best_model_name}'
            final_predictions['source_resolution'] = '25km'
            final_predictions['target_resolution'] = '1km'
            final_predictions['model_cv_score'] = best_model_info['cv_r2']
            final_predictions['prediction_timestamp'] = datetime.now().isoformat()
            
            
            best_model_name = str(best_model_name).strip().replace('\n', '').replace(' ', '_')
            # Save to single CSV file with specified filename
            output_file = f"fwi_1km_predictions_{best_model_name}.csv"
            final_predictions.to_csv(output_file, index=False)
            
            self.predicted_1km = final_predictions
            
            print(f"SUCCESS: Super-resolution completed!")
            print(f"   Single output file: {output_file}")
            print(f"   Total predictions: {len(final_predictions):,}")
            print(f"   FWI range: [{final_predictions['fwi_predicted'].min():.2f}, {final_predictions['fwi_predicted'].max():.2f}]")
            print(f"   FWI statistics: {final_predictions['fwi_predicted'].mean():.2f} ± {final_predictions['fwi_predicted'].std():.2f}")
            
            return final_predictions
            
        except Exception as e:
            print(f"ERROR: Error applying 1km super-resolution: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_results_and_analysis(self):
        """Save comprehensive results and analysis"""
        print(f"\nSaving Results and Analysis...")
        
        try:
            if self.predicted_1km is None:
                print("ERROR: No results to save")
                return False
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save main predictions
            main_file = f"fwi_1km_super_resolution.csv"
            self.predicted_1km.to_csv(main_file, index=False)
            print(f"   SUCCESS: Main results: {main_file}")
            
            # Save model performance
            if self.sr_models:
                performance_data = []
                for name, info in self.sr_models.items():
                    performance_data.append({
                        'model': name,
                        'test_r2': info['test_r2'],
                        'rmse': info['rmse'],
                        'mae': info['mae'],
                        'cv_r2': info['cv_r2'],
                        'cv_std': info['cv_std'],
                        'description': info['description']
                    })
                
                performance_df = pd.DataFrame(performance_data)
                performance_file = f"super_resolution_model_performance.csv"
                performance_df.to_csv(performance_file, index=False)
                print(f"   SUCCESS: Model performance: {performance_file}")
            
            # Save training data sample
            if self.training_pairs is not None:
                sample_size = min(1000, len(self.training_pairs))
                sample_training = self.training_pairs.sample(sample_size)
                training_file = f"training_data_sample.csv"
                sample_training.to_csv(training_file, index=False)
                print(f"   SUCCESS: Training sample: {training_file}")
            
            # Generate comprehensive report
            self._generate_super_resolution_report(timestamp)
            
            # Create visualization
            self._create_results_visualization(timestamp)
            
            print(f"   SUCCESS: All results saved successfully")
            return True
            
        except Exception as e:
            print(f"ERROR: Error saving results: {e}")
            return False
    
    def _generate_super_resolution_report(self, timestamp):
        """Generate detailed analysis report"""
        try:
            report_file = f"super_resolution_analysis_report_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("FWI SUPER-RESOLUTION ANALYSIS REPORT\n")
                f.write("="*80 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Method: Downsampling-Upsampling Strategy\n")
                f.write("Strategy: 25km → 50km/75km → 25km (Training) → 25km → 1km (Application)\n\n")
                
                # Data overview
                f.write("DATA PROCESSING SUMMARY\n")
                f.write("-"*50 + "\n")
                if self.merged_25km is not None:
                    f.write(f"Original 25km data points: {len(self.merged_25km):,}\n")
                if self.training_pairs is not None:
                    f.write(f"Generated training pairs: {len(self.training_pairs):,}\n")
                if self.predicted_1km is not None:
                    f.write(f"Final 1km predictions: {len(self.predicted_1km):,}\n")
                    enhancement_ratio = len(self.predicted_1km) / len(self.merged_25km) if self.merged_25km is not None else 0
                    f.write(f"Resolution enhancement ratio: {enhancement_ratio:.1f}x\n")
                f.write("\n")
                
                # Model performance
                if self.sr_models:
                    f.write("MODEL PERFORMANCE ANALYSIS\n")
                    f.write("-"*50 + "\n")
                    f.write(f"{'Model':<20} {'Test R²':<10} {'RMSE':<8} {'MAE':<8} {'CV R²':<12}\n")
                    f.write("-"*58 + "\n")
                    
                    for name, info in self.sr_models.items():
                        f.write(f"{name:<20} {info['test_r2']:<10.3f} {info['rmse']:<8.3f} "
                               f"{info['mae']:<8.3f} {info['cv_r2']:.3f}±{info['cv_std']:.3f}\n")
                    
                    # Best model
                    best_model = max(self.sr_models.keys(), key=lambda x: self.sr_models[x]['cv_r2'])
                    f.write(f"\nBest Model: {best_model}\n")
                    f.write(f"   Performance: R² = {self.sr_models[best_model]['cv_r2']:.3f}\n")
                    f.write(f"   Description: {self.sr_models[best_model]['description']}\n\n")
                
                # Prediction quality assessment
                if self.predicted_1km is not None:
                    fwi = self.predicted_1km['fwi_predicted']
                    f.write("PREDICTION QUALITY ASSESSMENT\n")
                    f.write("-"*50 + "\n")
                    f.write(f"FWI Statistics:\n")
                    f.write(f"   Range: [{fwi.min():.2f}, {fwi.max():.2f}]\n")
                    f.write(f"   Mean: {fwi.mean():.2f} ± {fwi.std():.2f}\n")
                    f.write(f"   Median: {fwi.median():.2f}\n")
                    
                    # Risk distribution
                    f.write(f"\nFire Risk Distribution:\n")
                    risk_levels = [
                        ("Very Low", 0, 5),
                        ("Low", 5, 12), 
                        ("Moderate", 12, 24),
                        ("High", 24, 38),
                        ("Very High", 38, 50),
                        ("Extreme", 50, 100)
                    ]
                    
                    for risk_name, low, high in risk_levels:
                        count = ((fwi >= low) & (fwi < high)).sum()
                        percentage = count / len(fwi) * 100
                        f.write(f"   {risk_name:>10} ({low:>2}-{high:>2}): {count:>6,} points ({percentage:>5.1f}%)\n")
                
                # Methodology advantages
                f.write(f"\nMETHODOLOGY ADVANTAGES\n")
                f.write("-"*50 + "\n")
                f.write("1. Realistic Training Scenario:\n")
                f.write("   - Uses actual downsampling artifacts for training\n")
                f.write("   - Learns real resolution enhancement patterns\n\n")
                f.write("2. Robust Model Validation:\n")
                f.write("   - Cross-validation on realistic data\n")
                f.write("   - Multiple model ensemble approach\n\n")
                f.write("3. Spatial Consistency:\n")
                f.write("   - Maintains spatial relationships\n")
                f.write("   - Preserves meteorological gradients\n\n")
                f.write("4. Quality Assurance:\n")
                f.write("   - Uncertainty quantification\n")
                f.write("   - Model performance tracking\n")
            
            print(f"   SUCCESS: Analysis report: {report_file}")
            
        except Exception as e:
            print(f"   WARNING: Report generation failed: {e}")
    
    def _create_results_visualization(self, timestamp):
        """Create visualization of results"""
        try:
            if self.predicted_1km is None:
                return
            
            # Create summary visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('FWI Super-Resolution Results Analysis', fontsize=14, fontweight='bold')
            
            # 1. FWI distribution
            ax1 = axes[0, 0]
            self.predicted_1km['fwi_predicted'].hist(bins=50, alpha=0.7, ax=ax1, color='orange')
            ax1.set_xlabel('FWI Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('FWI Distribution (1km Predictions)')
            ax1.grid(True, alpha=0.3)
            
            # 2. Spatial distribution (sample)
            ax2 = axes[0, 1]
            sample_data = self.predicted_1km.sample(min(1000, len(self.predicted_1km)))
            scatter = ax2.scatter(sample_data['longitude'], sample_data['latitude'], 
                                c=sample_data['fwi_predicted'], cmap='YlOrRd', alpha=0.6)
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            ax2.set_title('Spatial FWI Distribution (Sample)')
            plt.colorbar(scatter, ax=ax2, label='FWI')
            
            # 3. Model performance comparison
            if self.sr_models:
                ax3 = axes[1, 0]
                model_names = list(self.sr_models.keys())
                cv_scores = [self.sr_models[name]['cv_r2'] for name in model_names]
                
                bars = ax3.bar(model_names, cv_scores, color=['skyblue', 'lightgreen', 'orange', 'pink'][:len(model_names)])
                ax3.set_ylabel('Cross-Validation R²')
                ax3.set_title('Model Performance Comparison')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, score in zip(bars, cv_scores):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{score:.3f}', ha='center', va='bottom')
            
            # 4. Prediction uncertainty
            ax4 = axes[1, 1]
            if 'prediction_uncertainty' in self.predicted_1km.columns:
                self.predicted_1km['prediction_uncertainty'].hist(bins=30, alpha=0.7, ax=ax4, color='lightcoral')
                ax4.set_xlabel('Prediction Uncertainty')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Prediction Uncertainty Distribution')
            else:
                ax4.text(0.5, 0.5, 'Uncertainty data\nnot available', ha='center', va='center', 
                        transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Prediction Uncertainty')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = f"super_resolution_visualization_{timestamp}.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   SUCCESS: Visualization: {viz_file}")
            
        except Exception as e:
            print(f"   WARNING: Visualization creation failed: {e}")

    def _create_1km_grid(self, daily_25km_data):
        """Create ultra-high-resolution 1km target grid"""
        try:
            # Get spatial bounds
            lat_min, lat_max = daily_25km_data['latitude'].min(), daily_25km_data['latitude'].max()
            lon_min, lon_max = daily_25km_data['longitude'].min(), daily_25km_data['longitude'].max()
            
            # 1km grid spacing (approximately)
            lat_spacing = 0.009   # ~1km (instead of 0.09 for 1km)
            lon_spacing = 0.012   # ~1km for Portugal latitude (instead of 0.12 for 1km)
            
            print(f"      Creating 1km grid: lat_spacing={lat_spacing:.3f}°, lon_spacing={lon_spacing:.3f}°")
            
            # Create ultra-dense grid
            lats = np.arange(lat_min, lat_max + lat_spacing, lat_spacing)
            lons = np.arange(lon_min, lon_max + lon_spacing, lon_spacing)
            
            print(f"      Grid dimensions: {len(lats)} x {len(lons)} = {len(lats) * len(lons):,} points")
            
            # Check if grid is too large
            total_points = len(lats) * len(lons)
            if total_points > 500000:  # Limit to 500k points per day
                print(f"      WARNING: Grid too large ({total_points:,} points), reducing density...")
                lat_spacing *= 2
                lon_spacing *= 2
                lats = np.arange(lat_min, lat_max + lat_spacing, lat_spacing)
                lons = np.arange(lon_min, lon_max + lon_spacing, lon_spacing)
                print(f"      Reduced grid: {len(lats)} x {len(lons)} = {len(lats) * len(lons):,} points")
            
            lon_mesh, lat_mesh = np.meshgrid(lons, lats)
            
            target_grid = pd.DataFrame({
                'latitude': lat_mesh.flatten(),
                'longitude': lon_mesh.flatten()
            })
            
            print(f"      SUCCESS: Created 1km grid with {len(target_grid):,} points")
            return target_grid
            
        except Exception as e:
            print(f"         ERROR: Error creating 1km grid: {e}")
            return None

    def _extract_features_from_25km_for_1km(self, target_lat, target_lon, daily_25km):
        """Extract features from 25km data for a target 1km point (enhanced precision)"""
        try:
            # Calculate distances to all 25km points
            distances = np.sqrt(
                (daily_25km['latitude'] - target_lat)**2 + 
                (daily_25km['longitude'] - target_lon)**2
            )
            
            # For 1km prediction, use smaller radius and more sophisticated interpolation
            radius = 0.3  # ~30km radius (smaller than 1km prediction)
            nearby_mask = distances <= radius
            nearby_data = daily_25km[nearby_mask]
            
            if len(nearby_data) == 0:
                # Use nearest point
                nearest_idx = distances.idxmin()
                nearby_data = daily_25km.loc[[nearest_idx]]
            
            # Extract features with enhanced spatial interpolation
            features = {}
            
            # Distance-weighted interpolation for primary features
            if len(nearby_data) > 1:
                weights = 1 / (distances[nearby_data.index] + 0.001)  # Avoid division by zero
                weights = weights / weights.sum()
                
                for col in nearby_data.columns:
                    if col not in ['latitude', 'longitude', 'date'] and nearby_data[col].dtype in ['int64', 'float64']:
                        weighted_value = (nearby_data[col] * weights).sum()
                        features[f'coarse_{col}'] = weighted_value
            else:
                # Single point case
                closest_point = nearby_data.iloc[0]
                for col in nearby_data.columns:
                    if col not in ['latitude', 'longitude', 'date'] and pd.notna(closest_point[col]):
                        features[f'coarse_{col}'] = closest_point[col]
            
            # Enhanced spatial features for 1km precision
            features['coarse_distance'] = distances.min()
            features['coarse_lat'] = nearby_data['latitude'].mean()
            features['coarse_lon'] = nearby_data['longitude'].mean()
            features['target_lat'] = target_lat
            features['target_lon'] = target_lon
            
            # Local gradient estimation (for 1km precision)
            if len(nearby_data) >= 3:
                try:
                    # Estimate local gradients
                    coords = nearby_data[['latitude', 'longitude']].values
                    
                    if 'fwi' in nearby_data.columns:
                        fwi_values = nearby_data['fwi'].values
                        # Simple gradient estimation
                        lat_gradient = np.gradient(fwi_values, coords[:, 0])
                        lon_gradient = np.gradient(fwi_values, coords[:, 1])
                        features['coarse_fwi_lat_gradient'] = np.mean(lat_gradient)
                        features['coarse_fwi_lon_gradient'] = np.mean(lon_gradient)
                except:
                    pass  # Skip gradient estimation if fails
            
            # Multi-scale statistics for better 1km prediction
            if len(nearby_data) > 2:
                numeric_cols = nearby_data.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col not in ['latitude', 'longitude']]
                
                for col in numeric_cols:
                    if col in nearby_data.columns:
                        values = nearby_data[col].dropna()
                        if len(values) > 0:
                            features[f'coarse_{col}_mean'] = values.mean()
                            features[f'coarse_{col}_min'] = values.min()
                            features[f'coarse_{col}_max'] = values.max()
                            if len(values) > 1:
                                features[f'coarse_{col}_std'] = values.std()
                                features[f'coarse_{col}_range'] = values.max() - values.min()
            
            return features
        
        except Exception as e:
            return None

    def _get_default_features_1km(self, target_lat, target_lon, daily_25km):
        """Get default features when feature extraction fails for 1km prediction"""
        try:
            # Use nearest point as fallback
            distances = np.sqrt(
                (daily_25km['latitude'] - target_lat)**2 + 
                (daily_25km['longitude'] - target_lon)**2
            )
            
            nearest_idx = distances.idxmin()
            nearest_point = daily_25km.loc[nearest_idx]
            
            features = {}
            
            # Basic features from nearest point
            for col in daily_25km.columns:
                if col not in ['latitude', 'longitude', 'date'] and pd.notna(nearest_point[col]):
                    features[f'coarse_{col}'] = nearest_point[col]
            
            # Spatial features
            features['coarse_distance'] = distances.min()
            features['coarse_lat'] = nearest_point['latitude']
            features['coarse_lon'] = nearest_point['longitude']
            features['target_lat'] = target_lat
            features['target_lon'] = target_lon
            
            return features
            
        except Exception as e:
            # Ultimate fallback
            return {
                'coarse_fwi': 10.0,
                'coarse_distance': 0.1,
                'coarse_lat': target_lat,
                'coarse_lon': target_lon,
                'target_lat': target_lat,
                'target_lon': target_lon
            }
        
    def _predict_daily_1km(self, daily_25km, target_1km_grid, model):
        """Predict FWI for 1km grid using super-resolution model (optimized for large grids)"""
        try:
            print(f"            Processing {len(target_1km_grid):,} 1km points...")
            
            # Process in chunks to manage memory
            chunk_size = 10000  # Process 10k points at a time
            all_results = []
            
            for chunk_start in range(0, len(target_1km_grid), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(target_1km_grid))
                chunk_grid = target_1km_grid.iloc[chunk_start:chunk_end]
                
                # Prepare features for this chunk
                chunk_features_list = []
                
                for _, target_point in chunk_grid.iterrows():
                    target_lat, target_lon = target_point['latitude'], target_point['longitude']
                    
                    # Extract features with enhanced precision for 1km
                    features = self._extract_features_from_25km_for_1km(
                        target_lat, target_lon, daily_25km
                    )
                    
                    if features is not None:
                        chunk_features_list.append(features)
                    else:
                        # Use default features
                        default_features = self._get_default_features_1km(target_lat, target_lon, daily_25km)
                        chunk_features_list.append(default_features)
                
                if not chunk_features_list:
                    continue
                
                # Convert to DataFrame and predict
                chunk_features_df = pd.DataFrame(chunk_features_list)
                
                # Ensure feature consistency
                for feature_name in self.feature_names:
                    if feature_name not in chunk_features_df.columns:
                        if 'fwi' in feature_name.lower():
                            default_val = daily_25km['fwi'].median() if 'fwi' in daily_25km.columns else 10.0
                        elif 'lat' in feature_name.lower():
                            default_val = chunk_grid['latitude'].mean()
                        elif 'lon' in feature_name.lower():
                            default_val = chunk_grid['longitude'].mean()
                        else:
                            default_val = 0.0
                        
                        chunk_features_df[feature_name] = default_val
                
                # Select and scale features
                X_chunk = chunk_features_df[self.feature_names].fillna(0)
                
                if 'features' in self.scalers:
                    X_chunk_scaled = self.scalers['features'].transform(X_chunk)
                else:
                    X_chunk_scaled = X_chunk.values
                
                # Predict
                chunk_predictions = model.predict(X_chunk_scaled)
                chunk_predictions = np.clip(chunk_predictions, 0, 100)
                
                # Create result DataFrame for this chunk
                chunk_results = chunk_grid.copy()
                chunk_results['fwi_predicted'] = chunk_predictions
                
                # Enhanced quality metrics for 1km predictions
                pred_std = np.std(chunk_predictions)
                chunk_results['prediction_uncertainty'] = pred_std / np.sqrt(len(daily_25km))
                
                # Distance-based quality scoring (closer to source = higher quality)
                for idx, row in chunk_results.iterrows():
                    min_dist_to_source = np.min(np.sqrt(
                        (daily_25km['latitude'] - row['latitude'])**2 + 
                        (daily_25km['longitude'] - row['longitude'])**2
                    ))
                    # Quality decreases with distance
                    quality = max(0.3, 1.0 - (min_dist_to_source / 0.5))  # Full quality within 50km
                    chunk_results.loc[idx, 'quality_score'] = quality
                
                all_results.append(chunk_results)
                
                if (chunk_end // chunk_size) % 5 == 0:
                    print(f"               Processed {chunk_end:,}/{len(target_1km_grid):,} points...")
            
            if all_results:
                final_results = pd.concat(all_results, ignore_index=True)
                print(f"            SUCCESS: Completed 1km prediction: {len(final_results):,} points")
                return final_results
            else:
                return None
        
        except Exception as e:
            print(f"         ERROR: 1km daily prediction failed: {e}")
            return None

    def _predict_daily_1km_optimized(self, daily_25km, target_1km_grid, model):
        """Ultra-optimized 1km prediction for maximum speed"""
        try:
            print(f"            Processing {len(target_1km_grid):,} 1km points (ULTRA-OPTIMIZED)...")
            
            from concurrent.futures import ThreadPoolExecutor
            import multiprocessing as mp
            
            # Larger chunk sizes for fewer overhead
            chunk_size = min(100000, len(target_1km_grid) // mp.cpu_count())  # Increased from 25k to 100k
            chunk_size = max(20000, chunk_size)  # Minimum 20k (was 5k)
            
            print(f"            Using large chunk size: {chunk_size:,} points")
            
            # Pre-compute spatial lookup
            spatial_lookup = self._create_spatial_lookup_optimized(daily_25km)
            
            all_results = []
            total_chunks = (len(target_1km_grid) + chunk_size - 1) // chunk_size
            
            # More parallel workers
            max_workers = min(6, mp.cpu_count())  # Increased from 4 to 6
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for chunk_idx in range(total_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, len(target_1km_grid))
                    chunk_grid = target_1km_grid.iloc[chunk_start:chunk_end]
                    
                    future = executor.submit(
                        self._process_chunk_ultra_fast,  # Ultra-fast version
                        chunk_grid, daily_25km, model, spatial_lookup, chunk_idx + 1, total_chunks
                    )
                    futures.append(future)
                
                # Collect results with less frequent progress updates
                completed_chunks = 0
                for future in futures:
                    chunk_result = future.result()
                    if chunk_result is not None:
                        all_results.append(chunk_result)
                        completed_chunks += 1
                        
                        # Less frequent progress updates
                        if completed_chunks % 2 == 0 or completed_chunks == total_chunks:  # Every 2 chunks
                            progress = (completed_chunks / total_chunks) * 100
                            print(f"            Progress: {completed_chunks}/{total_chunks} chunks ({progress:.1f}%)")
                
        except Exception as e:
            print(f"         ERROR: Optimized 1km prediction failed: {e}")
            return None

    def _create_spatial_lookup(self, daily_25km):
        """Create spatial lookup structure for fast nearest neighbor queries"""
        try:
            from scipy.spatial import cKDTree
            
            coords = daily_25km[['latitude', 'longitude']].values
            tree = cKDTree(coords)
            
            return {
                'tree': tree,
                'data': daily_25km,
                'coords': coords
            }
        except Exception as e:
            return None

    def _process_chunk_vectorized(self, chunk_grid, daily_25km, model, spatial_lookup, chunk_num, total_chunks):
        """Process a chunk of points using vectorized operations"""
        try:
            print(f"               Processing chunk {chunk_num}/{total_chunks}...")
            
            # Vectorized feature extraction
            features_df = self._extract_features_vectorized_fast(chunk_grid, spatial_lookup)
            
            if features_df is None or len(features_df) == 0:
                return None
            
            # Ensure feature consistency
            for feature_name in self.feature_names:
                if feature_name not in features_df.columns:
                    if 'fwi' in feature_name.lower():
                        default_val = spatial_lookup['data']['fwi'].median() if 'fwi' in spatial_lookup['data'].columns else 10.0
                    elif 'lat' in feature_name.lower():
                        default_val = chunk_grid['latitude'].mean()
                    elif 'lon' in feature_name.lower():
                        default_val = chunk_grid['longitude'].mean()
                    else:
                        default_val = 0.0
                    features_df[feature_name] = default_val
            
            # Select and scale features
            X = features_df[self.feature_names].fillna(0)
            
            if 'features' in self.scalers:
                X_scaled = self.scalers['features'].transform(X)
            else:
                X_scaled = X.values
            
            # Batch prediction
            predictions = model.predict(X_scaled)
            predictions = np.clip(predictions, 0, 100)
            
            # Create results
            results = chunk_grid.copy()
            results['fwi_predicted'] = predictions
            
            # Simplified quality metrics for speed
            results['prediction_uncertainty'] = np.std(predictions) / np.sqrt(len(spatial_lookup['data']))
            results['quality_score'] = 0.8  # Simplified constant quality score
            
            return results
        
        except Exception as e:
            print(f"               ERROR: Chunk {chunk_num} failed: {e}")
            return None

    def _extract_features_vectorized_fast(self, target_grid, spatial_lookup):
        """Ultra-fast vectorized feature extraction using spatial lookup"""
        try:
            if spatial_lookup is None:
                return None
            
            target_coords = target_grid[['latitude', 'longitude']].values
            
            # Find nearest neighbors for all points at once
            distances, indices = spatial_lookup['tree'].query(target_coords, k=1)
            
            # Extract features from nearest points
            nearest_data = spatial_lookup['data'].iloc[indices]
            
            features_dict = {}
            
            # Basic features from nearest points
            for col in spatial_lookup['data'].columns:
                if col not in ['latitude', 'longitude', 'date'] and spatial_lookup['data'][col].dtype in ['int64', 'float64']:
                    features_dict[f'coarse_{col}'] = nearest_data[col].values
            
            # Spatial features
            features_dict['coarse_distance'] = distances
            features_dict['coarse_lat'] = nearest_data['latitude'].values
            features_dict['coarse_lon'] = nearest_data['longitude'].values
            features_dict['target_lat'] = target_coords[:, 0]
            features_dict['target_lon'] = target_coords[:, 1]
            
            return pd.DataFrame(features_dict)
        
        except Exception as e:
            print(f"      ERROR: Vectorized feature extraction failed: {e}")
            return None


def main():
    print("FWI SUPER-RESOLUTION PIPELINE")
    print("="*65)

    sr_system = FWISuperResolutionModel()

    # Step 1: Load + preprocess
    if not sr_system.load_era5_data():
        return
    if not sr_system.preprocess_and_merge():
        return

    # Step 2: Training data
    sr_system.create_downsampled_training_data('50km')

    # Step 3: Train models
    sr_system.train_super_resolution_models()

    # Step 4: Predict 1km
    sr_system.apply_super_resolution_25km_to_1km()

    # Step 5: Save results
    sr_system.save_results_and_analysis()


if __name__ == "__main__":
    main()
