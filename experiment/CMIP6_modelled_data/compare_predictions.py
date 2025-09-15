"""
compare_predictions.py

Compare ML-predicted FWI with ERA5 ground truth data
Evaluate Random Forest, CNN, and Enhanced ML model performance
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(script_dir):
    """Load all datasets for comparison"""
    datasets = {}
    
    # Load ERA5 ground truth
    era5_file = "era5_fwi_2013_portugal_3decimal.csv"
    era5_path = os.path.join(script_dir, era5_file)
    
    if os.path.exists(era5_path):
        print(f"Loading ERA5 ground truth: {era5_file}")
        datasets['era5'] = pd.read_csv(era5_path)
        print(f"  ERA5 shape: {datasets['era5'].shape}")
        print(f"  ERA5 columns: {list(datasets['era5'].columns)}")
    else:
        print(f"❌ ERA5 file not found: {era5_file}")
        return None
    
    # Load RF predictions
    rf_files = [f for f in os.listdir(script_dir) if f.startswith('fwi_rf_predictions_') and f.endswith('.csv')]
    if rf_files:
        rf_file = rf_files[0]
        print(f"Loading RF predictions: {rf_file}")
        datasets['rf'] = pd.read_csv(os.path.join(script_dir, rf_file))
        print(f"  RF shape: {datasets['rf'].shape}")
        print(f"  RF columns: {list(datasets['rf'].columns)}")
    else:
        print("❌ No RF prediction files found")
    
    # Load CNN predictions
    cnn_files = [f for f in os.listdir(script_dir) if f.startswith('fwi_cnn_predictions_') and f.endswith('.csv')]
    if cnn_files:
        cnn_file = cnn_files[0]
        print(f"Loading CNN predictions: {cnn_file}")
        datasets['cnn'] = pd.read_csv(os.path.join(script_dir, cnn_file))
        print(f"  CNN shape: {datasets['cnn'].shape}")
        print(f"  CNN columns: {list(datasets['cnn'].columns)}")
    else:
        print("❌ No CNN prediction files found")
    
    # Load Enhanced ML predictions
    enhanced_files = [f for f in os.listdir(script_dir) if f.startswith('enhanced_fwi_predictions_') and f.endswith('.csv')]
    if enhanced_files:
        enhanced_file = enhanced_files[0]
        print(f"Loading Enhanced ML predictions: {enhanced_file}")
        datasets['enhanced'] = pd.read_csv(os.path.join(script_dir, enhanced_file))
        print(f"  Enhanced ML shape: {datasets['enhanced'].shape}")
        print(f"  Enhanced ML columns: {list(datasets['enhanced'].columns)}")
    else:
        print("❌ No Enhanced ML prediction files found")
    
    return datasets

def standardize_columns(df, dataset_name):
    """Standardize column names across datasets"""
    print(f"Standardizing columns for {dataset_name}...")
    
    # Common column mappings
    column_mappings = {
        # Time columns
        'time': 'time',
        'date': 'time',
        'datetime': 'time',
        
        # Latitude columns
        'lat': 'lat',
        'latitude': 'lat',
        'Latitude': 'lat',
        'LAT': 'lat',
        
        # Longitude columns
        'lon': 'lon',
        'longitude': 'lon',
        'Longitude': 'lon',
        'LON': 'lon',
        
        # FWI columns
        'fwi': 'fwi',
        'FWI': 'fwi',
        'fire_weather_index': 'fwi',
        'Fire_Weather_Index': 'fwi',
        'fwi_rf': 'fwi_rf',
        'fwi_cnn': 'fwi_cnn',
        'fwi_predicted': 'fwi_enhanced',  # Enhanced ML prediction
        'fwi_random_forest': 'fwi_rf_individual',
        'fwi_gradient_boosting': 'fwi_gb',
        'fwi_xgboost': 'fwi_xgb'
    }
    
    # Apply column mappings
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
            print(f"  Renamed '{old_name}' to '{new_name}'")
    
    # Check for required columns
    required_cols = ['time', 'lat', 'lon']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"  Still missing columns: {missing_cols}")
        print(f"  Available columns: {list(df.columns)}")
        
        # Try to infer columns from patterns
        for col in df.columns:
            if 'lat' in col.lower() and 'lat' not in df.columns:
                df = df.rename(columns={col: 'lat'})
                print(f"  Inferred latitude column: '{col}' -> 'lat'")
            elif 'lon' in col.lower() and 'lon' not in df.columns:
                df = df.rename(columns={col: 'lon'})
                print(f"  Inferred longitude column: '{col}' -> 'lon'")
            elif ('time' in col.lower() or 'date' in col.lower()) and 'time' not in df.columns:
                df = df.rename(columns={col: 'time'})
                print(f"  Inferred time column: '{col}' -> 'time'")
    
    # Final check
    final_missing = [col for col in required_cols if col not in df.columns]
    if final_missing:
        print(f"  ❌ Cannot find required columns: {final_missing}")
        return None
    
    return df

def preprocess_data(datasets):
    """Preprocess and align all datasets"""
    print("\nPreprocessing data...")
    
    processed_datasets = {}
    
    # Process each dataset
    for name, df in datasets.items():
        print(f"\nProcessing {name} data...")
        
        # Standardize column names
        df = standardize_columns(df, name)
        if df is None:
            print(f"❌ Failed to standardize {name} columns")
            continue
        
        # Parse time
        print(f"  Parsing time column...")
        try:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        except Exception as e:
            print(f"  Warning: Time parsing issue: {e}")
            # Try alternative parsing
            try:
                df['time'] = pd.to_datetime(df['time'], format='mixed', errors='coerce')
            except:
                print(f"  ❌ Cannot parse time column for {name}")
                continue
        
        # Remove rows with invalid time
        invalid_time = df['time'].isna()
        if invalid_time.any():
            print(f"  Removing {invalid_time.sum()} rows with invalid time")
            df = df.dropna(subset=['time'])
        
        # Handle longitude format conversion
        print(f"  Original longitude range: {df['lon'].min():.3f} to {df['lon'].max():.3f}")
        
        if name != 'era5':
            # Convert from 0-360 to -180/180 if needed
            if df['lon'].max() > 180:
                print(f"  Converting {name} longitude from 0-360 to -180/180 format")
                df['lon'] = df['lon'] - 360
        
        print(f"  Converted longitude range: {df['lon'].min():.3f} to {df['lon'].max():.3f}")
        
        # Add date column for alignment
        df['date'] = df['time'].dt.date
        
        # Remove rows with invalid coordinates
        invalid_coords = df[['lat', 'lon']].isna().any(axis=1)
        if invalid_coords.any():
            print(f"  Removing {invalid_coords.sum()} rows with invalid coordinates")
            df = df.dropna(subset=['lat', 'lon'])
        
        print(f"  Time range: {df['time'].min()} to {df['time'].max()}")
        print(f"  Spatial range: lat {df['lat'].min():.3f} to {df['lat'].max():.3f}, lon {df['lon'].min():.3f} to {df['lon'].max():.3f}")
        print(f"  Unique coordinates: {len(df[['lat', 'lon']].drop_duplicates())}")
        
        # Check FWI columns
        fwi_cols = [col for col in df.columns if 'fwi' in col.lower()]
        print(f"  FWI columns: {fwi_cols}")
        
        # Ensure we have at least one FWI column
        if not fwi_cols:
            print(f"  ❌ No FWI columns found in {name}")
            continue
        
        processed_datasets[name] = df
    
    return processed_datasets

def interpolate_to_era5_grid(source_df, target_df, value_col, max_distance_km=25):
    """Interpolate source data to ERA5 grid using nearest neighbor"""
    print(f"\nInterpolating {value_col} to ERA5 grid...")
    
    # Get unique dates
    dates = sorted(set(source_df['date'].unique()) & set(target_df['date'].unique()))
    print(f"  Common dates: {len(dates)}")
    
    if len(dates) == 0:
        print("  ❌ No common dates found")
        return None
    
    interpolated_data = []
    
    for i, date in enumerate(dates):
        if i % 30 == 0:  # Progress every 30 days
            print(f"  Processing date {i+1}/{len(dates)}: {date}")
        
        # Get source data for this date
        source_day = source_df[source_df['date'] == date]
        target_day = target_df[target_df['date'] == date]
        
        if len(source_day) == 0 or len(target_day) == 0:
            continue
        
        # Build KDTree for source coordinates
        source_coords = source_day[['lat', 'lon']].values
        source_values = source_day[value_col].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(source_values)
        source_coords = source_coords[valid_mask]
        source_values = source_values[valid_mask]
        
        if len(source_coords) == 0:
            continue
        
        # Build tree
        tree = cKDTree(source_coords)
        
        # Query target points
        target_coords = target_day[['lat', 'lon']].values
        distances, indices = tree.query(target_coords)
        
        # Convert distance to kilometers (approximate)
        distances_km = distances * 111.0  # 1 degree ≈ 111 km
        
        # Only keep points within max_distance
        valid_interpolation = distances_km <= max_distance_km
        
        # Create interpolated data
        for j, (_, target_point) in enumerate(target_day.iterrows()):
            if valid_interpolation[j]:
                interpolated_data.append({
                    'date': date,
                    'lat': target_point['lat'],
                    'lon': target_point['lon'],
                    'time': target_point['time'],
                    value_col: source_values[indices[j]],
                    'interpolation_distance_km': distances_km[j]
                })
    
    if len(interpolated_data) == 0:
        print("  ❌ No valid interpolations")
        return None
    
    result_df = pd.DataFrame(interpolated_data)
    print(f"  Interpolated {len(result_df)} points")
    print(f"  Distance stats: mean={result_df['interpolation_distance_km'].mean():.1f}km, max={result_df['interpolation_distance_km'].max():.1f}km")
    
    return result_df

def align_datasets_with_interpolation(datasets):
    """Align datasets using spatial interpolation"""
    print("\n" + "="*60)
    print("SPATIAL INTERPOLATION ALIGNMENT")
    print("="*60)
    
    if 'era5' not in datasets:
        print("❌ No ERA5 data for alignment")
        return None
    
    era5_df = datasets['era5'].copy()
    
    # Identify FWI column in ERA5
    era5_fwi_cols = [col for col in era5_df.columns if 'fwi' in col.lower() and 'predicted' not in col.lower()]
    if not era5_fwi_cols:
        print("❌ No FWI column found in ERA5 data")
        return None
    
    era5_fwi_col = era5_fwi_cols[0]
    print(f"Using ERA5 FWI column: {era5_fwi_col}")
    
    # Start with ERA5 data
    aligned_df = era5_df[['time', 'date', 'lat', 'lon', era5_fwi_col]].copy()
    aligned_df = aligned_df.rename(columns={era5_fwi_col: 'fwi_era5'})
    
    print(f"Starting with ERA5 data: {len(aligned_df)} points")
    
    # Interpolate other datasets to ERA5 grid
    model_mappings = {
        'rf': 'fwi_rf',
        'cnn': 'fwi_cnn',
        'enhanced': 'fwi_enhanced'
    }
    
    for name, value_col in model_mappings.items():
        if name not in datasets:
            continue
        
        print(f"\nInterpolating {name.upper()} predictions to ERA5 grid...")
        
        source_df = datasets[name]
        
        if value_col not in source_df.columns:
            print(f"  ❌ {value_col} not found in {name} data")
            continue
        
        # Interpolate to ERA5 grid
        interpolated = interpolate_to_era5_grid(
            source_df, aligned_df, value_col, max_distance_km=30
        )
        
        if interpolated is not None:
            # Merge with aligned dataset
            aligned_df = aligned_df.merge(
                interpolated[['date', 'lat', 'lon', value_col]],
                on=['date', 'lat', 'lon'],
                how='left'
            )
            
            coverage = aligned_df[value_col].notna().sum()
            total = len(aligned_df)
            print(f"  {name.upper()} coverage: {coverage}/{total} ({coverage/total*100:.1f}%)")
        else:
            print(f"  ❌ No {name.upper()} data interpolated")
    
    # Keep only rows with ERA5 data
    aligned_df = aligned_df[aligned_df['fwi_era5'].notna()]
    
    print(f"\nFinal aligned dataset: {len(aligned_df)} points")
    
    # Show coverage for each model
    pred_cols = [col for col in aligned_df.columns if col.startswith('fwi_') and col != 'fwi_era5']
    for col in pred_cols:
        coverage = aligned_df[col].notna().sum()
        print(f"  {col}: {coverage}/{len(aligned_df)} ({coverage/len(aligned_df)*100:.1f}%)")
    
    return aligned_df

def calculate_metrics(df):
    """Calculate performance metrics"""
    print("\nCalculating performance metrics...")
    
    metrics = {}
    
    # Available prediction columns
    pred_cols = [col for col in df.columns if col.startswith('fwi_') and col != 'fwi_era5']
    print(f"Prediction columns found: {pred_cols}")
    
    model_name_mapping = {
        'fwi_rf': 'Random Forest',
        'fwi_cnn': 'CNN',
        'fwi_enhanced': 'Enhanced ML'
    }
    
    for pred_col in pred_cols:
        model_name = model_name_mapping.get(pred_col, pred_col.replace('fwi_', '').upper())
        
        # Get valid data points
        valid_data = df[['fwi_era5', pred_col]].dropna()
        
        if len(valid_data) == 0:
            print(f"❌ No valid data for {model_name}")
            continue
        
        y_true = valid_data['fwi_era5'].values
        y_pred = valid_data[pred_col].values
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Pearson correlation
        corr, p_value = pearsonr(y_true, y_pred)
        
        # Bias
        bias = np.mean(y_pred - y_true)
        
        # Relative error
        relative_error = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-6)) * 100
        
        metrics[model_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Correlation': corr,
            'P-value': p_value,
            'Bias': bias,
            'Relative Error (%)': relative_error,
            'N_samples': len(valid_data)
        }
        
        print(f"\n{model_name} Performance:")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  R²: {r2:.3f}")
        print(f"  Correlation: {corr:.3f} (p={p_value:.3e})")
        print(f"  Bias: {bias:.3f}")
        print(f"  Relative Error: {relative_error:.1f}%")
        print(f"  Sample size: {len(valid_data)}")
    
    return metrics

def create_comparison_plots(df, metrics, output_dir):
    """Create comprehensive comparison plots"""
    print("\nCreating comparison plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Available prediction columns
    pred_cols = [col for col in df.columns if col.startswith('fwi_') and col != 'fwi_era5']
    
    if not pred_cols:
        print("❌ No prediction columns found for plotting")
        return
    
    model_name_mapping = {
        'fwi_rf': 'Random Forest',
        'fwi_cnn': 'CNN',
        'fwi_enhanced': 'Enhanced ML'
    }
    
    # 1. Scatter plots
    fig, axes = plt.subplots(1, len(pred_cols), figsize=(6*len(pred_cols), 5))
    if len(pred_cols) == 1:
        axes = [axes]
    
    for i, pred_col in enumerate(pred_cols):
        model_name = model_name_mapping.get(pred_col, pred_col.replace('fwi_', '').upper())
        
        # Get valid data
        valid_data = df[['fwi_era5', pred_col]].dropna()
        
        if len(valid_data) == 0:
            continue
        
        # Scatter plot
        axes[i].scatter(valid_data['fwi_era5'], valid_data[pred_col], 
                       alpha=0.6, s=20, edgecolors='none')
        
        # 1:1 line
        min_val = min(valid_data['fwi_era5'].min(), valid_data[pred_col].min())
        max_val = max(valid_data['fwi_era5'].max(), valid_data[pred_col].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        # Labels and title
        axes[i].set_xlabel('ERA5 FWI (Ground Truth)')
        axes[i].set_ylabel(f'{model_name} FWI (Predicted)')
        axes[i].set_title(f'{model_name} vs ERA5\nR² = {metrics[model_name]["R²"]:.3f}, N = {len(valid_data)}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Spatial distribution comparison
    fig, axes = plt.subplots(1, len(pred_cols)+1, figsize=(5*(len(pred_cols)+1), 4))
    if len(pred_cols) == 0:
        axes = [axes]
    
    # ERA5 spatial distribution
    era5_data = df[['lat', 'lon', 'fwi_era5']].dropna()
    era5_spatial = era5_data.groupby(['lat', 'lon'])['fwi_era5'].mean().reset_index()
    
    scatter = axes[0].scatter(era5_spatial['lon'], era5_spatial['lat'], 
                             c=era5_spatial['fwi_era5'], cmap='YlOrRd', s=30)
    axes[0].set_title('ERA5 FWI (Mean)')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(scatter, ax=axes[0])
    
    # Prediction spatial distributions
    for i, pred_col in enumerate(pred_cols):
        model_name = model_name_mapping.get(pred_col, pred_col.replace('fwi_', '').upper())
        pred_data = df[['lat', 'lon', pred_col]].dropna()
        
        if len(pred_data) > 0:
            pred_spatial = pred_data.groupby(['lat', 'lon'])[pred_col].mean().reset_index()
            scatter = axes[i+1].scatter(pred_spatial['lon'], pred_spatial['lat'], 
                                       c=pred_spatial[pred_col], cmap='YlOrRd', s=30)
            axes[i+1].set_title(f'{model_name} FWI (Mean)')
            axes[i+1].set_xlabel('Longitude')
            axes[i+1].set_ylabel('Latitude')
            plt.colorbar(scatter, ax=axes[i+1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Model performance comparison bar chart
    if len(metrics) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        models = list(metrics.keys())
        metrics_names = ['RMSE', 'MAE', 'R²', 'Correlation']
        
        for i, metric in enumerate(metrics_names):
            values = [metrics[model][metric] for model in models]
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to: {output_dir}")

def create_metrics_table(metrics, output_dir):
    """Create a summary table of metrics"""
    print("\nCreating metrics summary table...")
    
    if not metrics:
        print("❌ No metrics to create table")
        return None
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics).T
    
    # Round values for display
    metrics_df = metrics_df.round(4)
    
    # Save to CSV
    metrics_path = os.path.join(output_dir, 'performance_metrics.csv')
    metrics_df.to_csv(metrics_path)
    
    # Print formatted table
    print("\nPerformance Metrics Summary:")
    print("=" * 80)
    print(metrics_df.to_string())
    print("=" * 80)
    
    return metrics_df

def main():
    """Main function"""
    print("Enhanced FWI Prediction Comparison Analysis")
    print("=" * 70)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load datasets
    datasets = load_data(script_dir)
    if datasets is None:
        return
    
    # Preprocess data
    datasets = preprocess_data(datasets)
    if not datasets:
        print("❌ No datasets successfully processed")
        return
    
    # Align datasets using spatial interpolation
    aligned_df = align_datasets_with_interpolation(datasets)
    if aligned_df is None or len(aligned_df) == 0:
        print("❌ No aligned data points")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(aligned_df)
    if not metrics:
        print("❌ No metrics calculated")
        return
    
    # Create output directory
    output_dir = os.path.join(script_dir, 'enhanced_comparison_results')
    
    # Create plots
    create_comparison_plots(aligned_df, metrics, output_dir)
    
    # Create metrics table
    metrics_df = create_metrics_table(metrics, output_dir)
    
    # Save aligned dataset
    aligned_path = os.path.join(output_dir, 'aligned_enhanced_dataset.csv')
    aligned_df.to_csv(aligned_path, index=False)
    print(f"\nAligned dataset saved to: {aligned_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("ENHANCED FWI COMPARISON ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"Total data points compared: {len(aligned_df):,}")
    print(f"Time range: {aligned_df['time'].min()} to {aligned_df['time'].max()}")
    print(f"Spatial coverage: {len(aligned_df[['lat', 'lon']].drop_duplicates())} locations")
    
    # Model ranking
    if len(metrics) > 1:
        print("\nModel Ranking (by R²):")
        ranking = sorted(metrics.items(), key=lambda x: x[1]['R²'], reverse=True)
        for i, (model, perf) in enumerate(ranking, 1):
            print(f"  {i}. {model}: R² = {perf['R²']:.3f}, RMSE = {perf['RMSE']:.3f}")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()