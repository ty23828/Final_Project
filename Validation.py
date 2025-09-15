import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import json
import os
warnings.filterwarnings('ignore')

def aggregate_1km_to_25km(df_1km, df_25km_ref, fwi_column='fwi_predicted'):
    """
    Aggregate 1km data back to 25km using 25km reference data coordinate system
    """
    print(f"Starting aggregation of 1km data to 25km using column: {fwi_column}")
    
    # Ensure correct data types
    df_1km = df_1km.copy()
    df_25km_ref = df_25km_ref.copy()
    
    # Handle different time column names
    time_col_1km = None
    time_col_25km = None
    
    for col in ['time', 'date']:
        if col in df_1km.columns:
            time_col_1km = col
            break
    
    for col in ['time', 'date']:
        if col in df_25km_ref.columns:
            time_col_25km = col
            break
    
    if time_col_1km is None:
        print("Warning: No time column found in 1km data, proceeding without temporal matching")
        use_time = False
    elif time_col_25km is None:
        print("Warning: No time column found in 25km reference data, proceeding without temporal matching")
        use_time = False
    else:
        use_time = True
        # Convert date formats
        df_1km[time_col_1km] = pd.to_datetime(df_1km[time_col_1km])
        df_25km_ref[time_col_25km] = pd.to_datetime(df_25km_ref[time_col_25km])
    
    # Get unique coordinates of 25km grid
    coords_25km = df_25km_ref[['longitude', 'latitude']].drop_duplicates()
    print(f"Number of 25km grid points: {len(coords_25km)}")
    
    # Get time range if available
    if use_time:
        common_dates = pd.to_datetime(sorted(set(df_1km[time_col_1km]).intersection(set(df_25km_ref[time_col_25km]))))
        print(f"Number of common time points: {len(common_dates)}")
    else:
        common_dates = [None]  # Process without time dimension
    
    aggregated_data = []
    
    for idx, (_, coord_25km) in enumerate(coords_25km.iterrows()):
        if idx % 10 == 0:
            print(f"Processing progress: {idx}/{len(coords_25km)}")
            
        lon_25km = coord_25km['longitude']
        lat_25km = coord_25km['latitude']
        
        # Define 25km grid boundaries (assuming grid spacing ~0.25 degrees)
        lon_min = lon_25km - 0.125
        lon_max = lon_25km + 0.125
        lat_min = lat_25km - 0.125
        lat_max = lat_25km + 0.125
        
        # Find all 1km points within this 25km grid
        mask_spatial = (
            (df_1km['longitude'] >= lon_min) & 
            (df_1km['longitude'] <= lon_max) &
            (df_1km['latitude'] >= lat_min) & 
            (df_1km['latitude'] <= lat_max)
        )
        
        grid_1km_data = df_1km[mask_spatial]
        
        if len(grid_1km_data) == 0:
            continue
        
        if use_time:
            # Aggregate by time
            for date in common_dates:
                if date is None:
                    continue
                date_mask = grid_1km_data[time_col_1km] == date
                daily_data = grid_1km_data[date_mask]
                
                if len(daily_data) == 0:
                    continue
                    
                # Calculate aggregated values (using mean)
                fwi_values = daily_data[fwi_column].dropna()
                
                if len(fwi_values) > 0:
                    aggregated_data.append({
                        'longitude': lon_25km,
                        'latitude': lat_25km,
                        'time': date,
                        'fwi_aggregated': fwi_values.mean(),
                        'fwi_std': fwi_values.std(),
                        'count_1km_points': len(fwi_values)
                    })
        else:
            # Aggregate without time dimension
            fwi_values = grid_1km_data[fwi_column].dropna()
            
            if len(fwi_values) > 0:
                aggregated_data.append({
                    'longitude': lon_25km,
                    'latitude': lat_25km,
                    'fwi_aggregated': fwi_values.mean(),
                    'fwi_std': fwi_values.std(),
                    'count_1km_points': len(fwi_values)
                })
    
    df_aggregated = pd.DataFrame(aggregated_data)
    print(f"Aggregation completed, result data points: {len(df_aggregated)}")
    
    return df_aggregated

def merge_and_validate(df_aggregated, df_25km_original):
    """
    Merge aggregated data with original 25km data for validation
    """
    print("Merging data for validation...")
    
    # Handle different time columns
    time_col_agg = 'time' if 'time' in df_aggregated.columns else None
    time_col_orig = None
    
    for col in ['time', 'date']:
        if col in df_25km_original.columns:
            time_col_orig = col
            break
    
    if time_col_agg and time_col_orig:
        # Ensure consistent time formats
        df_aggregated[time_col_agg] = pd.to_datetime(df_aggregated[time_col_agg])
        df_25km_original[time_col_orig] = pd.to_datetime(df_25km_original[time_col_orig])
        
        # Merge data with time
        merge_cols = ['longitude', 'latitude', time_col_agg]
        df_25km_original_renamed = df_25km_original.rename(columns={time_col_orig: time_col_agg})
        
        merged = pd.merge(
            df_aggregated,
            df_25km_original_renamed,
            on=merge_cols,
            how='inner',
            suffixes=('_agg', '_orig')
        )
    else:
        # Merge data without time (spatial only)
        merge_cols = ['longitude', 'latitude']
        
        merged = pd.merge(
            df_aggregated,
            df_25km_original,
            on=merge_cols,
            how='inner',
            suffixes=('_agg', '_orig')
        )
    
    print(f"Data size after merging: {len(merged)}")
    
    # Remove missing values
    merged = merged.dropna(subset=['fwi_aggregated', 'fwi'])
    print(f"Data size after removing missing values: {len(merged)}")
    
    return merged

def calculate_metrics(y_true, y_pred):
    """
    Calculate validation metrics including R-squared
    """
    # Remove infinite values and NaN
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {}
    
    # Calculate metrics
    correlation, p_value = pearsonr(y_true_clean, y_pred_clean)
    r_squared = r2_score(y_true_clean, y_pred_clean)  # Using sklearn
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    # Calculate relative errors
    mean_true = np.mean(y_true_clean)
    relative_rmse = rmse / mean_true * 100 if mean_true != 0 else np.inf
    relative_mae = mae / mean_true * 100 if mean_true != 0 else np.inf
    
    return {
        'correlation': correlation,
        'r_squared': r_squared,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae,
        'relative_rmse': relative_rmse,
        'relative_mae': relative_mae,
        'n_points': len(y_true_clean)
    }

def create_validation_plots(merged_df, model_name, output_prefix='validation'):
    """
    Create validation plots for a specific model
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'FWI Validation - {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot
    ax1 = axes[0, 0]
    scatter = ax1.scatter(merged_df['fwi'], merged_df['fwi_aggregated'], 
                         alpha=0.6, s=20, c='blue')
    
    # Add 1:1 line
    min_val = min(merged_df['fwi'].min(), merged_df['fwi_aggregated'].min())
    max_val = max(merged_df['fwi'].max(), merged_df['fwi_aggregated'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
    
    ax1.set_xlabel('Original 25km FWI')
    ax1.set_ylabel('Aggregated FWI (from 1km)')
    ax1.set_title('FWI: Original vs Aggregated')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals plot
    ax2 = axes[0, 1]
    residuals = merged_df['fwi_aggregated'] - merged_df['fwi']
    ax2.scatter(merged_df['fwi'], residuals, alpha=0.6, s=20, c='green')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Original 25km FWI')
    ax2.set_ylabel('Residuals (Aggregated - Original)')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Time series comparison (if time data available)
    ax3 = axes[1, 0]
    if 'time' in merged_df.columns:
        time_series = merged_df.groupby('time').agg({
            'fwi': 'mean',
            'fwi_aggregated': 'mean'
        }).reset_index()
        
        ax3.plot(time_series['time'], time_series['fwi'], 
                 label='Original 25km', linewidth=2, alpha=0.8)
        ax3.plot(time_series['time'], time_series['fwi_aggregated'], 
                 label='Aggregated from 1km', linewidth=2, alpha=0.8)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Mean FWI')
        ax3.set_title('Time Series Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    else:
        # Show spatial mean comparison instead
        spatial_mean = merged_df.groupby(['longitude', 'latitude']).agg({
            'fwi': 'mean',
            'fwi_aggregated': 'mean'
        }).reset_index()
        
        ax3.scatter(spatial_mean['fwi'], spatial_mean['fwi_aggregated'], alpha=0.6)
        ax3.plot([spatial_mean['fwi'].min(), spatial_mean['fwi'].max()], 
                [spatial_mean['fwi'].min(), spatial_mean['fwi'].max()], 'r--')
        ax3.set_xlabel('Original 25km FWI (Spatial Mean)')
        ax3.set_ylabel('Aggregated FWI (Spatial Mean)')
        ax3.set_title('Spatial Mean Comparison')
        ax3.grid(True, alpha=0.3)
    
    # 4. Distribution comparison
    ax4 = axes[1, 1]
    ax4.hist(merged_df['fwi'], bins=50, alpha=0.7, label='Original 25km', density=True)
    ax4.hist(merged_df['fwi_aggregated'], bins=50, alpha=0.7, label='Aggregated from 1km', density=True)
    ax4.set_xlabel('FWI Value')
    ax4.set_ylabel('Density')
    ax4.set_title('FWI Distribution Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'{output_prefix}_{model_name.lower().replace(" ", "_")}_plots.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation plots saved as: {plot_filename}")
    
    return fig

def validate_single_model(model_file, model_name, df_25km_original):
    """
    Validate a single 1km prediction model
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING MODEL: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load 1km prediction data
        print(f"Loading {model_file}...")
        df_1km = pd.read_csv(model_file)
        print(f"1km data shape: {df_1km.shape}")
        print(f"1km data columns: {list(df_1km.columns)}")
        
        # Check for required columns
        required_cols = ['latitude', 'longitude']
        fwi_col = None
        
        # Find FWI prediction column
        for col in ['fwi_predicted', 'fwi_pred', 'fwi']:
            if col in df_1km.columns:
                fwi_col = col
                break
        
        if fwi_col is None:
            print(f"ERROR: No FWI prediction column found in {model_file}")
            return None, None
        
        print(f"Using FWI column: {fwi_col}")
        
        # Check coordinate columns
        missing_cols = [col for col in required_cols if col not in df_1km.columns]
        if missing_cols:
            print(f"ERROR: Missing required columns: {missing_cols}")
            return None, None
        
        # Show basic statistics
        fwi_stats = df_1km[fwi_col].describe()
        print(f"FWI prediction statistics:")
        print(f"  Count: {fwi_stats['count']:,.0f}")
        print(f"  Mean: {fwi_stats['mean']:.4f}")
        print(f"  Std: {fwi_stats['std']:.4f}")
        print(f"  Range: [{fwi_stats['min']:.4f}, {fwi_stats['max']:.4f}]")
        
        # Check for unusual values
        negative_count = (df_1km[fwi_col] < 0).sum()
        extreme_count = (df_1km[fwi_col] > 100).sum()
        missing_count = df_1km[fwi_col].isnull().sum()
        
        print(f"Data quality:")
        print(f"  Negative values: {negative_count} ({negative_count/len(df_1km)*100:.2f}%)")
        print(f"  Extreme values (>100): {extreme_count} ({extreme_count/len(df_1km)*100:.2f}%)")
        print(f"  Missing values: {missing_count} ({missing_count/len(df_1km)*100:.2f}%)")
        
        # Aggregate 1km data to 25km
        print(f"\nAggregating 1km data to 25km...")
        df_aggregated = aggregate_1km_to_25km(df_1km, df_25km_original, fwi_col)
        
        if df_aggregated.empty:
            print("ERROR: Aggregation failed, no valid data generated")
            return None, None
        
        # Save aggregation results
        agg_filename = f'fwi_aggregated_25km_{model_name.lower().replace(" ", "_")}.csv'
        df_aggregated.to_csv(agg_filename, index=False)
        print(f"Aggregation results saved as: {agg_filename}")
        
        # Merge data for validation
        print(f"\nMerging data for validation...")
        merged_df = merge_and_validate(df_aggregated, df_25km_original)
        
        if merged_df.empty:
            print("ERROR: Merging failed, no matching data")
            return None, None
        
        # Save merged results
        merged_filename = f'validation_merged_data_{model_name.lower().replace(" ", "_")}.csv'
        merged_df.to_csv(merged_filename, index=False)
        print(f"Validation data saved as: {merged_filename}")
        
        # Calculate validation metrics
        print(f"\nCalculating validation metrics...")
        metrics = calculate_metrics(merged_df['fwi'].values, merged_df['fwi_aggregated'].values)
        
        print(f"\nVALIDATION RESULTS FOR {model_name}:")
        print(f"{'='*50}")
        print(f"Correlation coefficient: {metrics['correlation']:.4f}")
        print(f"R-squared (R²): {metrics['r_squared']:.4f}")
        print(f"P-value: {metrics['p_value']:.6f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"Relative RMSE: {metrics['relative_rmse']:.2f}%")
        print(f"Relative MAE: {metrics['relative_mae']:.2f}%")
        print(f"Valid data points: {metrics['n_points']:,}")
        
        # Create validation plots
        print(f"\nGenerating validation plots...")
        create_validation_plots(merged_df, model_name, 'validation')
        
        # Additional summary statistics
        print(f"\nAdditional Statistics:")
        print(f"Original 25km FWI - Mean: {merged_df['fwi'].mean():.3f}, Std: {merged_df['fwi'].std():.3f}")
        print(f"Aggregated FWI   - Mean: {merged_df['fwi_aggregated'].mean():.3f}, Std: {merged_df['fwi_aggregated'].std():.3f}")
        
        # Spatial and temporal coverage
        if 'count_1km_points' in merged_df.columns:
            print(f"Average 1km points per 25km grid: {merged_df['count_1km_points'].mean():.1f}")
            print(f"Min/Max 1km points per grid: {merged_df['count_1km_points'].min()}/{merged_df['count_1km_points'].max()}")
        
        print(f"Number of spatial grids: {len(merged_df.groupby(['longitude', 'latitude']))}")
        
        if 'time' in merged_df.columns:
            print(f"Number of unique dates: {merged_df['time'].nunique()}")
            print(f"Date range: {merged_df['time'].min()} to {merged_df['time'].max()}")
        
        return merged_df, metrics
        
    except Exception as e:
        print(f"ERROR validating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_comparison_summary(all_results):
    """
    Create comparison summary across all models
    """
    print(f"\n{'='*80}")
    print("CREATING COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    if not all_results:
        print("No valid results to compare")
        return
    
    # Create comparison table
    comparison_data = []
    for model_name, (merged_df, metrics) in all_results.items():
        if metrics:
            comparison_data.append({
                'Model': model_name,
                'Correlation': metrics['correlation'],
                'R_squared': metrics['r_squared'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'Relative_RMSE': metrics['relative_rmse'],
                'Relative_MAE': metrics['relative_mae'],
                'Data_Points': metrics['n_points']
            })
    
    if not comparison_data:
        print("No valid metrics to compare")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('R_squared', ascending=False)
    
    # Save comparison table
    comparison_df.to_csv('model_comparison_summary.csv', index=False)
    print("Model comparison saved as: model_comparison_summary.csv")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    models = comparison_df['Model'].values
    
    # 1. R-squared comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, comparison_df['R_squared'], color=['forestgreen', 'purple', 'orange', 'red'][:len(models)])
    ax1.set_ylabel('R²')
    ax1.set_title('R-squared Comparison')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, comparison_df['R_squared']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 2. RMSE comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, comparison_df['RMSE'], color=['forestgreen', 'purple', 'orange', 'red'][:len(models)])
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, comparison_df['RMSE']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(comparison_df['RMSE'])*0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 3. Correlation comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, comparison_df['Correlation'], color=['forestgreen', 'purple', 'orange', 'red'][:len(models)])
    ax3.set_ylabel('Correlation')
    ax3.set_title('Correlation Comparison')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars3, comparison_df['Correlation']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 4. Data points comparison
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, comparison_df['Data_Points'], color=['forestgreen', 'purple', 'orange', 'red'][:len(models)])
    ax4.set_ylabel('Data Points')
    ax4.set_title('Valid Data Points')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars4, comparison_df['Data_Points']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(comparison_df['Data_Points'])*0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison plots saved as: model_comparison_plots.png")
    
    # Print summary table
    print(f"\nMODEL PERFORMANCE RANKING:")
    print(f"{'='*80}")
    print(f"{'Rank':<4} {'Model':<20} {'R²':<8} {'Corr':<8} {'RMSE':<8} {'MAE':<8} {'Points':<10}")
    print(f"{'-'*80}")
    
    for idx, row in comparison_df.iterrows():
        print(f"{idx+1:<4} {row['Model']:<20} {row['R_squared']:<8.4f} {row['Correlation']:<8.4f} "
              f"{row['RMSE']:<8.4f} {row['MAE']:<8.4f} {row['Data_Points']:<10,}")
    
    # Identify best model
    best_model = comparison_df.iloc[0]['Model']
    best_r2 = comparison_df.iloc[0]['R_squared']
    
    print(f"\nBEST PERFORMING MODEL: {best_model}")
    print(f"   R² Score: {best_r2:.4f}")
    
    return comparison_df

def main():
    """
    Main function to validate all four 1km prediction models
    """
    print("FWI 1km Predictions Validation Analysis")
    print("="*80)
    
    # Define the four prediction files and model names
    prediction_files = [
        'fwi_1km_predictions_gradient_boosting.csv',
        'fwi_1km_predictions_neural_network.csv', 
        'fwi_1km_predictions_random_forest.csv',
        'fwi_1km_predictions_ridge_regression.csv'
    ]
    
    model_names = [
        'Gradient Boosting',
        'Neural Network',
        'Random Forest',
        'Ridge Regression'
    ]
    
    # Load 25km reference data
    print("\nLoading 25km reference data...")
    reference_files = [
        'experiment/ERA5_reanalysis_fwi/era5_fwi_2017_portugal_3decimal.csv',
        'merged_fwi_25km_processed.csv',
        'era5_fwi_2017_portugal_3decimal.csv',
        'merged_25km_data.csv'
    ]
    
    df_25km_original = None
    for ref_file in reference_files:
        if os.path.exists(ref_file):
            print(f"Found reference file: {ref_file}")
            df_25km_original = pd.read_csv(ref_file)
            print(f"25km data shape: {df_25km_original.shape}")
            print(f"25km data columns: {list(df_25km_original.columns)}")
            break
    
    if df_25km_original is None:
        print("ERROR: No 25km reference data found!")
        print("Please ensure one of these files exists:")
        for ref_file in reference_files:
            print(f"  - {ref_file}")
        return
    
    # Check if FWI column exists in reference data
    fwi_ref_col = None
    for col in ['fwi', 'FWI', 'fire_weather_index']:
        if col in df_25km_original.columns:
            fwi_ref_col = col
            break
    
    if fwi_ref_col is None:
        print("ERROR: No FWI column found in reference data!")
        print(f"Available columns: {list(df_25km_original.columns)}")
        return
    else:
        print(f"Using FWI reference column: {fwi_ref_col}")
        # Standardize column name
        if fwi_ref_col != 'fwi':
            df_25km_original = df_25km_original.rename(columns={fwi_ref_col: 'fwi'})
    
    # Validate each model
    all_results = {}
    successful_validations = 0
    
    print(f"\nChecking prediction files...")
    for model_file, model_name in zip(prediction_files, model_names):
        if os.path.exists(model_file):
            print(f"Found: {model_file}")
        else:
            print(f"Missing: {model_file}")
    
    print(f"\nStarting validation process...")
    
    for model_file, model_name in zip(prediction_files, model_names):
        if os.path.exists(model_file):
            merged_df, metrics = validate_single_model(model_file, model_name, df_25km_original)
            if merged_df is not None and metrics is not None:
                all_results[model_name] = (merged_df, metrics)
                successful_validations += 1
            else:
                print(f"ERROR: Validation failed for {model_name}")
        else:
            print(f"ERROR: File not found: {model_file}")
    
    # Print validation summary
    print(f"\nVALIDATION SUMMARY:")
    print(f"{'='*50}")
    print(f"Total models to validate: {len(prediction_files)}")
    print(f"Successful validations: {successful_validations}")
    print(f"Failed validations: {len(prediction_files) - successful_validations}")
    
    # Create comparison summary if we have multiple successful validations
    if successful_validations > 1:
        print(f"\nCreating comparison analysis...")
        comparison_df = create_comparison_summary(all_results)
        
        # Save comprehensive summary
        summary = {
            'validation_summary': {
                'total_models': len(prediction_files),
                'successful_validations': successful_validations,
                'failed_validations': len(prediction_files) - successful_validations,
                'reference_data_shape': df_25km_original.shape,
                'reference_fwi_stats': df_25km_original['fwi'].describe().to_dict()
            },
            'model_results': {}
        }
        
        for model_name, (merged_df, metrics) in all_results.items():
            # Convert numpy types to regular Python types for JSON serialization
            clean_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    clean_metrics[key] = float(value)
                else:
                    clean_metrics[key] = value
            
            summary['model_results'][model_name] = {
                'metrics': clean_metrics,
                'data_size': len(merged_df),
                'spatial_coverage': {
                    'unique_coordinates': len(merged_df[['longitude', 'latitude']].drop_duplicates()),
                    'lat_range': [float(merged_df['latitude'].min()), float(merged_df['latitude'].max())],
                    'lon_range': [float(merged_df['longitude'].min()), float(merged_df['longitude'].max())]
                },
                'data_distribution': {
                    'fwi_original': {k: float(v) for k, v in merged_df['fwi'].describe().to_dict().items()},
                    'fwi_aggregated': {k: float(v) for k, v in merged_df['fwi_aggregated'].describe().to_dict().items()}
                }
            }
        
        # Save summary to JSON file
        with open('validation_comprehensive_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Comprehensive summary saved: validation_comprehensive_summary.json")
        
    elif successful_validations == 1:
        print(f"\nOnly one model validated successfully. Individual results available.")
        model_name = list(all_results.keys())[0]
        merged_df, metrics = all_results[model_name]
        
        print(f"\nSINGLE MODEL VALIDATION RESULTS:")
        print(f"Model: {model_name}")
        print(f"R²: {metrics['r_squared']:.4f}")
        print(f"Correlation: {metrics['correlation']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"Data points: {metrics['n_points']:,}")
        
    else:
        print(f"\nNo successful validations completed.")
        print(f"Please check:")
        print(f"  1. Prediction files exist and are readable")
        print(f"  2. Files contain required columns (latitude, longitude, fwi_predicted)")
        print(f"  3. Reference data is available and contains FWI column")
        return
    
    # List all generated files
    print(f"\nGenerated files:")
    output_files = [
        'model_comparison_summary.csv',
        'model_comparison_plots.png',
        'validation_comprehensive_summary.json'
    ]
    
    # Add model-specific files
    for model_name in all_results.keys():
        model_safe_name = model_name.lower().replace(" ", "_")
        output_files.extend([
            f'fwi_aggregated_25km_{model_safe_name}.csv',
            f'validation_merged_data_{model_safe_name}.csv',
            f'validation_{model_safe_name}_plots.png'
        ])
    
    for file in output_files:
        if os.path.exists(file):
            print(f"  {file}")
        else:
            print(f"  {file} (not created)")
    
    print(f"\nVALIDATION ANALYSIS COMPLETED!")
    print(f"="*60)
    
    if successful_validations > 0:
        print(f"{successful_validations} model(s) successfully validated")
        if successful_validations > 1:
            best_model = comparison_df.iloc[0]['Model'] if 'comparison_df' in locals() else "Unknown"
            best_r2 = comparison_df.iloc[0]['R_squared'] if 'comparison_df' in locals() else 0
            print(f"Best performing model: {best_model} (R² = {best_r2:.4f})")
        print(f"Check the generated files for detailed results and visualizations")
    else:
        print(f"No models were successfully validated")
        print(f"Please check the input files and try again")


if __name__ == "__main__":
    main()