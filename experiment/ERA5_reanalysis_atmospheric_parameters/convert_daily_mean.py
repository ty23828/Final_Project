#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5 Daily Mean Data NetCDF to CSV Converter - 2017 Data Only

Convert ERA5 daily mean 2017 NetCDF files to a single combined CSV format.
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import warnings
import zipfile
import tempfile
import shutil
warnings.filterwarnings('ignore')

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..")  
OUTPUT_DIR = SCRIPT_DIR  # Save directly to script directory, no subfolder

# No need to create output directory since we're using the script directory
# os.makedirs(OUTPUT_DIR, exist_ok=True)  # Remove this line

# Required variables - these are what we need
REQUIRED_VARS = {
    "10m_u_component_of_wind": ["u10", "10m_u_component_of_wind", "u_component_of_wind_10m", "wind_u_10m"],
    "10m_v_component_of_wind": ["v10", "10m_v_component_of_wind", "v_component_of_wind_10m", "wind_v_10m"],
    "2m_dewpoint_temperature": ["d2m", "2m_dewpoint_temperature", "dewpoint_temperature_2m", "dewpoint_temp_2m"],
    "total_precipitation": ["tp", "total_precipitation", "precipitation", "precip"]
}

# Variable mappings for cleaner column names
VARIABLE_MAPPING = {
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind', 
    'd2m': '2m_dewpoint_temperature',
    't2m': '2m_air_temperature',
    'tp': 'total_precipitation',
    'sp': 'surface_pressure',
    'msl': 'mean_sea_level_pressure'
}

def detect_file_type(file_path):
    """Detect if file is ZIP or NetCDF"""
    with open(file_path, "rb") as f:
        magic = f.read(4)
    
    if magic == b"PK\x03\x04":
        return "zip"
    elif magic == b"CDF\x01" or magic == b"CDF\x02":
        return "netcdf"
    elif magic.startswith(b"\x89HDF"):
        return "hdf5"
    else:
        return "unknown"

def open_netcdf_files_from_zip(zip_path):
    """Extract all NetCDF files from ZIP and return dataset information"""
    extracted_datasets = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"    ZIP contents: {file_list}")
            
            # Find all .nc files
            nc_files = [f for f in file_list if f.lower().endswith('.nc')]
            if not nc_files:
                raise ValueError(f"No .nc files found in ZIP archive: {zip_path}")
            
            print(f"    Found {len(nc_files)} NetCDF files in ZIP")
            
            # Extract and open each NetCDF file
            for nc_file in nc_files:
                print(f"    Extracting: {nc_file}")
                extracted_path = zip_ref.extract(nc_file, temp_dir)
                
                try:
                    # Try to open with netcdf4 engine first
                    ds = xr.open_dataset(extracted_path, engine="netcdf4")
                    extracted_datasets.append({
                        'dataset': ds,
                        'filename': nc_file,
                        'path': extracted_path
                    })
                    print(f"      ✓ Opened {nc_file} successfully")
                except Exception as e:
                    print(f"      Trying scipy engine for {nc_file}...")
                    try:
                        ds = xr.open_dataset(extracted_path, engine="scipy")
                        extracted_datasets.append({
                            'dataset': ds,
                            'filename': nc_file,
                            'path': extracted_path
                        })
                        print(f"      ✓ Opened {nc_file} with scipy engine")
                    except Exception as e2:
                        print(f"      ✗ Failed to open {nc_file}: {e2}")
                        continue
        
        return extracted_datasets, temp_dir
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e

def cleanup_temp_files(temp_dir):
    """Clean up temporary directory"""
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"    ✓ Cleaned up temporary files")
        except Exception as e:
            print(f"    Warning: Could not clean up temporary files: {e}")

def find_variable_mappings(ds, filename=""):
    """Find which variables are available in the dataset"""
    available_vars = {}
    dataset_vars = list(ds.data_vars.keys())
    
    print(f"      Variables in dataset: {dataset_vars}")
    
    for required_var, possible_names in REQUIRED_VARS.items():
        found_var = None
        for possible_name in possible_names:
            if possible_name in dataset_vars:
                found_var = possible_name
                break
        
        if found_var:
            available_vars[required_var] = found_var
            print(f"      ✓ Found {required_var} -> {found_var}")
        else:
            # Check if variable name is in the filename
            filename_lower = filename.lower()
            for possible_name in possible_names:
                if possible_name in filename_lower:
                    # The variable might be the only data variable
                    if len(dataset_vars) == 1:
                        found_var = dataset_vars[0]
                        available_vars[required_var] = found_var
                        print(f"      ✓ Inferred {required_var} -> {found_var} (from filename)")
                        break
            
            # Additional inference based on filename patterns
            if not found_var:
                if "10m_u_component_of_wind" in filename_lower:
                    available_vars["10m_u_component_of_wind"] = dataset_vars[0] if dataset_vars else None
                    print(f"      ✓ Inferred 10m_u_component_of_wind -> {dataset_vars[0]} (from filename)")
                elif "10m_v_component_of_wind" in filename_lower:
                    available_vars["10m_v_component_of_wind"] = dataset_vars[0] if dataset_vars else None
                    print(f"      ✓ Inferred 10m_v_component_of_wind -> {dataset_vars[0]} (from filename)")
                elif "2m_dewpoint_temperature" in filename_lower:
                    available_vars["2m_dewpoint_temperature"] = dataset_vars[0] if dataset_vars else None
                    print(f"      ✓ Inferred 2m_dewpoint_temperature -> {dataset_vars[0]} (from filename)")
                elif "total_precipitation" in filename_lower:
                    available_vars["total_precipitation"] = dataset_vars[0] if dataset_vars else None
                    print(f"      ✓ Inferred total_precipitation -> {dataset_vars[0]} (from filename)")
    
    return available_vars

def list_era5_2017_files(input_dir):
    """List all ERA5 2017 NetCDF files in the input directory and subdirectories"""
    patterns = ['*.nc', '*.netcdf', '*.NC']
    files = []
    
    print(f"Searching for ERA5 2017 NetCDF files in: {os.path.abspath(input_dir)}")
    
    # Search in the main directory and subdirectories
    for root, dirs, filenames in os.walk(input_dir):
        for pattern in patterns:
            found_files = glob.glob(os.path.join(root, pattern))
            # Filter for files containing "era5_daily_mean_2017"
            era5_2017_files = [f for f in found_files if "era5_daily_mean_2017" in os.path.basename(f).lower()]
            files.extend(era5_2017_files)
    
    # Remove duplicates and sort
    files = list(set(files))
    files.sort()
    
    print(f"Found {len(files)} ERA5 2017 NetCDF files:")
    for i, file in enumerate(files, 1):
        relative_path = os.path.relpath(file, input_dir)
        file_size = os.path.getsize(file) / (1024 * 1024)  # MB
        file_type = detect_file_type(file)
        print(f"  {i:2d}. {relative_path} ({file_size:.1f} MB) - {file_type}")
    
    return files

def inspect_netcdf_file(file_path):
    """Inspect NetCDF file structure"""
    print(f"\nInspecting: {os.path.basename(file_path)}")
    print("-" * 50)
    
    file_type = detect_file_type(file_path)
    
    if file_type == "zip":
        try:
            extracted_datasets, temp_dir = open_netcdf_files_from_zip(file_path)
            
            print(f"ZIP file contains {len(extracted_datasets)} NetCDF files:")
            
            for i, ds_info in enumerate(extracted_datasets, 1):
                ds = ds_info['dataset']
                filename = ds_info['filename']
                
                print(f"\n  File {i}: {filename}")
                print(f"    Dimensions: {dict(ds.dims)}")
                print(f"    Variables: {list(ds.data_vars.keys())}")
                
                # Find variable mappings
                available_vars = find_variable_mappings(ds, filename)
                
                # Check coordinates
                if 'time' in ds.dims:
                    time_data = ds['time'].values
                    if len(time_data) > 0:
                        print(f"    Time range: {pd.to_datetime(time_data[0])} to {pd.to_datetime(time_data[-1])}")
                
                # Close dataset
                ds.close()
            
            cleanup_temp_files(temp_dir)
            
        except Exception as e:
            print(f"Error inspecting ZIP file: {e}")
    
    else:
        try:
            ds = xr.open_dataset(file_path, engine="netcdf4")
            
            print("Dimensions:")
            for dim, size in ds.dims.items():
                print(f"  {dim}: {size}")
            
            print("\nVariables:")
            for var in ds.data_vars:
                shape = ds[var].shape
                dtype = ds[var].dtype
                print(f"  {var}: {shape} ({dtype})")
            
            available_vars = find_variable_mappings(ds)
            
            ds.close()
            
        except Exception as e:
            print(f"Error inspecting file: {e}")

def process_zip_file(zip_path, round_coords=3):
    """Process a single ZIP file containing multiple NetCDF files"""
    print(f"  Processing ZIP: {os.path.basename(zip_path)}")
    
    try:
        extracted_datasets, temp_dir = open_netcdf_files_from_zip(zip_path)
        
        # Collect data from all NetCDF files in this ZIP
        zip_data = {}  # {(time, lat, lon): {var1: value1, var2: value2, ...}}
        
        for ds_info in extracted_datasets:
            ds = ds_info['dataset']
            filename = ds_info['filename']
            
            print(f"    Processing NetCDF: {filename}")
            
            # Find variable mappings
            available_vars = find_variable_mappings(ds, filename)
            
            if not available_vars:
                print(f"      No required variables found in {filename}")
                ds.close()
                continue
            
            # Convert to DataFrame
            df = ds.to_dataframe()
            df = df.reset_index()
            
            print(f"      DataFrame shape: {df.shape}")
            print(f"      DataFrame columns: {df.columns.tolist()}")
            
            # Handle different time column names
            time_columns = ['time', 'valid_time', 'datetime', 'date']
            time_col = None
            for col in time_columns:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col is None:
                print(f"      No time column found in {filename}")
                ds.close()
                continue
            
            # Rename time column to standard name
            if time_col != 'time':
                df['time'] = df[time_col]
                print(f"      Renamed {time_col} to time")
            
            # Check if we have the necessary columns
            required_columns = ['time', 'latitude', 'longitude']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                # Try alternative column names
                if 'lat' in df.columns and 'latitude' not in df.columns:
                    df['latitude'] = df['lat']
                if 'lon' in df.columns and 'longitude' not in df.columns:
                    df['longitude'] = df['lon']
                
                # Check again
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"      Missing required columns: {missing_columns}")
                    ds.close()
                    continue
            
            print(f"      ✓ All required columns found: time, latitude, longitude")
            
            # Round coordinates to 3 decimal places
            if round_coords is not None:
                df['longitude'] = df['longitude'].round(round_coords)
                df['latitude'] = df['latitude'].round(round_coords)
            
            # Convert time to datetime
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
            
            # Remove rows with NaN values in essential columns
            essential_columns = ['time', 'latitude', 'longitude']
            df = df.dropna(subset=essential_columns)
            
            print(f"      After cleaning, DataFrame shape: {df.shape}")
            
            # Extract data for each available variable
            for required_var, actual_var in available_vars.items():
                if actual_var in df.columns:
                    # Remove rows where the variable value is NaN
                    df_var = df.dropna(subset=[actual_var])
                    
                    for _, row in df_var.iterrows():
                        try:
                            key = (row['time'], row['latitude'], row['longitude'])
                            if key not in zip_data:
                                zip_data[key] = {}
                            zip_data[key][required_var] = row[actual_var]
                        except Exception as e:
                            print(f"        Error processing row: {e}")
                            continue
            
            print(f"      Added data with variables: {list(available_vars.keys())}")
            ds.close()
        
        cleanup_temp_files(temp_dir)
        
        print(f"    ZIP processed: {len(zip_data)} unique time-location combinations")
        
        # Show sample of data
        if zip_data:
            sample_keys = list(zip_data.keys())[:3]
            print(f"    Sample data:")
            for key in sample_keys:
                print(f"      {key}: {zip_data[key]}")
        
        return zip_data
        
    except Exception as e:
        print(f"    Error processing ZIP {os.path.basename(zip_path)}: {e}")
        import traceback
        traceback.print_exc()
        return {}

def combine_era5_2017_to_csv(files, output_dir, round_coords=3):
    """Convert multiple ERA5 2017 NetCDF files to a single combined CSV"""
    output_path = os.path.join(output_dir, "era5_daily_mean_2017_combined_3decimal.csv")
    
    if os.path.exists(output_path):
        print(f"Combined CSV already exists: {os.path.basename(output_path)}")
        response = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if response != 'y':
            return output_path
    
    print(f"Creating combined CSV from {len(files)} ERA5 2017 files...")
    
    # Collect all data from all files
    all_data = {}  # {(time, lat, lon): {var1: value1, var2: value2, ...}}
    
    for file_path in files:
        file_type = detect_file_type(file_path)
        
        if file_type == "zip":
            zip_data = process_zip_file(file_path, round_coords)
            
            # Merge zip data into all_data
            for key, variables in zip_data.items():
                if key not in all_data:
                    all_data[key] = {}
                all_data[key].update(variables)
        
        else:
            # Handle direct NetCDF files (if any)
            print(f"  Processing NetCDF: {os.path.basename(file_path)}")
            # Similar processing for direct NetCDF files...
    
    if all_data:
        print(f"\nCombining data from {len(all_data)} unique time-location combinations...")
        
        # Convert to DataFrame
        rows = []
        for (time, lat, lon), variables in all_data.items():
            row = {
                'time': time,
                'latitude': lat,
                'longitude': lon,
                'date': time.date(),
                'year': time.year,
                'month': time.month,
                'day': time.day
            }
            row.update(variables)
            rows.append(row)
        
        combined_df = pd.DataFrame(rows)
        
        # Sort by time
        combined_df = combined_df.sort_values('time')
        
        # Save to CSV
        print(f"Saving combined CSV...")
        combined_df.to_csv(output_path, index=False)
        
        print(f"\nCombined CSV saved: {os.path.basename(output_path)}")
        print(f"Total rows: {len(combined_df):,}")
        print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        
        # Show column information
        print(f"\nColumns in combined CSV:")
        for col in combined_df.columns:
            print(f"  - {col}")
        
        # Show coordinate precision
        print(f"\nCoordinate precision verification:")
        print(f"Longitude range: {combined_df['longitude'].min():.3f} to {combined_df['longitude'].max():.3f}")
        print(f"Latitude range: {combined_df['latitude'].min():.3f} to {combined_df['latitude'].max():.3f}")
        print(f"Unique coordinates: {combined_df[['longitude', 'latitude']].drop_duplicates().shape[0]}")
        
        # Show variable coverage
        print(f"\nVariable coverage:")
        for var in REQUIRED_VARS.keys():
            if var in combined_df.columns:
                non_null_count = combined_df[var].notna().sum()
                print(f"  {var}: {non_null_count}/{len(combined_df)} ({non_null_count/len(combined_df)*100:.1f}%)")
        
        # Show sample data
        print(f"\nSample data (first 5 rows):")
        print(combined_df.head())
        
        return output_path
    else:
        print("No data to combine!")
        return None

def create_summary_statistics(csv_path):
    """Create summary statistics from CSV data - Display only, no file output"""
    print(f"\nCreating summary statistics...")
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Create summary statistics - only display, don't save
        summary = df[numeric_columns].describe()
        
        print(f"\nSummary statistics:")
        print(summary)
        
        # Show monthly averages - only display, don't save
        if 'year' in df.columns and 'month' in df.columns:
            monthly_avg = df.groupby(['year', 'month'])[numeric_columns].mean()
            print(f"\nMonthly averages:")
            print(monthly_avg)
        
    except Exception as e:
        print(f"Error creating summary statistics: {e}")

def main():
    """Main conversion function"""
    print("ERA5 Daily Mean 2017 NetCDF to CSV Converter (3 Decimal Places)")
    print("=" * 60)
    
    print(f"Required variables:")
    for var in REQUIRED_VARS.keys():
        print(f"  - {var}")
    print()
    
    print(f"Input directory: {os.path.abspath(INPUT_DIR)}")
    print(f"Output will be saved directly to: {os.path.abspath(OUTPUT_DIR)}")
    
    # List ERA5 2017 NetCDF files
    netcdf_files = list_era5_2017_files(INPUT_DIR)
    
    if not netcdf_files:
        print("No ERA5 2017 NetCDF files found!")
        print("Please make sure the files with 'era5_daily_mean_2017' in their names are in the correct location.")
        return
    
    # Inspect first file
    inspect_netcdf_file(netcdf_files[0])
    
    # Combine all files to single CSV (with 3 decimal places)
    print(f"\nCombining all {len(netcdf_files)} ERA5 2017 files into single CSV...")
    combined_path = combine_era5_2017_to_csv(netcdf_files, OUTPUT_DIR, round_coords=3)
    
    if combined_path:
        create_summary_statistics(combined_path)  # Only display, no file output
    
    print(f"\nConversion completed!")
    print(f"Output file saved directly to script directory:")
    
    # Only show the main CSV file
    main_csv = os.path.join(OUTPUT_DIR, "era5_daily_mean_2017_combined_3decimal.csv")
    if os.path.exists(main_csv):
        file_size = os.path.getsize(main_csv) / (1024 * 1024)  # MB
        print(f"  - {os.path.basename(main_csv)} ({file_size:.1f} MB)")
        print(f"  - Full path: {main_csv}")
    else:
        print("  - Main CSV file not found!")

if __name__ == "__main__":
    main()