#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nc_to_csv.py

Convert UERRA MESCAN-SURFEX netCDF files to a single CSV table by day,
keeping only 2015–2018 data for Portugal sub-region (36°–43°N, –10°–-6°E).
Rounds coordinates to 3 decimal places and averages duplicate values.
"""
import os
import glob
import xarray as xr
import pandas as pd
import numpy as np

# Input/output paths - current folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NC_DIR     = SCRIPT_DIR
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "uerra_2017_PT_3decimal.csv")

# Variables to export (will be filtered based on availability)
DESIRED_VARS = ["si10", "r2", "t2m", "tp"]

# Common alternative variable names in UERRA datasets
VAR_ALIASES = {
    "si10": ["si10", "10m_wind_speed", "ws10", "windspeed"],
    "r2": ["r2", "2m_relative_humidity", "rh2m", "rh"],
    "t2m": ["t2m", "2m_temperature", "temp2m", "temperature"],
    "tp": ["tp", "total_precipitation", "precip", "pr", "precipitation", "pcp", "mtpr", "rainfall"]
}

# Portugal sub-region boundaries
LAT_MIN, LAT_MAX = 36.8, 42.2
LON_MIN, LON_MAX = -9.6, -6.2

# Only process these years
YEARS = {"2017"}

def nc_files_list(nc_dir):
    """List all .nc files in directory"""
    return sorted(glob.glob(os.path.join(nc_dir, "*.nc")))

def detailed_dataset_inspection(nc_file):
    """Detailed inspection of the dataset"""
    print(f"\n{'='*60}")
    print(f"DETAILED DATASET INSPECTION: {os.path.basename(nc_file)}")
    print(f"{'='*60}")
    
    ds = xr.open_dataset(nc_file)
    
    print(f"File size: {os.path.getsize(nc_file) / (1024*1024):.2f} MB")
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Coordinates: {list(ds.coords.keys())}")
    print(f"Data variables: {list(ds.data_vars.keys())}")
    
    print(f"\nAll Variables (including coordinates):")
    all_vars = list(ds.coords.keys()) + list(ds.data_vars.keys())
    for i, var in enumerate(all_vars, 1):
        print(f"  {i:2d}. {var}")
    
    print(f"\nDetailed Variable Information:")
    for var_name in ds.data_vars:
        var = ds[var_name]
        print(f"\n  Variable: {var_name}")
        print(f"    Shape: {var.shape}")
        print(f"    Dimensions: {var.dims}")
        print(f"    Data type: {var.dtype}")
        print(f"    Attributes:")
        for attr, value in var.attrs.items():
            print(f"      {attr}: {value}")
        
        # Show some sample values
        if var.size > 0:
            sample_vals = var.values.flat[:5]
            print(f"    Sample values: {sample_vals}")
    
    print(f"\nGlobal Attributes:")
    for attr, value in ds.attrs.items():
        print(f"  {attr}: {value}")
    
    ds.close()
    print(f"{'='*60}")

def find_available_vars(ds):
    """Find which variables are available in the dataset"""
    available_vars = {}
    dataset_vars = list(ds.data_vars.keys())
    
    print(f"\n  Available data variables: {dataset_vars}")
    
    for desired_var in DESIRED_VARS:
        found_var = None
        # Check if exact match exists
        if desired_var in dataset_vars:
            found_var = desired_var
        else:
            # Check aliases
            for alias in VAR_ALIASES.get(desired_var, []):
                if alias in dataset_vars:
                    found_var = alias
                    break
            
            # If still not found, check case-insensitive and partial matches
            if found_var is None:
                for var in dataset_vars:
                    var_lower = var.lower()
                    if desired_var == "tp":
                        if any(keyword in var_lower for keyword in ["precip", "rain", "tp", "pr", "pcp"]):
                            found_var = var
                            break
                    elif desired_var == "si10":
                        if any(keyword in var_lower for keyword in ["wind", "si10", "ws10"]):
                            found_var = var
                            break
                    elif desired_var == "r2":
                        if any(keyword in var_lower for keyword in ["humid", "rh", "r2"]):
                            found_var = var
                            break
                    elif desired_var == "t2m":
                        if any(keyword in var_lower for keyword in ["temp", "t2m"]):
                            found_var = var
                            break
        
        if found_var:
            available_vars[desired_var] = found_var
            print(f"  ✓ {desired_var} -> {found_var}")
        else:
            print(f"  ✗ {desired_var} not found")
    
    return available_vars

def check_precipitation_possibility(ds):
    """Check if precipitation data might be available under different names"""
    print(f"\n  Checking for precipitation-related variables...")
    dataset_vars = list(ds.data_vars.keys())
    precip_keywords = ["precip", "rain", "tp", "pr", "pcp", "mtpr", "rainfall"]
    
    potential_precip_vars = []
    for var in dataset_vars:
        var_lower = var.lower()
        for keyword in precip_keywords:
            if keyword in var_lower:
                potential_precip_vars.append(var)
                break
    
    if potential_precip_vars:
        print(f"  Potential precipitation variables found: {potential_precip_vars}")
    else:
        print(f"  No precipitation variables found with keywords: {precip_keywords}")
        print(f"  This dataset may not include precipitation data.")
    
    return potential_precip_vars

def process_and_append(nc_file, csv_path, first_write):
    """Process single netCDF file and append to CSV"""
    # Detailed inspection first
    detailed_dataset_inspection(nc_file)
    
    # Open dataset
    ds = xr.open_dataset(nc_file, engine="netcdf4")
    
    print(f"\nProcessing {os.path.basename(nc_file)}...")
    
    # Check for precipitation variables
    potential_precip = check_precipitation_possibility(ds)
    
    # Find available variables
    available_vars = find_available_vars(ds)
    
    if not available_vars:
        print(f"  ⚠ No required variables found, skipping")
        ds.close()
        return first_write
    
    print(f"\n  Final variable mapping: {available_vars}")

    # Convert longitude from [0,360) to [-180,180) if needed
    if ds.longitude.max() > 180:
        ds = ds.assign_coords(
            longitude = (((ds.longitude + 180) % 360) - 180)
        )
        print(f"  ✓ Converted longitude coordinates")

    # Find time coordinate
    time_dim = next((c for c, vals in ds.coords.items()
                     if np.issubdtype(vals.dtype, np.datetime64)), None)
    if time_dim is None:
        print("  ⚠ No time coordinate found, skipping")
        ds.close()
        return first_write

    print(f"  ✓ Time coordinate: {time_dim}")
    print(f"  ✓ Time range: {ds[time_dim].values[0]} to {ds[time_dim].values[-1]}")

    # Show original coordinate precision
    print(f"  ✓ Original latitude range: {ds.latitude.values.min():.6f} to {ds.latitude.values.max():.6f}")
    print(f"  ✓ Original longitude range: {ds.longitude.values.min():.6f} to {ds.longitude.values.max():.6f}")

    # Process each time step
    time_count = 0
    for t in ds[time_dim].values:
        ds_t = ds.sel({time_dim: t})
        
        # Get lat/lon coordinates
        if ds_t.latitude.ndim == 1 and ds_t.longitude.ndim == 1:
            # 1D coordinates - create meshgrid
            lon_1d = ds_t.longitude.values
            lat_1d = ds_t.latitude.values
            lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
            lat_flat = lat_2d.flatten()
            lon_flat = lon_2d.flatten()
        else:
            # 2D coordinates
            lat_flat = ds_t["latitude"].values.flatten()
            lon_flat = ds_t["longitude"].values.flatten()
        
        # Round coordinates to 3 decimal places
        lat_rounded = np.round(lat_flat, 3)
        lon_rounded = np.round(lon_flat, 3)
        
        # Extract variable data
        data = {}
        for desired_var, actual_var in available_vars.items():
            data[desired_var] = ds_t[actual_var].values.flatten()

        df = pd.DataFrame({
            "time":      [t] * lat_flat.size,
            "latitude":  lat_rounded,
            "longitude": lon_rounded,
            **data
        })
        
        # Filter Portugal sub-region
        df = df[
            (df.latitude  >= LAT_MIN) & (df.latitude  <= LAT_MAX) &
            (df.longitude >= LON_MIN)   & (df.longitude <= LON_MAX)
        ]
        
        # Remove NaN values
        df = df.dropna()
        
        if df.empty:
            continue

        # Additional rounding to ensure 3 decimal places precision
        df['latitude'] = df['latitude'].round(3)
        df['longitude'] = df['longitude'].round(3)

        # Group by time, latitude, longitude and take average of duplicate coordinates
        groupby_cols = ['time', 'latitude', 'longitude']
        value_cols = [col for col in df.columns if col not in groupby_cols]
        
        if value_cols:
            df_averaged = df.groupby(groupby_cols)[value_cols].mean().reset_index()
            
            # Show how many duplicates were found
            original_count = len(df)
            averaged_count = len(df_averaged)
            if original_count > averaged_count:
                print(f"  ✓ Averaged {original_count} points to {averaged_count} points (removed {original_count - averaged_count} duplicates) - coordinates rounded to 3 decimal places")
        else:
            df_averaged = df.drop_duplicates(subset=groupby_cols)
        
        df_averaged.to_csv(
            csv_path,
            mode='w' if first_write else 'a',
            header=first_write,
            index=False
        )
        first_write = False
        time_count += 1

    print(f"  ✓ Processed {time_count} time steps")
    print(f"  ✓ Variables included in output: {list(available_vars.keys())}")
    print(f"  ✓ Coordinates rounded to 3 decimal places")
    ds.close()
    return first_write

def create_summary_statistics(csv_path):
    """Create summary statistics from the output CSV - Display only, no file output"""
    print(f"\nCreating summary statistics...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df['time'].min()} to {df['time'].max()}")
        print(f"Latitude range: {df['latitude'].min():.3f} to {df['latitude'].max():.3f}")
        print(f"Longitude range: {df['longitude'].min():.3f} to {df['longitude'].max():.3f}")
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Create summary statistics - only display, don't save
        summary = df[numeric_columns].describe()
        
        print(f"\nSummary statistics:")
        print(summary)
        
        # Show coordinate precision
        print(f"\nCoordinate precision verification:")
        print(f"Unique latitudes: {df['latitude'].nunique()}")
        print(f"Unique longitudes: {df['longitude'].nunique()}")
        print(f"Sample coordinates (first 5):")
        print(df[['latitude', 'longitude']].head())
        
    except Exception as e:
        print(f"Error creating summary statistics: {e}")

def main():
    files = nc_files_list(NC_DIR)
    # Only keep files that contain target years in filename
    files = [f for f in files if any(year in os.path.basename(f) for year in YEARS)]
    
    print(f"Found {len(files)} files for years {sorted(YEARS)}. Start processing...")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Coordinate precision: 3 decimal places")
    
    first = True
    for nc in files:
        first = process_and_append(nc, OUTPUT_CSV, first)
    
    print(f"✅ Done. Subset CSV saved to: {OUTPUT_CSV}")
    
    # Create summary statistics if CSV was created - only display, no file output
    if os.path.exists(OUTPUT_CSV):
        create_summary_statistics(OUTPUT_CSV)
        
        # Show only the main output file
        file_size = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)  # MB
        print(f"\nMain output file:")
        print(f"  - {os.path.basename(OUTPUT_CSV)} ({file_size:.1f} MB)")
        print(f"  - Full path: {OUTPUT_CSV}")
    else:
        print("No data was processed.")

if __name__ == "__main__":
    main()