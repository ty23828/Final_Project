#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convertdailymaxtemp.py

Convert ERA5 daily maximum temperature NetCDF files to CSV format
for Portugal sub-region (36.8°–42.2°N, -9.6°–-6.2°E).

Input: era5_daily_max_temp_YYYY_MM.nc files
Output: era5_daily_max_temp_2017_portugal.csv (combined CSV)
"""

import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = SCRIPT_DIR
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "era5_daily_max_temp_2017_portugal.csv")

# Portugal sub-region boundaries
LAT_MIN, LAT_MAX = 36.8, 42.2
LON_MIN, LON_MAX = -9.6, -6.2

# Years to process
YEARS = {"2017"}

def list_temp_files(directory):
    """List all temperature NetCDF files in directory"""
    pattern = os.path.join(directory, "era5_daily_max_temp_*.nc")
    files = glob.glob(pattern)
    # Filter by year
    filtered_files = []
    for f in files:
        filename = os.path.basename(f)
        if any(year in filename for year in YEARS):
            filtered_files.append(f)
    return sorted(filtered_files)

def convert_temperature_file(nc_file):
    """Convert single temperature NetCDF file to DataFrame"""
    print(f"Processing {os.path.basename(nc_file)}...")
    
    try:
        # Try different engines
        ds = None
        for engine in [None, "netcdf4", "h5netcdf"]:
            try:
                if engine:
                    ds = xr.open_dataset(nc_file, engine=engine)
                else:
                    ds = xr.open_dataset(nc_file)
                print(f"  Successfully opened with engine: {engine or 'default'}")
                break
            except Exception as e:
                print(f"  Failed with engine {engine or 'default'}: {str(e)}")
                continue
        
        if ds is None:
            print(f"  Could not open {nc_file} with any engine")
            return pd.DataFrame()
        
        with ds:
            print(f"  Dataset dimensions: {dict(ds.dims)}")
            print(f"  Variables: {list(ds.data_vars.keys())}")
            print(f"  Coordinates: {list(ds.coords.keys())}")
            
            # Find temperature variable
            temp_var = None
            possible_temp_vars = ['t2m', '2m_temperature', 'temperature', 'temp', 'T2M']
            
            for var_name in ds.data_vars:
                if any(temp_key in var_name.lower() for temp_key in ['t2m', 'temperature', '2m']):
                    temp_var = var_name
                    break
            
            if temp_var is None:
                # Try exact matches
                for var_name in possible_temp_vars:
                    if var_name in ds.data_vars:
                        temp_var = var_name
                        break
            
            if temp_var is None:
                print(f"  Warning: No temperature variable found. Available variables: {list(ds.data_vars.keys())}")
                return pd.DataFrame()
            
            print(f"  Temperature variable: {temp_var}")
            
            # Find coordinate names
            lat_coord = None
            lon_coord = None
            time_coord = None
            
            for coord_name in ds.coords:
                coord_lower = coord_name.lower()
                if 'lat' in coord_lower:
                    lat_coord = coord_name
                elif 'lon' in coord_lower:
                    lon_coord = coord_name
                elif 'time' in coord_lower or np.issubdtype(ds[coord_name].dtype, np.datetime64):
                    time_coord = coord_name
            
            print(f"  Coordinates - Lat: {lat_coord}, Lon: {lon_coord}, Time: {time_coord}")
            
            if not all([lat_coord, lon_coord, time_coord]):
                print(f"  Warning: Missing coordinates. Available: {list(ds.coords.keys())}")
                return pd.DataFrame()
            
            # Convert longitude from [0,360) to [-180,180) if needed
            if ds[lon_coord].max() > 180:
                ds = ds.assign_coords({lon_coord: (((ds[lon_coord] + 180) % 360) - 180)})
                print(f"  Converted longitude coordinates")
            
            print(f"  Time range: {ds[time_coord].values[0]} to {ds[time_coord].values[-1]}")
            
            # Process each time step
            all_data = []
            
            for time_val in ds[time_coord].values:
                # Select data for this time step
                ds_time = ds.sel({time_coord: time_val})
                
                # Get coordinates
                if ds_time[lat_coord].ndim == 1 and ds_time[lon_coord].ndim == 1:
                    # 1D coordinates - create meshgrid
                    lon_1d = ds_time[lon_coord].values
                    lat_1d = ds_time[lat_coord].values
                    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
                    lat_flat = lat_2d.flatten()
                    lon_flat = lon_2d.flatten()
                else:
                    # 2D coordinates
                    lat_flat = ds_time[lat_coord].values.flatten()
                    lon_flat = ds_time[lon_coord].values.flatten()
                
                # Get temperature data
                temp_data = ds_time[temp_var].values.flatten()
                
                # Create DataFrame for this time step
                df_time = pd.DataFrame({
                    'time': [pd.to_datetime(time_val)] * len(lat_flat),
                    'latitude': lat_flat,
                    'longitude': lon_flat,
                    'temperature_max': temp_data
                })
                
                # Filter Portugal sub-region
                df_time = df_time[
                    (df_time.latitude >= LAT_MIN) & (df_time.latitude <= LAT_MAX) &
                    (df_time.longitude >= LON_MIN) & (df_time.longitude <= LON_MAX)
                ]
                
                # Remove NaN values
                df_time = df_time.dropna()
                
                if not df_time.empty:
                    all_data.append(df_time)
            
            # Combine all time steps
            if all_data:
                df_combined = pd.concat(all_data, ignore_index=True)
                print(f"  Processed {len(df_combined)} data points")
                return df_combined
            else:
                print(f"  No valid data found")
                return pd.DataFrame()
                
    except Exception as e:
        print(f"  Error processing {nc_file}: {str(e)}")
        return pd.DataFrame()

def main():
    """Main conversion function"""
    print("ERA5 Daily Maximum Temperature NetCDF to CSV Converter")
    print("=" * 60)
    print(f"Processing directory: {INPUT_DIR}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Portugal bounds: Lat[{LAT_MIN}, {LAT_MAX}], Lon[{LON_MIN}, {LON_MAX}]")
    print(f"Years: {sorted(YEARS)}")
    print("-" * 60)
    
    # Find all temperature files
    temp_files = list_temp_files(INPUT_DIR)
    
    if not temp_files:
        print("No temperature NetCDF files found!")
        return
    
    print(f"Found {len(temp_files)} temperature files:")
    for f in temp_files:
        print(f"  - {os.path.basename(f)}")
    
    print("\nStarting conversion...")
    
    # Process all files
    all_dataframes = []
    
    for nc_file in temp_files:
        df = convert_temperature_file(nc_file)
        if not df.empty:
            all_dataframes.append(df)
    
    # Combine all data
    if all_dataframes:
        print(f"\nCombining data from {len(all_dataframes)} files...")
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Sort by time and coordinates
        final_df = final_df.sort_values(['time', 'latitude', 'longitude'])
        
        # Round coordinates to reasonable precision
        final_df['latitude'] = final_df['latitude'].round(3)
        final_df['longitude'] = final_df['longitude'].round(3)
        
        # Convert temperature from Kelvin to Celsius
        final_df['temperature_max_celsius'] = final_df['temperature_max'] - 273.15
        
        # Reorder columns
        final_df = final_df[['time', 'latitude', 'longitude', 'temperature_max', 'temperature_max_celsius']]
        
        # Save to CSV
        final_df.to_csv(OUTPUT_CSV, index=False)
        
        print(f"\nConversion completed successfully!")
        print(f"Total data points: {len(final_df):,}")
        print(f"Date range: {final_df['time'].min()} to {final_df['time'].max()}")
        print(f"Latitude range: {final_df['latitude'].min():.3f} to {final_df['latitude'].max():.3f}")
        print(f"Longitude range: {final_df['longitude'].min():.3f} to {final_df['longitude'].max():.3f}")
        print(f"Temperature range: {final_df['temperature_max_celsius'].min():.1f}°C to {final_df['temperature_max_celsius'].max():.1f}°C")
        print(f"CSV saved to: {OUTPUT_CSV}")
        
        # Show sample data
        print(f"\nSample data (first 5 rows):")
        print(final_df.head())
        
    else:
        print("No valid data found in any files!")

if __name__ == "__main__":
    main()