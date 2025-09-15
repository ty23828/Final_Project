#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_daily_mean_temp.py

Convert ERA5 daily mean temperature NetCDF files to CSV format
for Portugal sub-region (36.8°–42.2°N, -9.6°–-6.2°E).

Input: era5_daily_mean_temp_YYYY_MM.nc files
Output: era5_daily_mean_temp_2017_portugal.csv (combined CSV)
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
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "era5_daily_mean_temp_2017_portugal.csv")

# Portugal sub-region boundaries
LAT_MIN, LAT_MAX = 36.8, 42.2
LON_MIN, LON_MAX = -9.6, -6.2

# Years to process
YEARS = {"2017"}

def list_temp_files(directory):
    """List all mean temperature NetCDF files in directory"""
    pattern = os.path.join(directory, "era5_daily_mean_temp_*.nc")
    files = glob.glob(pattern)
    # Filter by year
    filtered_files = []
    for f in files:
        filename = os.path.basename(f)
        if any(year in filename for year in YEARS):
            filtered_files.append(f)
    return sorted(filtered_files)

def convert_temperature_file(nc_file):
    """Convert single mean temperature NetCDF file to DataFrame"""
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
            
            # Find mean temperature variable
            temp_var = None
            possible_temp_vars = ['t2m', '2m_temperature', 'temperature', 'temp', 'T2M', 'mean_temp', 'temperature_mean']
            
            for var_name in ds.data_vars:
                if any(temp_key in var_name.lower() for temp_key in ['t2m', 'temperature', '2m', 'mean']):
                    temp_var = var_name
                    break
            
            if temp_var is None:
                # Try exact matches
                for var_name in possible_temp_vars:
                    if var_name in ds.data_vars:
                        temp_var = var_name
                        break
            
            if temp_var is None:
                print(f"  Warning: No mean temperature variable found. Available variables: {list(ds.data_vars.keys())}")
                return pd.DataFrame()
            
            print(f"  Mean temperature variable: {temp_var}")
            
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
                
                # Get mean temperature data
                temp_data = ds_time[temp_var].values.flatten()
                
                # Create DataFrame for this time step
                df_time = pd.DataFrame({
                    'time': [pd.to_datetime(time_val)] * len(lat_flat),
                    'latitude': lat_flat,
                    'longitude': lon_flat,
                    'temperature_mean': temp_data
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
    print("ERA5 Daily Mean Temperature NetCDF to CSV Converter")
    print("=" * 60)
    print(f"Processing directory: {INPUT_DIR}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Portugal bounds: Lat[{LAT_MIN}, {LAT_MAX}], Lon[{LON_MIN}, {LON_MAX}]")
    print(f"Years: {sorted(YEARS)}")
    print("-" * 60)
    
    # Find all mean temperature files
    temp_files = list_temp_files(INPUT_DIR)
    
    if not temp_files:
        print("No mean temperature NetCDF files found!")
        print("Expected file pattern: era5_daily_mean_temp_*.nc")
        print("Available files:")
        all_nc_files = glob.glob(os.path.join(INPUT_DIR, "*.nc"))
        for f in all_nc_files:
            print(f"  - {os.path.basename(f)}")
        return
    
    print(f"Found {len(temp_files)} mean temperature files:")
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
        final_df['temperature_mean_celsius'] = final_df['temperature_mean'] - 273.15
        
        # Calculate additional statistics
        print(f"\nCalculating temperature statistics...")
        
        # Group by location and calculate daily statistics
        location_stats = final_df.groupby(['latitude', 'longitude']).agg({
            'temperature_mean_celsius': ['mean', 'std', 'min', 'max', 'count']
        }).round(2)
        
        location_stats.columns = ['temp_mean', 'temp_std', 'temp_min', 'temp_max', 'data_count']
        location_stats = location_stats.reset_index()
        
        print(f"  Locations with data: {len(location_stats)}")
        print(f"  Average temperature across all locations: {location_stats['temp_mean'].mean():.1f}°C")
        print(f"  Temperature range: {location_stats['temp_min'].min():.1f}°C to {location_stats['temp_max'].max():.1f}°C")
        
        # Reorder columns
        final_df = final_df[['time', 'latitude', 'longitude', 'temperature_mean', 'temperature_mean_celsius']]
        
        # Save to CSV
        final_df.to_csv(OUTPUT_CSV, index=False)
        
        # Save location statistics
        stats_csv = OUTPUT_CSV.replace('.csv', '_location_stats.csv')
        location_stats.to_csv(stats_csv, index=False)
        
        print(f"\nConversion completed successfully!")
        print(f"=" * 60)
        print(f"📊 CONVERSION SUMMARY:")
        print(f"   Total data points: {len(final_df):,}")
        print(f"   Unique locations: {len(location_stats):,}")
        print(f"   Date range: {final_df['time'].min()} to {final_df['time'].max()}")
        print(f"   Days covered: {final_df['time'].nunique()}")
        print(f"   Latitude range: {final_df['latitude'].min():.3f}° to {final_df['latitude'].max():.3f}°")
        print(f"   Longitude range: {final_df['longitude'].min():.3f}° to {final_df['longitude'].max():.3f}°")
        print(f"   Temperature range: {final_df['temperature_mean_celsius'].min():.1f}°C to {final_df['temperature_mean_celsius'].max():.1f}°C")
        print(f"   Average daily mean temp: {final_df['temperature_mean_celsius'].mean():.1f}°C")
        print(f"   Temperature std dev: {final_df['temperature_mean_celsius'].std():.1f}°C")
        print(f"\n📁 OUTPUT FILES:")
        print(f"   Main data: {OUTPUT_CSV}")
        print(f"   Location stats: {stats_csv}")
        
        # Show sample data
        print(f"\n📋 SAMPLE DATA (first 10 rows):")
        print(final_df.head(10).to_string(index=False))
        
        # Show temporal coverage
        daily_counts = final_df.groupby(final_df['time'].dt.date).size()
        print(f"\n📅 TEMPORAL COVERAGE:")
        print(f"   Data points per day: {daily_counts.min()} to {daily_counts.max()}")
        print(f"   Average points per day: {daily_counts.mean():.1f}")
        print(f"   Total days: {len(daily_counts)}")
        
        # Show monthly breakdown
        monthly_counts = final_df.groupby([final_df['time'].dt.year, final_df['time'].dt.month]).size()
        print(f"\n📊 MONTHLY BREAKDOWN:")
        for (year, month), count in monthly_counts.items():
            print(f"   {year}-{month:02d}: {count:,} data points")
        
        print(f"\n✅ Mean temperature conversion completed successfully!")
        
    else:
        print("❌ No valid data found in any files!")
        print("Check that:")
        print("  1. NetCDF files contain mean temperature data")
        print("  2. Files have correct variable names (t2m, temperature, etc.)")
        print("  3. Files cover the Portugal region")
        print("  4. Files are not corrupted")

if __name__ == "__main__":
    main()