#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nc_to_csv_era5_land.py

Convert ERA5-Land reanalysis (~10 km) netCDF (or ZIP) files
to a single CSV table by day, keeping only 2015-2018 data
for Portugal sub-region (36°–43°N, –10°–-6°E).

Automatically detects ZIP packages and extracts the .nc file inside.

Output CSV columns:
  time,
  latitude, longitude,
  d2m,  # 2m dewpoint temperature (K)
  t2m,  # 2m temperature (K)
  u10,  # 10m u-component of wind (m/s)
  v10,  # 10m v-component of wind (m/s)
  tp    # total precipitation (m)
"""
import os
import glob
import zipfile
import tempfile
import xarray as xr
import pandas as pd
import numpy as np

# Input/output paths - current folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NC_DIR     = SCRIPT_DIR
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "era5_land_2017_PT_3decimal.csv")

# Only process these years
YEARS = {"2017"}

# Variables to export (short names)
VARS = ["d2m", "t2m", "u10", "v10", "tp"]

# Portugal sub-region boundaries
LAT_MIN, LAT_MAX = 36.8, 42.2
LON_MIN, LON_MAX = -9.6, -6.2

def list_nc_files(dirpath):
    """List all .nc files in directory and filter by year"""
    files = sorted(glob.glob(os.path.join(dirpath, "*.nc")))
    return [f for f in files if os.path.basename(f)[-7:-3] in YEARS]

def open_possibly_zipped(nc_path):
    """
    If file starts with ZIP magic number, extract first .nc file to temp directory;
    otherwise return original path.
    """
    with open(nc_path, "rb") as f:
        magic = f.read(4)
    if magic == b"PK\x03\x04":
        with zipfile.ZipFile(nc_path, "r") as z:
            members = [m for m in z.namelist() if m.lower().endswith(".nc")]
            if not members:
                raise RuntimeError(f"{os.path.basename(nc_path)} is ZIP but no .nc found inside")
            tmpdir = tempfile.mkdtemp(prefix="era5zip_")
            extracted = z.extract(members[0], tmpdir)
            return extracted
    return nc_path

def process_file(nc_file, out_csv, first_write):
    """Process single annual file: extract (if needed), slice by day, flatten, filter sub-region, write to CSV"""
    # Extract ZIP or open directly
    try:
        real_nc = open_possibly_zipped(nc_file)
    except Exception as e:
        print(f"⚠ Cannot extract {os.path.basename(nc_file)}: {e}, skipping")
        return first_write

    try:
        with xr.open_dataset(real_nc, engine="netcdf4", autoclose=True) as ds:
            print(f"Processing {os.path.basename(nc_file)}  dims={list(ds.dims)}  vars={list(ds.data_vars)}")

            # Check if all variables are present
            missing = [v for v in VARS if v not in ds.data_vars]
            if missing:
                print(f"  ⚠ Missing variables {missing}, skipping")
                return first_write

            # Convert longitude from [0,360) to [-180,180)
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))

            # Auto-detect time coordinate (usually 'time' or 'valid_time')
            time_dim = next((c for c, x in ds.coords.items()
                             if np.issubdtype(x.dtype, np.datetime64)), None)
            if time_dim is None:
                print("  ⚠ No time coordinate found, skipping")
                return first_write

            # Slice by day
            for t in ds[time_dim].values:
                slice_t = ds.sel({time_dim: t})
                # 1D latitude/longitude
                lat1d = slice_t.latitude.values
                lon1d = slice_t.longitude.values
                
                # Round coordinates to 3 decimal places
                lat1d_rounded = np.round(lat1d, 3)
                lon1d_rounded = np.round(lon1d, 3)
                
                # Generate 2D grid and flatten
                lon2d, lat2d = np.meshgrid(lon1d_rounded, lat1d_rounded)
                flat = {
                    "time":     [t] * lon2d.size,
                    "latitude": lat2d.flatten(),
                    "longitude":lon2d.flatten(),
                }
                for v in VARS:
                    flat[v] = slice_t[v].values.flatten()
                df = pd.DataFrame(flat)

                # Filter Portugal sub-region
                df = df[
                    (df.latitude  >= LAT_MIN) & (df.latitude  <= LAT_MAX) &
                    (df.longitude >= LON_MIN)   & (df.longitude <= LON_MAX)
                ]
                if df.empty:
                    continue

                # Additional rounding to ensure 3 decimal places in output
                df['latitude'] = df['latitude'].round(3)
                df['longitude'] = df['longitude'].round(3)

                # Remove duplicate coordinates (from rounding)
                df = df.drop_duplicates(subset=['time', 'latitude', 'longitude'])

                if not df.empty:
                    df.to_csv(
                        out_csv,
                        mode='w' if first_write else 'a',
                        header=first_write,
                        index=False
                    )
                    first_write = False
                    print(f"  ✓ Wrote {len(df)} rows for {t} (coordinates rounded to 3 decimal places)")

    except Exception as e:
        print(f"⚠ Failed to open or process {os.path.basename(nc_file)}: {e}")
    finally:
        # Clean up temporary extracted files
        if 'real_nc' in locals() and real_nc != nc_file and os.path.exists(real_nc):
            os.remove(real_nc)
            os.rmdir(os.path.dirname(real_nc))

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
    files = list_nc_files(NC_DIR)
    print(f"Found {len(files)} files for years {sorted(YEARS)}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Coordinate precision: 3 decimal places")
    
    first_write = True
    for fn in files:
        first_write = process_file(fn, OUTPUT_CSV, first_write)
    
    print(f"✅ Finished. Output CSV: {OUTPUT_CSV}")
    
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
