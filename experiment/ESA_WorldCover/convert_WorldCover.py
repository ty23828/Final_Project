#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESA WorldCover 10m Data Processing Script

Process ESA WorldCover 10m 2020 v100 data files:
- Read multiple .tif files
- Merge them into a single raster
- Extract data for Portugal region
- Convert to different formats (CSV, NetCDF)
- Generate statistics and visualizations
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from shapely.geometry import box
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "ESA_WorldCover_10m_2020_v100_60deg_macrotile_N30W060")
OUTPUT_DIR = SCRIPT_DIR

# Portugal boundaries (approximate)
PORTUGAL_BBOX = {
    'west': -9.6,
    'east': -6.2,
    'south': 36.8,
    'north': 42.2
}

# ESA WorldCover 10m land cover classes
LANDCOVER_CLASSES = {
    10: 'Tree cover',
    20: 'Shrubland',
    30: 'Grassland',
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare / sparse vegetation',
    70: 'Snow and ice',
    80: 'Permanent water bodies',
    90: 'Herbaceous wetland',
    95: 'Mangroves',
    100: 'Moss and lichen'
}

def list_tif_files(data_dir):
    """List all .tif files in the data directory"""
    pattern = os.path.join(data_dir, "*.tif")
    files = glob.glob(pattern)
    files.sort()
    print(f"Found {len(files)} .tif files:")
    for i, file in enumerate(files, 1):
        print(f"  {i:2d}. {os.path.basename(file)}")
    return files

def get_raster_info(file_path):
    """Get basic information about a raster file"""
    with rasterio.open(file_path) as src:
        return {
            'file': os.path.basename(file_path),
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': src.dtypes[0],
            'crs': src.crs,
            'bounds': src.bounds,
            'transform': src.transform
        }

def inspect_raster_files(files):
    """Inspect all raster files and show their properties"""
    print(f"\n{'='*60}")
    print("RASTER FILES INSPECTION")
    print(f"{'='*60}")
    
    for file in files:
        info = get_raster_info(file)
        print(f"\nFile: {info['file']}")
        print(f"  Dimensions: {info['width']} x {info['height']}")
        print(f"  Data type: {info['dtype']}")
        print(f"  CRS: {info['crs']}")
        print(f"  Bounds: {info['bounds']}")
        
        # Calculate file size
        file_size_mb = info['width'] * info['height'] * 4 / (1024 * 1024)  # Assuming 4 bytes per pixel
        print(f"  Estimated size: {file_size_mb:.1f} MB")
        
        # Show sample values from a small window
        try:
            with rasterio.open(file) as src:
                # Read a small window (1000x1000 pixels) to check unique values
                window = Window(0, 0, min(1000, src.width), min(1000, src.height))
                sample_data = src.read(1, window=window)
                unique_values = np.unique(sample_data[sample_data != src.nodata])
                print(f"  Sample unique values: {unique_values}")
                print(f"  Land cover classes in sample:")
                for val in unique_values:
                    if val in LANDCOVER_CLASSES:
                        print(f"    {val}: {LANDCOVER_CLASSES[val]}")
        except Exception as e:
            print(f"  Error reading sample data: {e}")

def create_portugal_geometry():
    """Create a polygon geometry for Portugal"""
    portugal_poly = box(
        PORTUGAL_BBOX['west'], 
        PORTUGAL_BBOX['south'],
        PORTUGAL_BBOX['east'], 
        PORTUGAL_BBOX['north']
    )
    return portugal_poly

def filter_files_by_region(files, target_bbox):
    """Filter files that intersect with the target region"""
    relevant_files = []
    
    for file in files:
        info = get_raster_info(file)
        bounds = info['bounds']
        
        # Check if file intersects with Portugal
        if (bounds.left <= target_bbox['east'] and 
            bounds.right >= target_bbox['west'] and
            bounds.bottom <= target_bbox['north'] and 
            bounds.top >= target_bbox['south']):
            relevant_files.append(file)
    
    print(f"\nFiles intersecting with Portugal region: {len(relevant_files)}")
    for file in relevant_files:
        print(f"  - {os.path.basename(file)}")
    
    return relevant_files

def merge_rasters(files, output_path):
    """Merge multiple raster files into a single file"""
    print(f"\nMerging {len(files)} raster files...")
    
    if len(files) == 1:
        print("Only one file found, copying instead of merging...")
        import shutil
        shutil.copy2(files[0], output_path)
        return output_path
    
    src_files_to_mosaic = []
    for file in files:
        src = rasterio.open(file)
        src_files_to_mosaic.append(src)
    
    # Merge files
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # Update metadata
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw"
    })
    
    # Write merged raster
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Close source files
    for src in src_files_to_mosaic:
        src.close()
    
    print(f"Merged raster saved to: {output_path}")
    return output_path

def clip_raster_to_portugal(input_path, output_path):
    """Clip raster to Portugal boundaries"""
    print(f"\nClipping raster to Portugal boundaries...")
    
    portugal_geom = create_portugal_geometry()
    
    with rasterio.open(input_path) as src:
        # Clip raster
        out_image, out_transform = mask(src, [portugal_geom], crop=True)
        out_meta = src.meta.copy()
        
        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw"
        })
        
        # Write clipped raster
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
    
    print(f"Clipped raster saved to: {output_path}")
    return output_path

def raster_to_csv_chunked(raster_path, csv_path, chunk_size=10000, sample_rate=0.1):
    """Convert large raster to CSV format using chunked processing"""
    print(f"\nConverting large raster to CSV using chunked processing...")
    print(f"Chunk size: {chunk_size} rows, Sample rate: {sample_rate*100}%")
    
    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        print(f"Raster dimensions: {width} x {height}")
        
        # Calculate total pixels and estimated output size
        total_pixels = height * width
        sampled_pixels = int(total_pixels * sample_rate)
        print(f"Total pixels: {total_pixels:,}")
        print(f"Estimated sampled pixels: {sampled_pixels:,}")
        
        # Initialize output file
        first_chunk = True
        total_rows_written = 0
        
        # Process raster in chunks
        for row_start in range(0, height, chunk_size):
            row_end = min(row_start + chunk_size, height)
            chunk_height = row_end - row_start
            
            print(f"Processing chunk: rows {row_start}-{row_end} ({chunk_height} rows)")
            
            # Create window for this chunk
            window = Window(0, row_start, width, chunk_height)
            
            # Read chunk data
            chunk_data = src.read(1, window=window)
            
            # Generate coordinates for this chunk
            rows_chunk = np.arange(row_start, row_end)
            cols_chunk = np.arange(width)
            cols_grid, rows_grid = np.meshgrid(cols_chunk, rows_chunk)
            
            # Transform to geographic coordinates in smaller batches
            batch_size = 1000000  # Process 1M pixels at a time
            chunk_dfs = []
            
            for i in range(0, chunk_data.size, batch_size):
                end_idx = min(i + batch_size, chunk_data.size)
                
                # Get indices for this batch
                batch_rows = rows_grid.flat[i:end_idx]
                batch_cols = cols_grid.flat[i:end_idx]
                batch_data = chunk_data.flat[i:end_idx]
                
                # Transform coordinates
                batch_xs, batch_ys = rasterio.transform.xy(
                    src.transform, batch_rows, batch_cols
                )
                
                # Round coordinates to 3 decimal places
                batch_xs_rounded = [round(x, 3) for x in batch_xs]
                batch_ys_rounded = [round(y, 3) for y in batch_ys]
                
                # Create DataFrame for this batch
                batch_df = pd.DataFrame({
                    'longitude': batch_xs_rounded,
                    'latitude': batch_ys_rounded,
                    'landcover': batch_data
                })
                
                # Remove nodata values
                if src.nodata is not None:
                    batch_df = batch_df[batch_df.landcover != src.nodata]
                
                if len(batch_df) > 0:
                    chunk_dfs.append(batch_df)
            
            # Combine all batches from this chunk
            if chunk_dfs:
                chunk_df = pd.concat(chunk_dfs, ignore_index=True)
                
                # Group by rounded coordinates and take the most common landcover class
                # This handles multiple pixels that round to the same coordinate
                if len(chunk_df) > 0:
                    # Group by latitude and longitude, take mode of landcover
                    grouped = chunk_df.groupby(['latitude', 'longitude'])['landcover'].agg(
                        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
                    ).reset_index()
                    
                    # Sample the data to reduce size
                    sample_size = max(1, int(len(grouped) * sample_rate))
                    if sample_size < len(grouped):
                        grouped = grouped.sample(n=sample_size, random_state=42)
                    
                    # Add landcover class names
                    grouped['landcover_class'] = grouped['landcover'].map(LANDCOVER_CLASSES)
                    
                    # Write to CSV
                    grouped.to_csv(
                        csv_path,
                        mode='w' if first_chunk else 'a',
                        header=first_chunk,
                        index=False
                    )
                    
                    total_rows_written += len(grouped)
                    first_chunk = False
                    
                    print(f"  Wrote {len(grouped)} rows to CSV (rounded coordinates to 3 decimal places)")
            
            # Clear memory
            del chunk_data
            if 'chunk_df' in locals():
                del chunk_df
            if 'grouped' in locals():
                del grouped
    
    print(f"CSV conversion completed!")
    print(f"Total rows written: {total_rows_written:,}")
    print(f"CSV saved to: {csv_path}")
    print(f"Coordinates rounded to 3 decimal places")
    
    return csv_path

def downsample_raster(input_path, output_path, scale_factor=10):
    """Downsample raster to reduce size"""
    print(f"\nDownsampling raster by factor of {scale_factor}...")
    
    with rasterio.open(input_path) as src:
        # Calculate new dimensions
        new_width = src.width // scale_factor
        new_height = src.height // scale_factor
        
        print(f"Original size: {src.width} x {src.height}")
        print(f"New size: {new_width} x {new_height}")
        
        # Calculate new transform
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        
        # Read and resample data
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.nearest
        )
        
        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": new_height,
            "width": new_width,
            "transform": new_transform,
            "compress": "lzw"
        })
        
        # Write downsampled raster
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(data)
    
    print(f"Downsampled raster saved to: {output_path}")
    return output_path

def generate_statistics(csv_path):
    """Generate statistics about the landcover data - Display only, no file output"""
    print(f"\nGenerating landcover statistics...")
    
    # Read CSV in chunks if it's large
    chunk_size = 100000
    stats_list = []
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk_stats = chunk.groupby(['landcover', 'landcover_class']).size().reset_index(name='pixel_count')
        stats_list.append(chunk_stats)
    
    # Combine all chunks
    all_stats = pd.concat(stats_list, ignore_index=True)
    stats = all_stats.groupby(['landcover', 'landcover_class'])['pixel_count'].sum().reset_index()
    
    # Calculate percentages
    total_pixels = stats['pixel_count'].sum()
    stats['percentage'] = (stats['pixel_count'] / total_pixels) * 100
    
    # Calculate area (assuming 10m resolution, but adjusted for sampling)
    stats['area_km2'] = stats['pixel_count'] * (10 * 10) / 1000000  # Convert to km²
    
    print("\nLandcover Statistics:")
    print(stats.to_string(index=False))
    
    # Don't save statistics file - only display
    print(f"Statistics displayed above (not saved to file)")
    
    return stats

def create_simple_visualization(csv_path):
    """Create simple visualizations - Display only, no file output"""
    print(f"\nCreating visualizations...")
    
    # Sample data for visualization if file is large
    sample_size = 50000
    df = pd.read_csv(csv_path, nrows=sample_size)
    
    print(f"Using {len(df)} samples for visualization")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ESA WorldCover 10m - Portugal Region Analysis', fontsize=14)
    
    # 1. Sample spatial distribution
    ax1 = axes[0, 0]
    ax1.scatter(df['longitude'], df['latitude'], c=df['landcover'], 
               cmap='tab10', s=1, alpha=0.6)
    ax1.set_title('Sample Spatial Distribution')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # 2. Landcover class distribution
    ax2 = axes[0, 1]
    class_counts = df['landcover_class'].value_counts()
    class_counts.plot(kind='bar', ax=ax2)
    ax2.set_title('Land Cover Distribution')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Longitude distribution
    ax3 = axes[1, 0]
    ax3.hist(df['longitude'], bins=30, alpha=0.7)
    ax3.set_title('Longitude Distribution')
    ax3.set_xlabel('Longitude')
    
    # 4. Latitude distribution
    ax4 = axes[1, 1]
    ax4.hist(df['latitude'], bins=30, alpha=0.7)
    ax4.set_title('Latitude Distribution')
    ax4.set_xlabel('Latitude')
    
    plt.tight_layout()
    
    # Display plot instead of saving
    plt.show()
    plt.close()
    
    print(f"Visualization displayed (not saved to file)")

def raster_to_csv_ultra_efficient(raster_path, csv_path, target_resolution=0.001, max_samples=10000):
    """Ultra-efficient raster to CSV conversion using strategic sampling"""
    print(f"\nConverting raster to CSV using ultra-efficient method...")
    print(f"Target resolution: {target_resolution}°")
    print(f"Maximum samples: {max_samples:,}")
    
    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        print(f"Original raster dimensions: {width} x {height}")
        print(f"Original total pixels: {height * width:,}")
        
        # Get bounds
        bounds = src.bounds
        lon_min, lat_min = bounds.left, bounds.bottom
        lon_max, lat_max = bounds.right, bounds.top
        
        print(f"Bounds: lon[{lon_min:.3f}, {lon_max:.3f}], lat[{lat_min:.3f}, {lat_max:.3f}]")
        
        # Calculate regular grid for sampling
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        
        # Calculate grid size based on target resolution
        n_lon = int(lon_range / target_resolution) + 1
        n_lat = int(lat_range / target_resolution) + 1
        
        print(f"Target grid: {n_lon} x {n_lat} = {n_lon * n_lat:,} points")
        
        # If still too many points, reduce further
        if n_lon * n_lat > max_samples:
            # Calculate reduction factor
            reduction_factor = (n_lon * n_lat / max_samples) ** 0.5
            n_lon = max(10, int(n_lon / reduction_factor))
            n_lat = max(10, int(n_lat / reduction_factor))
            print(f"Reduced grid: {n_lon} x {n_lat} = {n_lon * n_lat:,} points")
        
        # Generate sample coordinates
        lons = np.linspace(lon_min, lon_max, n_lon)
        lats = np.linspace(lat_min, lat_max, n_lat)
        
        # Round coordinates to 3 decimal places
        lons_rounded = np.round(lons, 3)
        lats_rounded = np.round(lats, 3)
        
        print(f"Sampling {len(lons_rounded)} x {len(lats_rounded)} = {len(lons_rounded) * len(lats_rounded)} points...")
        
        # Create results lists
        result_lons = []
        result_lats = []
        result_values = []
        
        # Sample point by point to avoid memory issues
        for i, lat in enumerate(lats_rounded):
            if i % 10 == 0:
                print(f"  Processing latitude row {i+1}/{len(lats_rounded)}")
            
            for lon in lons_rounded:
                try:
                    # Convert geographic coordinates to pixel coordinates
                    row, col = rasterio.transform.rowcol(src.transform, lat, lon)
                    
                    # Check if within bounds
                    if 0 <= row < height and 0 <= col < width:
                        # Read single pixel value
                        value = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                        
                        # Check if valid value
                        if src.nodata is None or value != src.nodata:
                            result_lons.append(lon)
                            result_lats.append(lat)
                            result_values.append(value)
                            
                except Exception as e:
                    # Skip problematic points
                    continue
        
        print(f"Successfully sampled {len(result_values)} points")
        
        # Create DataFrame
        df = pd.DataFrame({
            'longitude': result_lons,
            'latitude': result_lats,
            'landcover': result_values
        })
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['longitude', 'latitude'])
        
        # Add landcover class names
        df['landcover_class'] = df['landcover'].map(LANDCOVER_CLASSES)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        
        print(f"CSV conversion completed!")
        print(f"Final data points: {len(df):,}")
        print(f"Coordinates rounded to 3 decimal places")
        print(f"CSV saved to: {csv_path}")
        
        return csv_path

def quick_raster_sample(raster_path, csv_path, sample_every_n=1000):
    """Quick sampling method - take every nth pixel"""
    print(f"\nQuick raster sampling (every {sample_every_n}th pixel)...")
    
    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        print(f"Raster dimensions: {width} x {height}")
        
        # Calculate sample positions
        sample_rows = range(0, height, sample_every_n)
        sample_cols = range(0, width, sample_every_n)
        
        total_samples = len(sample_rows) * len(sample_cols)
        print(f"Will sample {total_samples:,} pixels")
        
        # Read sampled data
        result_data = []
        
        for i, row in enumerate(sample_rows):
            if i % 10 == 0:
                print(f"  Processing row {i+1}/{len(sample_rows)}")
            
            for col in sample_cols:
                try:
                    # Read single pixel
                    value = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                    
                    if src.nodata is None or value != src.nodata:
                        # Convert pixel coordinates to geographic
                        lon, lat = rasterio.transform.xy(src.transform, row, col)
                        
                        # Round coordinates to 3 decimal places
                        lon_rounded = round(lon, 3)
                        lat_rounded = round(lat, 3)
                        
                        result_data.append({
                            'longitude': lon_rounded,
                            'latitude': lat_rounded,
                            'landcover': value
                        })
                        
                except Exception as e:
                    continue
        
        print(f"Successfully sampled {len(result_data)} valid pixels")
        
        # Create DataFrame
        df = pd.DataFrame(result_data)
        
        # Remove duplicates (from rounding)
        df = df.drop_duplicates(subset=['longitude', 'latitude'])
        
        # Add landcover class names
        df['landcover_class'] = df['landcover'].map(LANDCOVER_CLASSES)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        
        print(f"CSV saved to: {csv_path}")
        print(f"Final unique coordinates: {len(df):,}")
        print(f"Coordinates rounded to 3 decimal places")
        
        return csv_path

def raster_to_csv_robust(raster_path, csv_path, sample_every_n=1000):
    """Robust raster to CSV conversion with better error handling"""
    print(f"\nConverting raster to CSV using robust method...")
    print(f"Sample every {sample_every_n}th pixel")
    
    try:
        with rasterio.open(raster_path) as src:
            height, width = src.height, src.width
            print(f"Raster dimensions: {width} x {height}")
            print(f"Transform: {src.transform}")
            print(f"CRS: {src.crs}")
            print(f"Bounds: {src.bounds}")
            print(f"NoData value: {src.nodata}")
            
            # Calculate sample positions
            sample_rows = list(range(0, height, sample_every_n))
            sample_cols = list(range(0, width, sample_every_n))
            
            total_samples = len(sample_rows) * len(sample_cols)
            print(f"Will sample {total_samples:,} pixels")
            
            # Read sampled data
            result_data = []
            valid_count = 0
            
            for i, row in enumerate(sample_rows):
                if i % 10 == 0:
                    print(f"  Processing row {i+1}/{len(sample_rows)}, found {valid_count} valid pixels so far")
                
                for col in sample_cols:
                    try:
                        # Check bounds
                        if row >= height or col >= width:
                            continue
                        
                        # Read single pixel
                        value = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                        
                        # Check if valid value
                        if src.nodata is None or value != src.nodata:
                            # Convert pixel coordinates to geographic
                            lon, lat = rasterio.transform.xy(src.transform, row, col)
                            
                            # Round coordinates to 3 decimal places
                            lon_rounded = round(float(lon), 3)
                            lat_rounded = round(float(lat), 3)
                            
                            result_data.append({
                                'longitude': lon_rounded,
                                'latitude': lat_rounded,
                                'landcover': int(value)
                            })
                            
                            valid_count += 1
                            
                    except Exception as e:
                        print(f"    Error processing pixel ({row}, {col}): {e}")
                        continue
            
            print(f"Successfully sampled {len(result_data)} valid pixels")
            
            if len(result_data) == 0:
                print("No valid data found! Trying with smaller sample interval...")
                # Try with smaller sample interval
                return raster_to_csv_robust(raster_path, csv_path, sample_every_n // 2)
            
            # Create DataFrame
            df = pd.DataFrame(result_data)
            
            # Remove duplicates (from rounding)
            df = df.drop_duplicates(subset=['longitude', 'latitude'])
            
            # Add landcover class names
            df['landcover_class'] = df['landcover'].map(LANDCOVER_CLASSES)
            
            # Fill missing class names
            df['landcover_class'] = df['landcover_class'].fillna(f'Unknown_{df["landcover"]}')
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            print(f"CSV saved to: {csv_path}")
            print(f"Final unique coordinates: {len(df):,}")
            print(f"Coordinates rounded to 3 decimal places")
            
            # Show sample of data
            print("\nSample of data:")
            print(df.head())
            
            return csv_path
            
    except Exception as e:
        print(f"Error in raster_to_csv_robust: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main processing function with improved error handling"""
    print("ESA WorldCover 10m Data Processing (3 Decimal Places Precision)")
    print("=" * 60)
    
    # List all .tif files
    files = list_tif_files(DATA_DIR)
    
    if not files:
        print("No .tif files found in the data directory!")
        return
    
    # Inspect first file only
    print(f"\nInspecting first file: {os.path.basename(files[0])}")
    inspect_raster_files(files[:1])
    
    # Filter files relevant to Portugal
    portugal_files = filter_files_by_region(files, PORTUGAL_BBOX)
    
    if not portugal_files:
        print("No files intersect with Portugal region!")
        return
    
    # Define output paths
    merged_path = os.path.join(OUTPUT_DIR, "esa_worldcover_merged.tif")
    clipped_path = os.path.join(OUTPUT_DIR, "esa_worldcover_portugal.tif")
    csv_path = os.path.join(OUTPUT_DIR, "esa_worldcover_portugal_3decimal.csv")
    
    # Process the data
    try:
        # Merge relevant files
        merge_rasters(portugal_files, merged_path)
        
        # Clip to Portugal boundaries
        clip_raster_to_portugal(merged_path, clipped_path)
        
        # Check file size
        file_size = os.path.getsize(clipped_path) / (1024 * 1024)  # MB
        print(f"\nClipped raster size: {file_size:.1f} MB")
        
        # Use robust conversion method with 3 decimal places
        print("Using robust raster to CSV conversion (3 decimal places)...")
        result = raster_to_csv_robust(clipped_path, csv_path, sample_every_n=1000)
        
        if result is None:
            print("CSV conversion failed!")
            return
        
        # Generate statistics - only display, no file output
        generate_statistics(csv_path)
        
        # Skip visualizations - not needed
        # create_simple_visualization(csv_path)
        
        print(f"\n{'='*60}")
        print("Processing completed successfully!")
        print(f"Main output file:")
        
        # Show only the main CSV file
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
            print(f"- {os.path.basename(csv_path)} ({file_size:.1f} MB)")
            print(f"- Full path: {csv_path}")
            print(f"- Coordinates precision: 3 decimal places")
        else:
            print("- Main CSV file not found!")
        
        # Clean up intermediate files (optional)
        for temp_file in [merged_path, clipped_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"- Cleaned up temporary file: {os.path.basename(temp_file)}")
                except:
                    pass
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()