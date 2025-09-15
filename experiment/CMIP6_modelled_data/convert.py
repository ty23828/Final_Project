"""
convert.py

Convert multiple CMIP6 NetCDF files to single CSV format
Input: Auto-detect CMIP6 NetCDF files in current directory (4 variables)
Output: Single merged CSV file with all variables
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import glob
import sys
import zipfile
import tempfile
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CMIP6_Converter")

def check_dependencies():
    """Check if required packages are installed"""
    missing_packages = []
    
    try:
        import xarray as xr
        logger.info("✓ xarray found")
    except ImportError:
        missing_packages.append("xarray")
    
    try:
        import netCDF4
        logger.info("✓ netCDF4 found")
    except ImportError:
        missing_packages.append("netCDF4")
    
    try:
        import h5netcdf
        logger.info("✓ h5netcdf found")
    except ImportError:
        logger.warning("h5netcdf not found (optional)")
    
    if missing_packages:
        print("ERROR: Missing required packages!")
        print("Please install the following packages:")
        for package in missing_packages:
            print(f"  pip install {package}")
        print("\nOr install all at once:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    return True

# Only import xarray after checking dependencies
if check_dependencies():
    import xarray as xr
else:
    sys.exit(1)

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Expected variables and their patterns
VARIABLE_PATTERNS = {
    'tasmax': ['tasmax', 'daily_maximum_near_surface_air_temperature'],
    'huss': ['huss', 'near_surface_specific_humidity'],
    'sfcWind': ['sfcWind', 'near_surface_wind_speed'], 
    'pr': ['pr', 'precipitation']
}

def is_zip_file(filename):
    """Check if file is a ZIP archive"""
    try:
        with zipfile.ZipFile(filename, 'r') as zip_file:
            return True
    except:
        return False

def extract_netcdf_from_zip(zip_filename):
    """Extract NetCDF files from ZIP archive"""
    temp_dir = tempfile.mkdtemp()
    extracted_files = []
    
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_file:
            logger.info(f"ZIP contents: {zip_file.namelist()}")
            
            for file_info in zip_file.infolist():
                if file_info.filename.endswith('.nc'):
                    extracted_path = zip_file.extract(file_info, temp_dir)
                    extracted_files.append(extracted_path)
                    logger.info(f"Extracted: {file_info.filename}")
            
        return extracted_files, temp_dir
        
    except Exception as e:
        logger.error(f"Error extracting ZIP file: {e}")
        shutil.rmtree(temp_dir)
        return [], None

def find_cmip6_files():
    """Find all CMIP6 NetCDF files in script directory"""
    search_dir = SCRIPT_DIR
    
    patterns = [
        os.path.join(search_dir, "cmip6*.nc"),
        os.path.join(search_dir, "CMIP6*.nc"), 
        os.path.join(search_dir, "*cmip6*.nc"),
        os.path.join(search_dir, "*CMIP6*.nc")
    ]
    
    found_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        found_files.extend(files)
    
    # Remove duplicates and sort
    found_files = sorted(list(set(found_files)))
    
    logger.info(f"Searching in directory: {search_dir}")
    logger.info(f"Found {len(found_files)} CMIP6 NetCDF files:")
    for i, file in enumerate(found_files):
        filename = os.path.basename(file)
        size_mb = os.path.getsize(file) / (1024*1024)
        is_zip = is_zip_file(file)
        file_type = "ZIP archive" if is_zip else "NetCDF file"
        
        # Identify variable from filename
        variable = identify_variable_from_filename(filename)
        logger.info(f"  {i+1}. {filename} ({size_mb:.2f} MB) - {file_type} - Variable: {variable}")
    
    return found_files

def identify_variable_from_filename(filename):
    """Identify which variable a file contains based on filename"""
    filename_lower = filename.lower()
    
    for var_name, patterns in VARIABLE_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in filename_lower:
                return var_name
    
    return "unknown"

def load_netcdf_file(filename):
    """Load the NetCDF file and return xarray dataset"""
    if not os.path.exists(filename):
        logger.error(f"Input file not found: {filename}")
        return None, None
    
    temp_dir = None
    
    # Check if file is a ZIP archive
    if is_zip_file(filename):
        logger.info(f"File is a ZIP archive, extracting...")
        extracted_files, temp_dir = extract_netcdf_from_zip(filename)
        
        if not extracted_files:
            logger.error("No NetCDF files found in ZIP archive")
            return None, None
        
        # Use the first NetCDF file found
        filename = extracted_files[0]
        logger.info(f"Using extracted file: {os.path.basename(filename)}")
    
    # Try different engines
    engines = ['netcdf4', 'h5netcdf', 'scipy']
    
    for engine in engines:
        try:
            logger.info(f"Trying to load with engine: {engine}")
            ds = xr.open_dataset(filename, engine=engine)
            logger.info(f"✓ Successfully loaded NetCDF file with {engine}: {os.path.basename(filename)}")
            logger.info(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
            return ds, temp_dir
        except Exception as e:
            logger.warning(f"Failed with {engine}: {e}")
            continue
    
    logger.error("Failed to load NetCDF file with any available engine")
    if temp_dir:
        shutil.rmtree(temp_dir)
    return None, None

def explore_dataset(ds, filename):
    """Explore and log dataset structure"""
    basename = os.path.basename(filename)
    variable_from_name = identify_variable_from_filename(basename)
    
    logger.info(f"Dataset structure for {basename}:")
    logger.info(f"Variables: {list(ds.data_vars.keys())}")
    logger.info(f"Coordinates: {list(ds.coords.keys())}")
    logger.info(f"Dimensions: {dict(ds.dims)}")
    
    print(f"\n📊 ANALYZING FILE: {basename}")
    print(f"   Expected variable: {variable_from_name}")
    
    # Show main data variables (exclude bounds)
    data_vars = [var for var in ds.data_vars.keys() if not var.endswith('_bnds')]
    print(f"   Data variables found: {data_vars}")
    
    # Show details of main variable
    for var_name in data_vars:
        var = ds[var_name]
        print(f"\n   📈 Variable: '{var_name}'")
        print(f"      Shape: {var.shape}")
        print(f"      Dimensions: {var.dims}")
        if hasattr(var, 'long_name'):
            print(f"      Long name: {var.long_name}")
        if hasattr(var, 'units'):
            print(f"      Units: {var.units}")

def merge_datasets_to_csv(datasets_info, output_filename):
    """Merge multiple datasets into a single CSV file"""
    try:
        print(f"\n🔄 MERGING {len(datasets_info)} DATASETS...")
        print("="*60)
        
        # Convert each dataset to DataFrame separately
        all_dataframes = []
        
        for i, file_info in enumerate(datasets_info):
            filename = file_info['filename']
            ds = file_info['dataset']
            expected_var = file_info['expected_variable']
            
            # Get the main data variable (exclude bounds)
            data_vars = [var for var in ds.data_vars.keys() if not var.endswith('_bnds')]
            
            if not data_vars:
                logger.warning(f"No data variables found in {filename}")
                continue
                
            main_var = data_vars[0]  # Take the first (and usually only) data variable
            
            print(f"Processing file {i+1}/{len(datasets_info)}: {filename}")
            print(f"   Expected: {expected_var} -> Found: {main_var}")
            print(f"   Shape: {ds[main_var].shape}")
            
            # Convert to DataFrame
            df = ds.to_dataframe().reset_index()
            
            # Rename the main variable to the expected name if different
            if main_var != expected_var and main_var in df.columns:
                df = df.rename(columns={main_var: expected_var})
                print(f"   Renamed {main_var} to {expected_var}")
            
            # Store the DataFrame with metadata
            all_dataframes.append({
                'df': df,
                'variable': expected_var,
                'filename': filename
            })
        
        if not all_dataframes:
            logger.error("No valid datasets found to merge")
            return False
        
        print(f"\nMerging strategy: Join on coordinate columns")
        
        # Use the first DataFrame as base
        merged_df = all_dataframes[0]['df'].copy()
        base_var = all_dataframes[0]['variable']
        
        print(f"Base DataFrame from {base_var}: {merged_df.shape}")
        print(f"Base columns: {list(merged_df.columns)}")
        
        # Identify coordinate columns (common across all datasets)
        coord_columns = ['time', 'lat', 'lon']
        available_coords = [col for col in coord_columns if col in merged_df.columns]
        
        print(f"Coordinate columns for merging: {available_coords}")
        
        # Merge additional variables from other DataFrames
        for i in range(1, len(all_dataframes)):
            df_info = all_dataframes[i]
            df = df_info['df']
            var_name = df_info['variable']
            
            print(f"\nMerging {var_name} data...")
            print(f"   DataFrame shape: {df.shape}")
            
            # Select only coordinate columns and the variable column
            merge_columns = available_coords + [var_name]
            df_to_merge = df[merge_columns].copy()
            
            print(f"   Columns to merge: {merge_columns}")
            
            # Merge on coordinate columns
            merged_df = merged_df.merge(
                df_to_merge, 
                on=available_coords, 
                how='outer',
                suffixes=('', f'_{var_name}')
            )
            
            print(f"   After merge shape: {merged_df.shape}")
        
        print(f"\nFinal merged DataFrame:")
        print(f"   Shape: {merged_df.shape}")
        print(f"   Columns: {list(merged_df.columns)}")
        
        # Create output file path
        output_file = os.path.join(SCRIPT_DIR, output_filename)
        
        # Save to CSV
        merged_df.to_csv(output_file, index=False)
        
        # File info
        file_size_mb = os.path.getsize(output_file) / (1024*1024)
        
        print(f"\n🎉 MERGED CSV CREATED SUCCESSFULLY!")
        print(f"   File: {output_filename}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print(f"   Rows: {merged_df.shape[0]:,}")
        print(f"   Columns: {merged_df.shape[1]}")
        
        # Show sample data
        data_cols = [col for col in merged_df.columns if col in VARIABLE_PATTERNS.keys()]
        print(f"\n📊 SAMPLE DATA (first 3 rows):")
        sample_cols = ['time', 'lat', 'lon'] + data_cols
        sample_cols = [col for col in sample_cols if col in merged_df.columns]
        sample_df = merged_df[sample_cols].head(3)
        print(sample_df.to_string(index=False))
        
        return True
        
    except Exception as e:
        logger.error(f"Error merging datasets: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_multiple_files(cmip6_files):
    """Process multiple NetCDF files and merge them"""
    print(f"\n🔍 PROCESSING {len(cmip6_files)} FILES FOR MERGING...")
    print("="*60)
    
    datasets_info = []
    temp_dirs = []
    
    try:
        # Load all datasets
        for file_path in cmip6_files:
            basename = os.path.basename(file_path)
            expected_var = identify_variable_from_filename(basename)
            
            print(f"\n📂 Loading: {basename}")
            print(f"   Expected variable: {expected_var}")
            
            # Load dataset
            ds, temp_dir = load_netcdf_file(file_path)
            if ds is None:
                logger.warning(f"Failed to load {basename}, skipping...")
                continue
            
            if temp_dir:
                temp_dirs.append(temp_dir)
            
            # Quick exploration
            explore_dataset(ds, file_path)
            
            datasets_info.append({
                'filename': basename,
                'dataset': ds,
                'expected_variable': expected_var,
                'file_path': file_path
            })
        
        if len(datasets_info) < 2:
            logger.error("Need at least 2 datasets to merge")
            return False
        
        # Check if we have the expected variables
        found_vars = [info['expected_variable'] for info in datasets_info]
        expected_vars = list(VARIABLE_PATTERNS.keys())
        
        print(f"\n📋 VARIABLE COVERAGE:")
        print(f"   Expected: {expected_vars}")
        print(f"   Found: {found_vars}")
        
        missing_vars = set(expected_vars) - set(found_vars)
        if missing_vars:
            print(f"   ⚠️  Missing: {list(missing_vars)}")
        
        # Generate output filename
        year = "2013"  # Default, could be extracted from filenames
        output_filename = f"cmip6_merged_all_variables_{year}_portugal.csv"
        
        # Merge datasets
        success = merge_datasets_to_csv(datasets_info, output_filename)
        
        return success
        
    finally:
        # Clean up all datasets and temp directories
        for info in datasets_info:
            info['dataset'].close()
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")

def main():
    """Main conversion function"""
    print("CMIP6 NetCDF to CSV Converter - Multi-File Merger")
    print("=" * 60)
    print(f"Script directory: {SCRIPT_DIR}")
    print("Merges multiple NetCDF files (4 variables) into single CSV")
    print("Expected variables: tasmax, huss, sfcWind, pr")
    print("=" * 60)
    
    # Find CMIP6 files
    cmip6_files = find_cmip6_files()
    
    if not cmip6_files:
        print("No CMIP6 NetCDF files found in script directory.")
        print("Looking for files with patterns: cmip6*.nc, CMIP6*.nc, *cmip6*.nc")
        print(f"\nAvailable .nc files in {SCRIPT_DIR}:")
        all_nc_files = glob.glob(os.path.join(SCRIPT_DIR, "*.nc"))
        if all_nc_files:
            for file in all_nc_files:
                basename = os.path.basename(file)
                size_mb = os.path.getsize(file) / (1024*1024)
                is_zip = is_zip_file(file)
                file_type = "ZIP" if is_zip else "NetCDF"
                print(f"  - {basename} ({size_mb:.2f} MB) - {file_type}")
        else:
            print("  No .nc files found")
        return
    
    if len(cmip6_files) == 1:
        print(f"\n⚠️  Only 1 file found. This script is designed to merge multiple files.")
        print("For single file conversion, use the regular convert script.")
        return
    
    # Process and merge multiple files
    print(f"\n🎯 STRATEGY: Merge {len(cmip6_files)} files into single CSV")
    
    success = process_multiple_files(cmip6_files)
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 MERGING COMPLETED SUCCESSFULLY!")
        
        # List output files
        output_files = [f for f in os.listdir(SCRIPT_DIR) if f.startswith('cmip6_merged_') and f.endswith('.csv')]
        if output_files:
            print(f"\nGenerated merged CSV files ({len(output_files)}):")
            total_size = 0
            for file in sorted(output_files):
                file_path = os.path.join(SCRIPT_DIR, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                total_size += size_mb
                print(f"  ✅ {file} ({size_mb:.2f} MB)")
            
            print(f"\nTotal output size: {total_size:.2f} MB")
            print("✓ All 4 variables merged into single CSV file!")
        
    else:
        print("❌ MERGING FAILED!")
        print("Check the error messages above for troubleshooting.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()