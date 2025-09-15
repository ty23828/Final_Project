#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_cmip6_daily.py

Download CMIP6 daily climate model data for validation:
 - Daily maximum near-surface air temperature
 - Near-surface specific humidity
 - Near-surface wind speed
 - Precipitation

Dataset: projections-cmip6
Temporal Resolution: Daily
Experiment: Historical
Model: UKESM1-0-LL
Year: 2013
Area: Portugal [42.2, -9.6, 36.8, -6.2]
Output: Save by variable, netCDF format
"""
import os
import logging
import cdsapi

# Sub-region boundaries [North, West, South, East]
PORTUGAL_AREA = [42.2, -9.6, 36.8, -6.2]

# Time parameters
YEARS = ["2013"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]  # All months
DAYS = [f"{d:02d}" for d in range(1, 32)]    # All days

# CMIP6 model and experiment
MODEL = "ukesm1_0_ll"
EXPERIMENT = "historical"

# Variables to download - EACH SEPARATELY
VARIABLES = [
    "daily_maximum_near_surface_air_temperature",
    "near_surface_specific_humidity", 
    "near_surface_wind_speed",
    "precipitation"
]

# Variable name mapping for output files
VARIABLE_NAMES = {
    "daily_maximum_near_surface_air_temperature": "tasmax",
    "near_surface_specific_humidity": "huss",
    "near_surface_wind_speed": "sfcWind", 
    "precipitation": "pr"
}

# Output directory - same folder as this script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CMIP6_Daily")

# CDS API configuration file location
CDSAPI_RCFILE = os.path.join(
    os.environ.get("USERPROFILE", os.path.expanduser("~")), ".cdsapirc"
)

def setup_client():
    """Initialize CDS API client"""
    if not os.path.isfile(CDSAPI_RCFILE):
        raise FileNotFoundError(f"Missing .cdsapirc configuration file: {CDSAPI_RCFILE}")
    url = key = None
    with open(CDSAPI_RCFILE) as f:
        for line in f:
            if line.startswith("url:"):
                url = line.split(":",1)[1].strip()
            elif line.startswith("key:"):
                key = line.split(":",1)[1].strip()
    if not url or not key:
        raise ValueError(f"Missing url or key in {CDSAPI_RCFILE}")
    logger.info(f"Using CDS API configuration: {CDSAPI_RCFILE}")
    return cdsapi.Client(url=url, key=key)

def download_single_variable(client, variable, year):
    """Download a single variable for the entire year"""
    var_short_name = VARIABLE_NAMES.get(variable, variable)
    outpath = os.path.join(OUTPUT_DIR, f"cmip6_{var_short_name}_{MODEL}_{year}_portugal.nc")
    
    if os.path.exists(outpath):
        logger.info(f"File for {var_short_name} {year} already exists, skipping: {outpath}")
        return True

    request = {
        "temporal_resolution": "daily",
        "experiment": EXPERIMENT,
        "variable": [variable],  # Single variable in list
        "model": MODEL,
        "area": PORTUGAL_AREA,
        "year": year,
        "month": MONTHS,  # All 12 months
        "day": DAYS       # All days
    }

    try:
        logger.info(f"Starting download for {var_short_name} ({variable}) {year}")
        logger.info(f"Model: {MODEL}, Experiment: {EXPERIMENT}")
        logger.info(f"Area: Portugal {PORTUGAL_AREA}")
        logger.info(f"Time period: {year}, all 12 months")
        
        client.retrieve("projections-cmip6", request, outpath)
        logger.info(f"✓ Download completed: {outpath}")
        
        # Check file size
        if os.path.exists(outpath):
            file_size = os.path.getsize(outpath) / (1024 * 1024)  # MB
            logger.info(f"File size: {file_size:.2f} MB")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Download failed for {var_short_name} {year}: {e}")
        logger.error(f"Request details: {request}")
        
        # Try monthly download as fallback
        logger.info(f"Trying monthly download fallback for {var_short_name}...")
        return download_variable_monthly(client, variable, year)

def download_variable_monthly(client, variable, year):
    """Download a variable month by month as fallback"""
    var_short_name = VARIABLE_NAMES.get(variable, variable)
    logger.info(f"Attempting monthly downloads for {var_short_name} {year}")
    
    successful_months = 0
    
    for month in MONTHS:
        outpath = os.path.join(OUTPUT_DIR, f"cmip6_{var_short_name}_{MODEL}_{year}_{month}_portugal.nc")
        
        if os.path.exists(outpath):
            logger.info(f"File for {var_short_name} {year}-{month} already exists, skipping")
            successful_months += 1
            continue

        request = {
            "temporal_resolution": "daily",
            "experiment": EXPERIMENT,
            "variable": [variable],
            "model": MODEL,
            "area": PORTUGAL_AREA,
            "year": year,
            "month": [month],
            "day": DAYS
        }

        try:
            logger.info(f"Downloading {var_short_name} for {year}-{month}...")
            client.retrieve("projections-cmip6", request, outpath)
            logger.info(f"✓ Monthly download completed: {outpath}")
            successful_months += 1
            
        except Exception as e:
            logger.error(f"✗ Monthly download failed for {var_short_name} {year}-{month}: {e}")
    
    logger.info(f"Monthly fallback completed for {var_short_name}: {successful_months}/12 months successful")
    return successful_months > 0

def download_all_variables_separately(client, year):
    """Download all 4 variables separately for the specified year"""
    logger.info(f"Starting separate downloads for all 4 variables in {year}")
    logger.info("=" * 60)
    
    successful_variables = 0
    failed_variables = []
    
    for i, variable in enumerate(VARIABLES, 1):
        var_short_name = VARIABLE_NAMES.get(variable, variable)
        
        print(f"\n📥 DOWNLOADING VARIABLE {i}/4: {var_short_name}")
        print(f"   Full name: {variable}")
        print("-" * 60)
        
        success = download_single_variable(client, variable, year)
        
        if success:
            successful_variables += 1
            print(f"✓ {var_short_name} download completed successfully!")
        else:
            failed_variables.append(var_short_name)
            print(f"✗ {var_short_name} download failed!")
        
        print(f"Progress: {i}/{len(VARIABLES)} variables processed")
        print("-" * 60)
    
    # Summary
    print(f"\n📊 DOWNLOAD SUMMARY FOR {year}:")
    print(f"   Successful: {successful_variables}/{len(VARIABLES)} variables")
    print(f"   Failed: {len(failed_variables)} variables")
    
    if failed_variables:
        print(f"   Failed variables: {failed_variables}")
    
    return successful_variables, failed_variables

def main():
    """Main download function"""
    print("CMIP6 Daily Data Download - Separate Variables")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Experiment: {EXPERIMENT}")
    print(f"Year: {YEARS}")
    print(f"Area: Portugal {PORTUGAL_AREA}")
    print("=" * 60)
    print("VARIABLES TO DOWNLOAD:")
    for i, variable in enumerate(VARIABLES, 1):
        var_short_name = VARIABLE_NAMES.get(variable, variable)
        print(f"  {i}. {var_short_name} - {variable}")
    print("=" * 60)
    print("STRATEGY: Download each variable separately")
    print("FALLBACK: If annual download fails, try monthly downloads")
    print("=" * 60)
    
    c = setup_client()
    
    all_successful = 0
    all_failed = []
    
    # Download each year
    for year in YEARS:
        print(f"\n🗓️  PROCESSING YEAR: {year}")
        print("=" * 60)
        
        successful, failed = download_all_variables_separately(c, year)
        all_successful += successful
        all_failed.extend(failed)
        
        if successful == len(VARIABLES):
            print(f"✓ ALL VARIABLES DOWNLOADED SUCCESSFULLY FOR {year}!")
        else:
            print(f"⚠️  PARTIAL SUCCESS FOR {year}: {successful}/{len(VARIABLES)} variables")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 FINAL DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # List all downloaded files
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("cmip6_") and f.endswith(".nc")]
    
    if files:
        print(f"\n📁 Downloaded files ({len(files)}):")
        total_size = 0
        
        # Group files by variable
        by_variable = {}
        for file in sorted(files):
            for var_name in VARIABLE_NAMES.values():
                if var_name in file:
                    if var_name not in by_variable:
                        by_variable[var_name] = []
                    by_variable[var_name].append(file)
                    break
        
        for var_name in VARIABLE_NAMES.values():
            print(f"\n   {var_name.upper()}:")
            if var_name in by_variable:
                for file in by_variable[var_name]:
                    file_path = os.path.join(OUTPUT_DIR, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    total_size += size_mb
                    print(f"     - {file} ({size_mb:.2f} MB)")
            else:
                print(f"     ✗ No files found")
        
        print(f"\n📈 Total download size: {total_size:.2f} MB")
        
        # Check completeness
        expected_vars = set(VARIABLE_NAMES.values())
        found_vars = set(by_variable.keys())
        missing_vars = expected_vars - found_vars
        
        if not missing_vars:
            print("🎉 ALL 4 VARIABLES DOWNLOADED SUCCESSFULLY!")
        else:
            print(f"⚠️  MISSING VARIABLES: {list(missing_vars)}")
            print("   You may need to re-run the script or check for errors above.")
        
    else:
        print("\n❌ ERROR: No files found!")
        print("   Check the error messages above for troubleshooting.")
        exit(1)

if __name__ == "__main__":
    main()