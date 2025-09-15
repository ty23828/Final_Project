#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_era5_land_yearly.py

Download ERA5-Land reanalysis (~10 km) data yearly:
 - 2m dewpoint temperature
 - 2m temperature
 - 10m u-component of wind
 - 10m v-component of wind
 - total precipitation

Dataset: reanalysis-era5-land
Product type: reanalysis
Time: 2015–2018, daily 00:00 (adjustable as needed)
Area: Portugal [42.2, -9.6, 36.8, -6.2]
Output: Save by year, netCDF format
"""
import os
import logging
import cdsapi

# Sub-region boundaries [North, West, South, East]
PORTUGAL_AREA = [42.2, -9.6, 36.8, -6.2]

# Years
YEARS = [str(y) for y in range(2017, 2018)]
# Months, days, times
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]
TIMES  = ["06:00"]

# Variables to download
VARIABLES = [
    "2m_dewpoint_temperature",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]

# Output directory - same folder as this script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ERA5_LAND_Yearly")

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

def download_yearly_data(client, year):
    """Download ERA5-Land data for specified year"""
    outpath = os.path.join(OUTPUT_DIR, f"era5_land_{year}.nc")
    if os.path.exists(outpath):
        logger.info(f"File for year {year} already exists, skipping: {outpath}")
        return

    request = {
        "product_type": "reanalysis",
        "format":       "netcdf",
        "area":         PORTUGAL_AREA,
        "year":         year,
        "month":        MONTHS,
        "day":          DAYS,
        "time":         TIMES,
        "variable":     VARIABLES,
    }

    try:
        logger.info(f"Starting download for year {year} ERA5-Land data")
        client.retrieve("reanalysis-era5-land", request, outpath)
        logger.info(f"Download completed: {outpath}")
    except Exception as e:
        logger.error(f"Download failed for year {year}: {e}")

def main():
    c = setup_client()
    for year in YEARS:
        download_yearly_data(c, year)

if __name__ == "__main__":
    main()
