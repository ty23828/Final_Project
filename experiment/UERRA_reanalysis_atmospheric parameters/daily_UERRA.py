#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_uerra_mescan_surfex_yearly.py

Download UERRA reanalysis (~5km) MESCAN-SURFEX analysis data yearly:
 - 10m wind speed (10m_wind_speed)
 - 2m relative humidity (2m_relative_humidity)
 - 2m temperature (2m_temperature)
 - total precipitation (total_precipitation)

Dataset: reanalysis-uerra-europe-single-levels
Time: 2000-2019, daily 06:00
Area: Portugal sub-region [42.2, -9.6, 36.8, -6.2]
Output: Save by year, netCDF format
"""
import os
import logging
import cdsapi

# Configuration constants
PORTUGAL_BBOX = [42.2, -9.6, 36.8, -6.2]  # [North, West, South, East]
YEARS = [str(year) for year in range(2017, 2018)]
MONTHS = [f"{month:02d}" for month in range(1, 13)]
DAYS = [f"{day:02d}" for day in range(1, 32)]
HOURS = ["06:00"]
VARIABLES = [
    "10m_wind_speed",
    "2m_relative_humidity",
    "2m_temperature",
    "total_precipitation",
]
# Output directory - same folder as this script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("UERRA_MESCAN_Yearly")

CDSAPI_RCFILE = os.path.join(
    os.environ.get("USERPROFILE", os.path.expanduser("~")), ".cdsapirc"
)


def setup_client():
    """Initialize CDS API client"""
    if not os.path.isfile(CDSAPI_RCFILE):
        raise FileNotFoundError(f"Missing configuration file: {CDSAPI_RCFILE}")
    url = key = None
    with open(CDSAPI_RCFILE, 'r') as f:
        for line in f:
            if line.strip().startswith('url:'):
                url = line.split(':', 1)[1].strip()
            elif line.strip().startswith('key:'):
                key = line.split(':', 1)[1].strip()
    if not url or not key:
        raise ValueError(f"Missing url or key in configuration file {CDSAPI_RCFILE}")
    logger.info(f"Using configuration file: {CDSAPI_RCFILE}")
    return cdsapi.Client(url=url, key=key)


def download_yearly_data(client, year):
    """Download UERRA MESCAN-SURFEX data for specified year"""
    output_file = os.path.join(OUTPUT_DIR, f"uerra_mescan_{year}.nc")
    if os.path.exists(output_file):
        logger.info(f"File already exists, skipping: {output_file}")
        return

    request = {
        "origin": "mescan_surfex",
        "variable": VARIABLES,
        "year": year,
        "month": MONTHS,
        "day": DAYS,
        "time": HOURS,
        "data_format": "netcdf",
    }

    try:
        logger.info(f"Starting download for year {year}")
        client.retrieve(
            "reanalysis-uerra-europe-single-levels", request, output_file
        )
        logger.info(f"Download completed: {output_file}")
    except Exception as e:
        logger.error(f"Download failed for year {year}: {e}")


def main():
    """Main function to download data for all years"""
    client = setup_client()
    for year in YEARS:
        download_yearly_data(client, year)


if __name__ == "__main__":
    main()


