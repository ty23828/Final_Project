import cdsapi
import time
from datetime import datetime

dataset = "derived-era5-single-levels-daily-statistics"

# Base request template
base_request = {
    "product_type": "reanalysis",
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature"
    ],
    "daily_statistic": "daily_mean",
    "time_zone": "utc+00:00",
    "frequency": "6_hourly",
    "area": [42.2, -9.6, 36.8, -6.2]
}

client = cdsapi.Client()

# Loop through years 2015 to 2018
for year in range(2015, 2019):
    # Loop through months 1 to 12
    for month in range(1, 13):
        print(f"Downloading data for {year}-{month:02d}...")
        
        # Create request for current month
        request = base_request.copy()
        request["year"] = str(year)
        request["month"] = f"{month:02d}"
        
        # Add all days for the month
        if month in [1, 3, 5, 7, 8, 10, 12]:
            days = 31
        elif month in [4, 6, 9, 11]:
            days = 30
        else:  # February
            # Check for leap year
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                days = 29
            else:
                days = 28
        
        request["day"] = [f"{day:02d}" for day in range(1, days + 1)]
        
        # Create filename
        filename = f"era5_daily_mean_{year}_{month:02d}.nc"
        
        try:
            # Download the data
            client.retrieve(dataset, request).download(filename)
            print(f"Successfully downloaded {filename}")
            
            # Pause between downloads (30 seconds)
            if not (year == 2018 and month == 12):  # Don't pause after last download
                print("Pausing for 30 seconds...")
                time.sleep(30)
                
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            continue

print("All downloads completed!")
