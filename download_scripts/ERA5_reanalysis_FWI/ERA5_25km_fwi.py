import cdsapi
import time
import os

# Load credentials from .env file
def load_cds_credentials():
    """Load CDS API credentials from .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    else:
        print(f"Warning: .env file not found at {env_path}")

# Load credentials
load_cds_credentials()

dataset = "cems-fire-historical-v1"

# Initialize client with credentials from environment
client = cdsapi.Client(
    url=os.environ.get('CDSAPI_URL', 'https://cds.climate.copernicus.eu/api'),
    key=os.environ.get('CDSAPI_KEY')
)

# Loop through years 2015 to 2018
for year in range(2013, 2014):
    print(f"Downloading FWI data for {year}...")
    
    request = {
        "product_type": "reanalysis",
        "variable": ["fire_weather_index"],
        "dataset_type": "consolidated_dataset",
        "system_version": ["4_1"],
        "year": [str(year)],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "grid": "0.25/0.25",
        "data_format": "grib"
    }
    
    # Create filename
    filename = f"era5_fwi_{year}.grib"
    
    try:
        # Download the data
        client.retrieve(dataset, request).download(filename)
        print(f"Successfully downloaded {filename}")
        
        # Pause between downloads (30 seconds)
        if year != 2018:  # Don't pause after last download
            print("Pausing for 30 seconds...")
            time.sleep(30)
            
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        continue

print("All FWI downloads completed!")
