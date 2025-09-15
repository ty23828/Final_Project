"""
predict_fwi.py

Calculate Fire Weather Index (FWI) from CMIP6 climate data
Input: Merged CSV file with tasmax, huss, sfcWind, pr
Output: FWI predictions at 25km resolution
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FWI_Predictor")

class FWICalculator:
    """Calculate Fire Weather Index components"""
    
    def __init__(self):
        # FWI system constants
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.d = 0.0
        self.e = 0.0
        self.f = 0.0
        
        # Month length adjustment factors
        self.month_factors = {
            1: 6.5, 2: 7.5, 3: 9.0, 4: 12.8, 5: 13.9, 6: 13.9,
            7: 12.4, 8: 10.9, 9: 9.4, 10: 8.0, 11: 7.0, 12: 6.0
        }
    
    def calculate_ffmc(self, temp, humidity, wind, rain, ffmc_prev=85.0):
        """Calculate Fine Fuel Moisture Code (FFMC)"""
        # Convert temperature from Kelvin to Celsius
        temp_c = temp - 273.15
        
        # Convert specific humidity to relative humidity (approximate)
        # This is a simplified conversion
        rh = humidity * 100.0
        rh = np.clip(rh, 0, 100)
        
        # Convert wind speed from m/s to km/h
        wind_kmh = wind * 3.6
        
        # Convert precipitation from kg/m²/s to mm/day
        rain_mm = rain * 86400  # seconds in a day
        
        # FFMC calculation
        mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)
        
        if rain_mm > 0.5:
            rf = rain_mm - 0.5
            if mo <= 150.0:
                mo = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
            else:
                mo = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf)) + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf)
            
            if mo > 250.0:
                mo = 250.0
        
        ed = 0.942 * rh ** 0.679 + 11.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp_c) * (1.0 - np.exp(-0.115 * rh))
        
        if mo > ed:
            ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (1.0 - (rh / 100.0) ** 8)
            kd = ko * 0.581 * np.exp(0.0365 * temp_c)
            mo = ed + (mo - ed) * np.exp(-kd)
        
        ew = 0.618 * rh ** 0.753 + 10.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp_c) * (1.0 - np.exp(-0.115 * rh))
        
        if mo < ew:
            k1 = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
            kw = k1 * 0.581 * np.exp(0.0365 * temp_c)
            mo = ew - (ew - mo) * np.exp(-kw)
        
        ffmc = 59.5 * (250.0 - mo) / (147.2 + mo)
        
        return np.clip(ffmc, 0, 101)
    
    def calculate_dmc(self, temp, humidity, rain, dmc_prev=6.0, month=6):
        """Calculate Duff Moisture Code (DMC)"""
        # Convert temperature from Kelvin to Celsius
        temp_c = temp - 273.15
        
        # Convert specific humidity to relative humidity (approximate)
        rh = humidity * 100.0
        rh = np.clip(rh, 0, 100)
        
        # Convert precipitation from kg/m²/s to mm/day
        rain_mm = rain * 86400
        
        if rain_mm > 1.5:
            re = 0.92 * rain_mm - 1.27
            mo = 20.0 + np.exp(5.6348 - dmc_prev / 43.43)
            
            if dmc_prev <= 33.0:
                b = 100.0 / (0.5 + 0.3 * dmc_prev)
            elif dmc_prev <= 65.0:
                b = 14.0 - 1.3 * np.log(dmc_prev)
            else:
                b = 6.2 * np.log(dmc_prev) - 17.2
            
            mo = mo + 1000.0 * re / (48.77 + b * re)
            dmc_prev = 43.43 * (5.6348 - np.log(mo - 20.0))
        
        if temp_c > -1.1:
            k = 1.894 * (temp_c + 1.1) * (100.0 - rh) * self.month_factors[month] * 0.000001
        else:
            k = 0.0
        
        dmc = dmc_prev + k
        
        return np.maximum(dmc, 0)
    
    def calculate_dc(self, temp, rain, dc_prev=15.0, month=6):
        """Calculate Drought Code (DC)"""
        # Convert temperature from Kelvin to Celsius
        temp_c = temp - 273.15
        
        # Convert precipitation from kg/m²/s to mm/day
        rain_mm = rain * 86400
        
        if rain_mm > 2.8:
            rd = 0.83 * rain_mm - 1.27
            qo = 800.0 * np.exp(-dc_prev / 400.0)
            qr = qo + 3.937 * rd
            dr = 400.0 * np.log(800.0 / qr)
            
            if dr > 0.0:
                dc_prev = dr
            else:
                dc_prev = 0.0
        
        if temp_c > -2.8:
            v = 0.36 * (temp_c + 2.8) + self.month_factors[month]
        else:
            v = self.month_factors[month]
        
        if v < 0.0:
            v = 0.0
        
        dc = dc_prev + v
        
        return np.maximum(dc, 0)
    
    def calculate_isi(self, wind, ffmc):
        """Calculate Initial Spread Index (ISI)"""
        # Convert wind speed from m/s to km/h
        wind_kmh = wind * 3.6
        
        mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        ff = 19.115 * np.exp(mo * -0.1386) * (1.0 + (mo ** 5.31) / 49300000.0)
        isi = ff * np.exp(0.05039 * wind_kmh)
        
        return isi
    
    def calculate_bui(self, dmc, dc):
        """Calculate Buildup Index (BUI)"""
        if dmc <= 0.4 * dc:
            bui = 0.8 * dmc * dc / (dmc + 0.4 * dc)
        else:
            bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
        
        return np.maximum(bui, 0)
    
    def calculate_fwi(self, isi, bui):
        """Calculate Fire Weather Index (FWI)"""
        if bui <= 80.0:
            fD = 0.626 * bui ** 0.809 + 2.0
        else:
            fD = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui))
        
        B = 0.1 * isi * fD
        
        if B > 1.0:
            fwi = np.exp(2.72 * (0.434 * np.log(B)) ** 0.647)
        else:
            fwi = B
        
        return fwi

def load_cmip6_data(csv_file):
    """Load CMIP6 data from CSV file"""
    logger.info(f"Loading CMIP6 data from: {csv_file}")
    
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['time', 'lat', 'lon', 'tasmax', 'huss', 'sfcWind', 'pr']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Debug: show sample time values before conversion
        logger.info(f"Sample time values before conversion: {df['time'].head().tolist()}")
        
        # Convert time column to datetime with robust error handling
        try:
            # First try with mixed format
            df['time'] = pd.to_datetime(df['time'], format='mixed', errors='coerce')
            logger.info("Successfully parsed time with mixed format")
        except Exception as e:
            logger.warning(f"Mixed format failed: {e}")
            try:
                # Try ISO8601 format
                df['time'] = pd.to_datetime(df['time'], format='ISO8601', errors='coerce')
                logger.info("Successfully parsed time with ISO8601 format")
            except Exception as e2:
                logger.warning(f"ISO8601 format failed: {e2}")
                try:
                    # Try default pandas parsing
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                    logger.info("Successfully parsed time with default format")
                except Exception as e3:
                    logger.error(f"All time parsing methods failed: {e3}")
                    return None
        
        # Check for and handle invalid dates
        invalid_dates = df['time'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Found {invalid_dates} invalid dates out of {len(df)} total records")
            logger.warning("Removing rows with invalid dates...")
            df = df.dropna(subset=['time'])
            logger.info(f"Remaining data shape after removing invalid dates: {df.shape}")
        
        # Debug: show sample time values after conversion
        if len(df) > 0:
            logger.info(f"Sample time values after conversion: {df['time'].head().tolist()}")
        
        if len(df) == 0:
            logger.error("No valid data remaining after time conversion")
            return None
        
        # Sort by time, lat, lon for proper time series processing
        df = df.sort_values(['lat', 'lon', 'time']).reset_index(drop=True)
        
        logger.info(f"Data time range: {df['time'].min()} to {df['time'].max()}")
        logger.info(f"Spatial coverage: lat {df['lat'].min():.2f} to {df['lat'].max():.2f}, lon {df['lon'].min():.2f} to {df['lon'].max():.2f}")
        
        # Check data quality
        logger.info("Data quality check:")
        for col in ['tasmax', 'huss', 'sfcWind', 'pr']:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                null_count = df[col].isna().sum()
                logger.info(f"  {col}: min={min_val:.6f}, max={max_val:.6f}, nulls={null_count}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_fwi_timeseries(df, location_group):
    """Calculate FWI time series for a single location"""
    calculator = FWICalculator()
    
    # Initialize previous values
    ffmc_prev = 85.0
    dmc_prev = 6.0
    dc_prev = 15.0
    
    results = []
    
    # Sort by time to ensure proper chronological order
    location_group = location_group.sort_values('time')
    
    for _, row in location_group.iterrows():
        date = row['time']
        month = date.month
        
        # Data validation - check for missing values
        if pd.isna(row['tasmax']) or pd.isna(row['huss']) or pd.isna(row['sfcWind']) or pd.isna(row['pr']):
            logger.warning(f"Skipping row with missing data at {date}")
            continue
        
        # Calculate FFMC with error handling
        try:
            ffmc = calculator.calculate_ffmc(
                row['tasmax'], row['huss'], row['sfcWind'], row['pr'], ffmc_prev
            )
            # Validate result
            if pd.isna(ffmc) or ffmc < 0:
                ffmc = ffmc_prev
        except Exception as e:
            logger.warning(f"Error calculating FFMC at {date}: {e}")
            ffmc = ffmc_prev
        
        # Calculate DMC with error handling
        try:
            dmc = calculator.calculate_dmc(
                row['tasmax'], row['huss'], row['pr'], dmc_prev, month
            )
            # Validate result
            if pd.isna(dmc) or dmc < 0:
                dmc = dmc_prev
        except Exception as e:
            logger.warning(f"Error calculating DMC at {date}: {e}")
            dmc = dmc_prev
        
        # Calculate DC with error handling
        try:
            dc = calculator.calculate_dc(
                row['tasmax'], row['pr'], dc_prev, month
            )
            # Validate result
            if pd.isna(dc) or dc < 0:
                dc = dc_prev
        except Exception as e:
            logger.warning(f"Error calculating DC at {date}: {e}")
            dc = dc_prev
        
        # Calculate ISI with error handling
        try:
            isi = calculator.calculate_isi(row['sfcWind'], ffmc)
            # Validate result
            if pd.isna(isi) or isi < 0:
                isi = 0.0
        except Exception as e:
            logger.warning(f"Error calculating ISI at {date}: {e}")
            isi = 0.0
        
        # Calculate BUI with error handling
        try:
            bui = calculator.calculate_bui(dmc, dc)
            # Validate result
            if pd.isna(bui) or bui < 0:
                bui = 0.0
        except Exception as e:
            logger.warning(f"Error calculating BUI at {date}: {e}")
            bui = 0.0
        
        # Calculate FWI with error handling
        try:
            fwi = calculator.calculate_fwi(isi, bui)
            # Validate result
            if pd.isna(fwi) or fwi < 0:
                fwi = 0.0
        except Exception as e:
            logger.warning(f"Error calculating FWI at {date}: {e}")
            fwi = 0.0
        
        results.append({
            'time': date,
            'lat': row['lat'],
            'lon': row['lon'],
            'tasmax': row['tasmax'],
            'huss': row['huss'],
            'sfcWind': row['sfcWind'],
            'pr': row['pr'],
            'ffmc': ffmc,
            'dmc': dmc,
            'dc': dc,
            'isi': isi,
            'bui': bui,
            'fwi': fwi
        })
        
        # Update previous values for next iteration
        ffmc_prev = ffmc
        dmc_prev = dmc
        dc_prev = dc
    
    return results

def predict_fwi(csv_file, output_file=None):
    """Main function to predict FWI from CMIP6 data"""
    logger.info("Starting FWI prediction from CMIP6 data")
    
    # Load data
    df = load_cmip6_data(csv_file)
    if df is None:
        return False
    
    # Calculate FWI for each location
    logger.info("Calculating FWI for each grid point...")
    
    all_results = []
    locations = df.groupby(['lat', 'lon'])
    
    total_locations = len(locations)
    logger.info(f"Processing {total_locations} grid points...")
    
    for i, (location, location_group) in enumerate(locations):
        lat, lon = location
        
        if (i + 1) % 5 == 0 or i == 0:
            logger.info(f"Processing location {i+1}/{total_locations}: lat={lat:.3f}, lon={lon:.3f}")
        
        # Calculate FWI time series for this location
        location_results = calculate_fwi_timeseries(df, location_group)
        all_results.extend(location_results)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_file = f"{base_name}_fwi_predictions.csv"
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_file)
    
    results_df.to_csv(output_path, index=False)
    
    # Summary statistics
    logger.info("FWI Prediction Summary:")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Total records: {len(results_df):,}")
    logger.info(f"  Time range: {results_df['time'].min()} to {results_df['time'].max()}")
    logger.info(f"  Grid points: {len(results_df.groupby(['lat', 'lon']))}")
    
    print(f"\nFWI Statistics:")
    print(f"  Mean FWI: {results_df['fwi'].mean():.2f}")
    print(f"  Max FWI: {results_df['fwi'].max():.2f}")
    print(f"  Min FWI: {results_df['fwi'].min():.2f}")
    print(f"  Std FWI: {results_df['fwi'].std():.2f}")
    
    # Show sample results
    print(f"\nSample FWI predictions (first 5 rows):")
    sample_cols = ['time', 'lat', 'lon', 'tasmax', 'pr', 'ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi']
    print(results_df[sample_cols].head().to_string(index=False))
    
    # High FWI days
    high_fwi_threshold = 30.0
    high_fwi_days = results_df[results_df['fwi'] > high_fwi_threshold]
    print(f"\nHigh FWI days (FWI > {high_fwi_threshold}): {len(high_fwi_days)} records")
    
    if len(high_fwi_days) > 0:
        print(f"Highest FWI day:")
        max_fwi_day = high_fwi_days.loc[high_fwi_days['fwi'].idxmax()]
        print(f"  Date: {max_fwi_day['time']}")
        print(f"  Location: lat={max_fwi_day['lat']:.3f}, lon={max_fwi_day['lon']:.3f}")
        print(f"  FWI: {max_fwi_day['fwi']:.2f}")
        print(f"  Temperature: {max_fwi_day['tasmax']-273.15:.1f}°C")
        print(f"  Precipitation: {max_fwi_day['pr']*86400:.2f} mm/day")
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nOutput file size: {file_size_mb:.2f} MB")
    
    return True

def main():
    """Main function"""
    print("FWI Prediction from CMIP6 Data")
    print("=" * 50)
    
    # Look for merged CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(script_dir) if f.startswith('cmip6_merged_') and f.endswith('.csv')]
    
    if not csv_files:
        print("No merged CMIP6 CSV files found!")
        print("Please run convert.py first to create the merged CSV file.")
        print("Looking for files with pattern: cmip6_merged_*.csv")
        return
    
    # Use the first (or only) CSV file found
    csv_file = csv_files[0]
    print(f"Using input file: {csv_file}")
    
    csv_path = os.path.join(script_dir, csv_file)
    
    # Debug: First peek at the data structure
    try:
        print("\nDebugging data structure:")
        df_sample = pd.read_csv(csv_path, nrows=5)
        print(f"Sample data shape: {df_sample.shape}")
        print(f"Columns: {list(df_sample.columns)}")
        print("\nFirst few time values:")
        if 'time' in df_sample.columns:
            for i, time_val in enumerate(df_sample['time'].head(3)):
                print(f"  Row {i}: '{time_val}' (type: {type(time_val)})")
        print("\nFirst 3 rows:")
        print(df_sample.head(3).to_string())
    except Exception as e:
        print(f"Error in debug: {e}")
    
    # Predict FWI
    success = predict_fwi(csv_path)
    
    if success:
        print("\nFWI prediction completed successfully!")
        print("Output file contains all FWI components and final FWI values.")
    else:
        print("\nFWI prediction failed!")
        print("Check the error messages above for troubleshooting.")

if __name__ == "__main__":
    main()