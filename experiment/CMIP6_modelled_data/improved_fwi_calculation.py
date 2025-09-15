#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
improved_fwi_calculation.py

Improved FWI calculation based on Canadian Forest Fire Weather Index System
"""

import numpy as np
import pandas as pd

class ImprovedFWICalculator:
    """Improved FWI calculator based on meteorological principles"""
    
    def __init__(self):
        self.previous_ffmc = 85.0
        self.previous_dmc = 6.0
        self.previous_dc = 15.0
    
    def calculate_relative_humidity(self, temp_k, dewpoint_k):
        """Calculate relative humidity from temperature and dewpoint"""
        # August-Roche-Magnus approximation
        temp_c = temp_k - 273.15
        dewpoint_c = dewpoint_k - 273.15
        
        rh = 100 * np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c)) / \
             np.exp((17.625 * temp_c) / (243.04 + temp_c))
        
        return np.clip(rh, 0, 100)
    
    def calculate_ffmc(self, temp_c, rh, wind_kmh, rain_mm, previous_ffmc):
        """Calculate Fine Fuel Moisture Code"""
        # Rain effect
        if rain_mm > 0.5:
            rf = rain_mm - 0.5
            if previous_ffmc <= 65.0:
                mr = previous_ffmc + 42.5 * rf * np.exp(-100.0 / (251.0 - previous_ffmc)) * (1.0 - np.exp(-6.93 / rf))
            else:
                mr = previous_ffmc + 42.5 * rf * np.exp(-100.0 / (251.0 - previous_ffmc)) * (1.0 - np.exp(-6.93 / rf)) + \
                     0.0015 * (previous_ffmc - 65.0) ** 2 * rf ** 0.5
            
            mr = np.minimum(mr, 101.0)
            ffmc = 59.5 * (250.0 - mr) / (147.2 + mr)
        else:
            mr = 147.2 * (101.0 - previous_ffmc) / (59.5 + previous_ffmc)
        
        # Drying
        if rain_mm <= 0.5:
            # Calculate equilibrium moisture content
            if rh < 10.0:
                emc = 0.25 * (rh - 10.0) + 1.0
            elif rh <= 50.0:
                emc = 0.125 * (rh - 10.0) + 2.5
            else:
                emc = 2.5 + 0.0685 * (rh - 50.0)
            
            # Drying rate
            if mr > emc:
                ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * wind_kmh ** 0.5 * (1.0 - (rh / 100.0) ** 8)
                kd = ko * 0.581 * np.exp(0.0365 * temp_c)
                mr = emc + (mr - emc) * np.exp(-kd)
            
            ffmc = 59.5 * (250.0 - mr) / (147.2 + mr)
        
        return np.clip(ffmc, 0, 101)
    
    def calculate_dmc(self, temp_c, rh, rain_mm, previous_dmc, month):
        """Calculate Duff Moisture Code"""
        # Rain effect
        if rain_mm > 1.5:
            re = 0.92 * rain_mm - 1.27
            mo = 20.0 + 280.0 / np.exp(0.023 * previous_dmc)
            
            if previous_dmc <= 33.0:
                b = 100.0 / (0.5 + 0.3 * previous_dmc)
            elif previous_dmc <= 65.0:
                b = 14.0 - 1.3 * np.log(previous_dmc)
            else:
                b = 6.2 * np.log(previous_dmc) - 17.2
            
            mr = mo + 1000.0 * re / (48.77 + b * re)
            pr = 244.72 - 43.43 * np.log(mr - 20.0)
            dmc = np.maximum(pr, 0)
        else:
            dmc = previous_dmc
        
        # Drying
        if temp_c > -1.1:
            # Day length factor (simplified for Portugal latitude ~39°N)
            day_length_factors = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 8.4, 6.8, 6.2, 6.5]
            el = day_length_factors[month - 1]
            
            if rh <= 21.0:
                k = 1.894 * (temp_c + 1.1) * (100.0 - rh) * el * 0.000001
            else:
                k = 1.894 * (temp_c + 1.1) * (100.0 - rh) * el * 0.000001 * \
                    (1.0 + (rh - 21.0) * (rh - 21.0) / 1600.0)
            
            dmc = dmc + k
        
        return np.maximum(dmc, 0)
    
    def calculate_dc(self, temp_c, rain_mm, previous_dc, month):
        """Calculate Drought Code"""
        # Rain effect
        if rain_mm > 2.8:
            rd = 0.83 * rain_mm - 1.27
            qo = 800.0 * np.exp(-previous_dc / 400.0)
            qr = qo + 3.937 * rd
            dr = 400.0 * np.log(800.0 / qr)
            dc = np.maximum(dr, 0)
        else:
            dc = previous_dc
        
        # Drying
        if temp_c > -2.8:
            # Day length factor (simplified for Portugal latitude)
            day_length_factors = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
            lf = day_length_factors[month - 1]
            
            v = 0.36 * (temp_c + 2.8) + lf
            v = np.maximum(v, 0)
            dc = dc + 0.5 * v
        
        return np.maximum(dc, 0)
    
    def calculate_isi(self, wind_kmh, ffmc):
        """Calculate Initial Spread Index"""
        fm = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        ff = 19.115 * np.exp(-0.1386 * fm) * (1.0 + fm ** 5.31 / 4.93e7)
        isi = ff * np.exp(0.05039 * wind_kmh)
        return isi
    
    def calculate_bui(self, dmc, dc):
        """Calculate Buildup Index"""
        if dmc <= 0.4 * dc:
            bui = 0.8 * dmc * dc / (dmc + 0.4 * dc)
        else:
            bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
        
        return np.maximum(bui, 0)
    
    def calculate_fwi(self, isi, bui):
        """Calculate Fire Weather Index"""
        if bui <= 80.0:
            fd = 0.626 * bui ** 0.809 + 2.0
        else:
            fd = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui))
        
        b = 0.1 * isi * fd
        
        if b <= 1.0:
            s = b
        else:
            s = np.exp(2.72 * (0.434 * np.log(b)) ** 0.647)
        
        fwi = s
        return fwi
    
    def calculate_daily_fwi(self, df):
        """Calculate FWI for daily data"""
        results = []
        
        # Group by location and calculate FWI time series
        for (lat, lon), group in df.groupby(['lat', 'lon']):
            group = group.sort_values('time')
            
            # Initialize moisture codes
            ffmc = self.previous_ffmc
            dmc = self.previous_dmc
            dc = self.previous_dc
            
            for _, row in group.iterrows():
                # Convert units
                temp_c = row['tasmax'] - 273.15
                wind_kmh = row['sfcWind'] * 3.6
                rain_mm = row['pr'] * 86400  # Convert m/s to mm/day
                
                # Calculate relative humidity (simplified)
                # If specific humidity is available, use it
                rh = row['huss'] * 100  # Convert to percentage (simplified)
                rh = np.clip(rh, 0, 100)
                
                # Get month
                month = pd.to_datetime(row['time']).month
                
                # Calculate moisture codes
                ffmc = self.calculate_ffmc(temp_c, rh, wind_kmh, rain_mm, ffmc)
                dmc = self.calculate_dmc(temp_c, rh, rain_mm, dmc, month)
                dc = self.calculate_dc(temp_c, rain_mm, dc, month)
                
                # Calculate indices
                isi = self.calculate_isi(wind_kmh, ffmc)
                bui = self.calculate_bui(dmc, dc)
                fwi = self.calculate_fwi(isi, bui)
                
                results.append({
                    'time': row['time'],
                    'lat': lat,
                    'lon': lon,
                    'fwi_improved': fwi,
                    'ffmc': ffmc,
                    'dmc': dmc,
                    'dc': dc,
                    'isi': isi,
                    'bui': bui
                })
        
        return pd.DataFrame(results)