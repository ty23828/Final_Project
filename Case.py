import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def load_and_process_fwi_data(csv_file):
    """Load and process FWI data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['date'] = df['Date']
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def filter_june_17_data(df):
    """Filter data for June 17th (assuming 2017 based on the image)"""
    june_17_data = df[df['date'].dt.strftime('%m-%d') == '06-17']
    if len(june_17_data) == 0:
        print("No data found for June 17th, trying different date formats...")
        # Try different years
        for year in [2017, 2018, 2019, 2020, 2021]:
            june_17_data = df[df['date'] == f'{year}-06-17']
            if len(june_17_data) > 0:
                break
    return june_17_data

def filter_october_data(df):
    """Filter and average data for October"""
    october_data = df[df['date'].dt.month == 10]
    if len(october_data) == 0:
        print("No October data found")
        return pd.DataFrame()
    
    # Group by coordinates and calculate mean FWI for October
    october_avg = october_data.groupby(['latitude', 'longitude'])['fwi_predicted'].mean().reset_index()
    october_avg.rename(columns={'fwi_predicted': 'fwi'}, inplace=True)
    return october_avg

def filter_october_high_fwi_data(df, fire_bounds):
    """Filter October data and count days with FWI > 40 in fire area"""
    october_data = df[df['date'].dt.month == 10]
    if len(october_data) == 0:
        print("No October data found")
        return pd.DataFrame()
    
    # Extract fire area boundaries
    north, south, west, east = fire_bounds
    
    # Filter data within fire area boundaries
    fire_area_data = october_data[
        (october_data['latitude'] >= south) & 
        (october_data['latitude'] <= north) &
        (october_data['longitude'] >= west) & 
        (october_data['longitude'] <= east)
    ]
    
    if len(fire_area_data) == 0:
        print("No data found in the specified fire area")
        return pd.DataFrame()
    
    # Count days with FWI > 40 for each grid point
    high_fwi_counts = fire_area_data[fire_area_data['fwi_predicted'] > 40].groupby(['latitude', 'longitude']).size().reset_index(name='high_fwi_days')
    
    # For grid points with no high FWI days, set count to 0
    all_grid_points = fire_area_data.groupby(['latitude', 'longitude']).size().reset_index(name='total_days')
    
    # Merge to get complete grid with counts
    result = all_grid_points.merge(high_fwi_counts, on=['latitude', 'longitude'], how='left')
    result['high_fwi_days'] = result['high_fwi_days'].fillna(0)
    
    return result

def create_fwi_map(data, title, fire_coords=None, fire_type='point'):
    """Create FWI map with fire area highlighted"""
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map extent based on data
    if len(data) > 0:
        lat_min, lat_max = data['latitude'].min(), data['latitude'].max()
        lon_min, lon_max = data['longitude'].min(), data['longitude'].max()
        
        # Add some padding
        lat_padding = (lat_max - lat_min) * 0.1
        lon_padding = (lon_max - lon_min) * 0.1
        
        ax.set_extent([lon_min - lon_padding, lon_max + lon_padding, 
                      lat_min - lat_padding, lat_max + lat_padding], 
                     crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    
    # Create scatter plot - use the correct column name
    fwi_column = 'fwi' if 'fwi' in data.columns else 'fwi_predicted'
    if len(data) > 0:
        scatter = ax.scatter(data['longitude'], data['latitude'], 
                           c=data[fwi_column], cmap='YlOrRd', 
                           s=20, transform=ccrs.PlateCarree(),
                           vmin=0, vmax=data[fwi_column].quantile(0.95))
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label('Fire Weather Index (FWI)', fontsize=12)
    
    # Highlight fire areas
    if fire_coords:
        if fire_type == 'point':
            # For June 17th - highlight specific points
            for coord in fire_coords:
                lat, lon = coord
                ax.plot(lon, lat, 'ro', markersize=15, markerfacecolor='red', 
                       markeredgecolor='black', markeredgewidth=2,
                       transform=ccrs.PlateCarree(), label='Fire Location')
                
                # Add circle around fire location
                circle = plt.Circle((lon, lat), 0.05, fill=False, 
                                  color='red', linewidth=3, 
                                  transform=ccrs.PlateCarree())
                ax.add_patch(circle)
        
        elif fire_type == 'rectangle':
            # For October - highlight rectangular area
            north, south, west, east = fire_coords
            width = east - west
            height = north - south
            
            rect = Rectangle((west, south), width, height, 
                           fill=False, edgecolor='red', linewidth=3,
                           transform=ccrs.PlateCarree())
            ax.add_patch(rect)
            
            # Add label
            ax.text((west + east)/2, north + 0.02, 'Fire Area', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                   color='red', transform=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    if fire_coords and fire_type == 'point':
        plt.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

def create_fwi_count_map(data, title, fire_coords=None):
    """Create map showing count of high FWI days"""
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map extent based on data
    if len(data) > 0:
        lat_min, lat_max = data['latitude'].min(), data['latitude'].max()
        lon_min, lon_max = data['longitude'].min(), data['longitude'].max()
        
        # Add some padding
        lat_padding = (lat_max - lat_min) * 0.2
        lon_padding = (lon_max - lon_min) * 0.2
        
        ax.set_extent([lon_min - lon_padding, lon_max + lon_padding, 
                      lat_min - lat_padding, lat_max + lat_padding], 
                     crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8)
    ax.add_feature(cfeature.LAND, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.5)
    
    # Create scatter plot for high FWI day counts
    if len(data) > 0:
        scatter = ax.scatter(data['longitude'], data['latitude'], 
                           c=data['high_fwi_days'], cmap='Reds', 
                           s=50, transform=ccrs.PlateCarree(),
                           vmin=0, vmax=data['high_fwi_days'].max(),
                           edgecolors='black', linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label('Number of Days with FWI > 40', fontsize=12)
    
    # Highlight fire area boundary
    if fire_coords:
        north, south, west, east = fire_coords
        width = east - west
        height = north - south
        
        rect = Rectangle((west, south), width, height, 
                       fill=False, edgecolor='blue', linewidth=3,
                       linestyle='--', transform=ccrs.PlateCarree())
        ax.add_patch(rect)
        
        # Add label
        ax.text((west + east)/2, north + 0.01, 'Plano de Gestão Florestal Area', 
               ha='center', va='bottom', fontsize=12, fontweight='bold',
               color='blue', transform=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def main():
    """Main function to create FWI maps for June 17th and October high FWI count"""
    
    # Load data
    csv_file = 'fwi_1km_predictions_random_forest.csv'  # Update path if needed
    df = load_and_process_fwi_data(csv_file)
    
    if df is None:
        print("Failed to load data")
        return
    
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total records: {len(df)}")
    
    # Filter data for June 17th
    june_17_data = filter_june_17_data(df)
    print(f"June 17th data points: {len(june_17_data)}")
    
    # Add fwi column for June data (rename fwi_predicted to fwi for consistency)
    if len(june_17_data) > 0:
        june_17_data = june_17_data.copy()
        june_17_data['fwi'] = june_17_data['fwi_predicted']
    
    # Define coordinates
    # June 17th fire locations
    june_fire_coords = [
        (39.9606, -8.1762),  # First fire location
        (39.95, -8.25)       # Second fire location
    ]
    
    # Plano de Gestão Florestal area boundaries (October fire area)
    forest_management_bounds = (39.86, 39.72, -9.05, -8.90)  # (north, south, west, east)
    
    # Filter October data for high FWI count analysis
    october_high_fwi_data = filter_october_high_fwi_data(df, forest_management_bounds)
    print(f"October high FWI analysis data points: {len(october_high_fwi_data)}")
    
    # Create June 17th map
    if len(june_17_data) > 0:
        fig1 = create_fwi_map(june_17_data, 
                             'Fire Weather Index (FWI) - June 17th, 2017\nFire Locations Highlighted',
                             june_fire_coords, 'point')
        plt.savefig('fwi_june_17_fire_locations.png', dpi=300, bbox_inches='tight')
        print("June 17th map saved as 'fwi_june_17_fire_locations.png'")
        plt.show()
    else:
        print("No data available for June 17th")
    
    # Create October high FWI count map
    if len(october_high_fwi_data) > 0:
        fig2 = create_fwi_count_map(october_high_fwi_data, 
                                   'October 2017 - Days with FWI > 40\nin Plano de Gestão Florestal Area',
                                   forest_management_bounds)
        plt.savefig('october_high_fwi_count_map.png', dpi=300, bbox_inches='tight')
        print("October high FWI count map saved as 'october_high_fwi_count_map.png'")
        plt.show()
    else:
        print("No data available for October high FWI analysis")
    
    # Print summary statistics
    if len(june_17_data) > 0:
        print(f"\nJune 17th FWI Statistics:")
        print(f"Mean FWI: {june_17_data['fwi_predicted'].mean():.2f}")
        print(f"Max FWI: {june_17_data['fwi_predicted'].max():.2f}")
        print(f"Min FWI: {june_17_data['fwi_predicted'].min():.2f}")
    
    if len(october_high_fwi_data) > 0:
        print(f"\nOctober High FWI (>40) Statistics in Forest Management Area:")
        print(f"Total grid points analyzed: {len(october_high_fwi_data)}")
        print(f"Max days with FWI > 40: {october_high_fwi_data['high_fwi_days'].max():.0f}")
        print(f"Mean days with FWI > 40: {october_high_fwi_data['high_fwi_days'].mean():.2f}")
        print(f"Grid points with FWI > 40 at least once: {(october_high_fwi_data['high_fwi_days'] > 0).sum()}")
        print(f"Percentage of grid points affected: {((october_high_fwi_data['high_fwi_days'] > 0).sum() / len(october_high_fwi_data) * 100):.1f}%")

if __name__ == "__main__":
    main()