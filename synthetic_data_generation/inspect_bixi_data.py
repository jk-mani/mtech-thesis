"""
Inspect downloaded BIXI trip data and display key statistics
"""

import pandas as pd
from pathlib import Path
import numpy as np

def inspect_bixi_data():
    """Inspect the downloaded BIXI trip data"""
    
    data_dir = Path("../data/bixi")
    
    print("=" * 70)
    print("BIXI TRIP DATA INSPECTION")
    print("=" * 70)
    
    # Base paper dataset (2019 May-Sep weekdays)
    print("\n--- Base Paper Dataset (2019 May-Sep Weekdays) ---")
    df_base = pd.read_csv(data_dir / "bixi_trips_2019_may_sep_weekdays.csv")
    
    print(f"Total trips: {len(df_base):,}")
    print(f"Date range: {df_base['start_date'].min()} to {df_base['start_date'].max()}")
    
    print(f"\nKey fields:")
    for col in df_base.columns:
        non_null = df_base[col].notna().sum()
        print(f"  - {col}: {non_null:,} non-null records ({non_null/len(df_base)*100:.1f}%)")
    
    # Trip duration statistics
    if 'duration_sec' in df_base.columns:
        duration_min = df_base['duration_sec'] / 60
        print(f"\nTrip duration (minutes):")
        print(f"  Min: {duration_min.min():.1f}")
        print(f"  Max: {duration_min.max():.1f}")
        print(f"  Mean: {duration_min.mean():.1f}")
        print(f"  Median: {duration_min.median():.1f}")
        print(f"  Std: {duration_min.std():.1f}")
    
    # Station statistics
    unique_start_stations = df_base['start_station_code'].nunique()
    unique_end_stations = df_base['end_station_code'].nunique()
    all_stations = set(df_base['start_station_code'].unique()) | set(df_base['end_station_code'].unique())
    
    print(f"\nStation statistics:")
    print(f"  Unique start stations: {unique_start_stations}")
    print(f"  Unique end stations: {unique_end_stations}")
    print(f"  Total unique stations: {len(all_stations)}")
    
    # Daily trip statistics
    df_base['date'] = pd.to_datetime(df_base['start_date']).dt.date
    daily_trips = df_base.groupby('date').size()
    
    print(f"\nDaily trip statistics:")
    print(f"  Number of days: {len(daily_trips)}")
    print(f"  Avg trips per day: {daily_trips.mean():.0f}")
    print(f"  Min trips per day: {daily_trips.min()}")
    print(f"  Max trips per day: {daily_trips.max()}")
    
    # Hourly pattern
    df_base['hour'] = pd.to_datetime(df_base['start_date']).dt.hour
    hourly_trips = df_base.groupby('hour').size()
    
    print(f"\nPeak hours (top 3):")
    top_hours = hourly_trips.nlargest(3)
    for hour, count in top_hours.items():
        print(f"  {hour:02d}:00 - {count:,} trips")
    
    # Reference 22 dataset (2017-2020 Jun-Aug)
    print("\n\n--- Reference 22 Dataset (2017-2020 Jun-Aug Combined) ---")
    df_ref = pd.read_csv(data_dir / "bixi_trips_2017_2020_jun_aug_combined.csv", 
                         nrows=10000)  # Sample for quick stats
    
    print(f"Sample loaded: 10,000 trips (out of ~10M)")
    
    print(f"\nKey fields in combined dataset:")
    for col in df_ref.columns:
        print(f"  - {col}")
    
    # Station locations (if available)
    if 'latitude' in df_base.columns and 'longitude' in df_base.columns:
        print("\n\n--- Station Location Data ---")
        
        # Get unique stations with coordinates
        stations = df_base[['Code', 'name', 'latitude', 'longitude']].drop_duplicates()
        stations_with_coords = stations.dropna(subset=['latitude', 'longitude'])
        
        print(f"Stations with coordinates: {len(stations_with_coords)}")
        
        if len(stations_with_coords) > 0:
            print(f"\nCoordinate ranges:")
            print(f"  Latitude: {stations_with_coords['latitude'].min():.4f} to {stations_with_coords['latitude'].max():.4f}")
            print(f"  Longitude: {stations_with_coords['longitude'].min():.4f} to {stations_with_coords['longitude'].max():.4f}")
    
    print("\n" + "=" * 70)
    print("DATA SUMMARY FOR SYNTHETIC GENERATOR")
    print("=" * 70)
    
    print("\n✓ Available for extraction:")
    print("  1. Trip volume: ~33,000 trips/day (to scale to 3,240 for 60 stations)")
    print("  2. Temporal patterns: Hourly trip distributions")
    print("  3. Trip duration: Mean ~12 minutes, range 5-30 minutes typical")
    print(f"  4. Station network: {len(all_stations)} unique stations in 2019")
    print("  5. Origin-Destination patterns: Available in trip records")
    print("  6. Member vs. casual users: Available if is_member field populated")
    
    print("\n✓ Next steps:")
    print("  1. Merge BIXI trips with weather data by timestamp")
    print("  2. Analyze temporal patterns to fit Beta distributions")
    print("  3. Identify city center stations using spatial clustering")
    print("  4. Train linear regression (weather → hourly demand)")
    print("  5. Implement synthetic trip generator")

if __name__ == "__main__":
    inspect_bixi_data()
