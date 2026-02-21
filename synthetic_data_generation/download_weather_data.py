"""
Download hourly weather data for Montreal from Government of Canada Climate Data
Time periods:
- Base paper: May-September 2019 (weekdays)
- Reference 22: June-August 2017-2020

Data source: https://climate.weather.gc.ca
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# Montreal Pierre Elliott Trudeau Int'l Airport Station
STATION_ID = 51157  # Station ID for hourly data
STATION_NAME = "MONTREAL/PIERRE ELLIOTT TRUDEAU INTL A"

def download_climate_data(year, month, station_id=STATION_ID):
    """
    Download hourly climate data for a specific year and month from Environment Canada.
    
    Parameters:
    - year: int, year to download
    - month: int, month to download (1-12)
    - station_id: int, station ID
    
    Returns:
    - DataFrame with hourly weather data
    """
    # Environment Canada bulk data URL format
    base_url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    
    params = {
        'format': 'csv',
        'stationID': station_id,
        'Year': year,
        'Month': month,
        'Day': 1,  # Not used for monthly download but required
        'timeframe': 1,  # 1 = hourly, 2 = daily, 3 = monthly
        'submit': 'Download Data'
    }
    
    print(f"Downloading {year}-{month:02d}...", end=" ")
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file and read with pandas
        temp_file = f"temp_{year}_{month:02d}.csv"
        with open(temp_file, 'wb') as f:
            f.write(response.content)
        
        # Read CSV (skip if empty or error)
        df = pd.read_csv(temp_file, encoding='utf-8')
        
        # Clean up temp file
        Path(temp_file).unlink()
        
        print(f"✓ ({len(df)} records)")
        return df
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def download_period(start_year, start_month, end_year, end_month, output_file):
    """
    Download weather data for a period and save to CSV.
    
    Parameters:
    - start_year, start_month: Start date
    - end_year, end_month: End date
    - output_file: Path to save combined CSV
    """
    all_data = []
    
    current_year = start_year
    current_month = start_month
    
    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        df = download_climate_data(current_year, current_month)
        
        if df is not None and not df.empty:
            all_data.append(df)
        
        # Move to next month
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
        
        # Be nice to the server
        time.sleep(1)
    
    if all_data:
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save to file
        combined_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved {len(combined_df)} records to {output_file}")
        
        return combined_df
    else:
        print("\n✗ No data downloaded")
        return None

def filter_weekdays(df):
    """Filter to only weekdays (Monday-Friday)"""
    # Convert to datetime - check for different column name formats
    datetime_col = None
    for col in ['Date/Time (LST)', 'Date/Time', 'DateTime']:
        if col in df.columns:
            datetime_col = col
            break
    
    if datetime_col is None:
        print("Warning: No datetime column found!")
        print(f"Available columns: {df.columns.tolist()}")
        return df
    
    # Convert to datetime
    df['DateTime'] = pd.to_datetime(df[datetime_col], format='mixed', errors='coerce')
    
    # Get day of week (0=Monday, 6=Sunday)
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    
    # Filter weekdays only (0-4)
    weekdays_df = df[df['DayOfWeek'] < 5].copy()
    
    print(f"Filtered to {len(weekdays_df)} weekday records (from {len(df)} total)")
    
    return weekdays_df

def main():
    """Main function to download weather data for both time periods"""
    
    # Create output directory
    output_dir = Path("../data/weather")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DOWNLOADING MONTREAL WEATHER DATA")
    print("=" * 70)
    print(f"Station: {STATION_NAME}")
    print(f"Station ID: {STATION_ID}")
    print()
    
    # Option 1: Base paper period (May-September 2019)
    print("\n--- Period 1: May-September 2019 (Base Paper) ---")
    df_2019 = download_period(
        start_year=2019, start_month=5,
        end_year=2019, end_month=9,
        output_file=output_dir / "montreal_weather_2019_may_sep.csv"
    )
    
    if df_2019 is not None:
        # Filter to weekdays only
        df_2019_weekdays = filter_weekdays(df_2019)
        df_2019_weekdays.to_csv(
            output_dir / "montreal_weather_2019_may_sep_weekdays.csv",
            index=False
        )
    
    # Option 2: Reference 22 period (June-August 2017-2020)
    print("\n--- Period 2: June-August 2017-2020 (Reference 22) ---")
    
    for year in [2017, 2018, 2019, 2020]:
        print(f"\nYear {year}:")
        df_year = download_period(
            start_year=year, start_month=6,
            end_year=year, end_month=8,
            output_file=output_dir / f"montreal_weather_{year}_jun_aug.csv"
        )
    
    # Combine all years for Reference 22 period
    print("\n--- Combining 2017-2020 data ---")
    all_years = []
    for year in [2017, 2018, 2019, 2020]:
        file_path = output_dir / f"montreal_weather_{year}_jun_aug.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            all_years.append(df)
    
    if all_years:
        combined_df = pd.concat(all_years, ignore_index=True)
        combined_df.to_csv(
            output_dir / "montreal_weather_2017_2020_jun_aug_combined.csv",
            index=False
        )
        print(f"✓ Combined dataset: {len(combined_df)} records")
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"\nFiles saved in: {output_dir.absolute()}")
    print("\nKey fields in the data:")
    if df_2019 is not None:
        print(df_2019.columns.tolist())

if __name__ == "__main__":
    main()
