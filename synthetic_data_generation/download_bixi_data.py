"""
Download BIXI Montreal trip data from their open data portal
Time periods:
- Base paper: May-September 2019 (weekdays)
- Reference 22: June-August 2017-2020

Data source: https://bixi.com/en/open-data-2/
"""

import requests
import zipfile
import pandas as pd
from pathlib import Path
import time
import io

# BIXI open data URLs (yearly files):
# https://s3.ca-central-1.amazonaws.com/cdn.bixi.com/wp-content/uploads/2023/06/Historique-BIXI-[YEAR].zip

BIXI_URLS = {
    2017: "https://s3.ca-central-1.amazonaws.com/cdn.bixi.com/wp-content/uploads/2023/06/Historique-BIXI-2017.zip",
    2018: "https://s3.ca-central-1.amazonaws.com/cdn.bixi.com/wp-content/uploads/2023/06/Historique-BIXI-2018.zip",
    2019: "https://s3.ca-central-1.amazonaws.com/cdn.bixi.com/wp-content/uploads/2023/06/Historique-BIXI-2019.zip",
    2020: "https://s3.ca-central-1.amazonaws.com/cdn.bixi.com/wp-content/uploads/2023/06/Historique-BIXI-2020.zip",
}

def download_bixi_year(year, output_dir):
    """
    Download BIXI trip data for a specific year.
    
    Parameters:
    - year: int, year to download
    - output_dir: Path object, directory to save data
    
    Returns:
    - DataFrame with trip data or None if failed
    """
    if year not in BIXI_URLS:
        print(f"✗ Year {year} not available")
        return None
    
    url = BIXI_URLS[year]
    print(f"Downloading {year}...", end=" ")
    
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        # Extract ZIP file
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Find CSV files in ZIP
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            
            if not csv_files:
                print(f"✗ No CSV files found in ZIP")
                return None
            
            # Read all CSV files (some years may have multiple files)
            dfs = []
            for csv_file in csv_files:
                with z.open(csv_file) as f:
                    # Try different encodings
                    try:
                        df = pd.read_csv(f, encoding='utf-8')
                        dfs.append(df)
                    except UnicodeDecodeError:
                        f.seek(0)
                        df = pd.read_csv(f, encoding='latin-1')
                        dfs.append(df)
            
            # Combine if multiple files
            if len(dfs) > 1:
                df = pd.concat(dfs, ignore_index=True)
            else:
                df = dfs[0]
            
            # Save to CSV
            output_file = output_dir / f"bixi_trips_{year}_full.csv"
            df.to_csv(output_file, index=False)
            
            print(f"✓ ({len(df):,} trips)")
            return df
                
    except requests.exceptions.RequestException as e:
        print(f"✗ Download failed: {e}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def filter_months(df, months):
    """
    Filter dataframe to specific months.
    
    Parameters:
    - df: DataFrame with trip data
    - months: list of months to keep (1-12)
    
    Returns:
    - Filtered DataFrame
    """
    # Check for date column names
    date_col = None
    for col in ['start_date', 'Start date', 'Start Date', 'DATE', 'Start Date']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        print("Warning: No date column found for month filtering")
        return df
    
    # Convert to datetime
    df['DateTime'] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
    df['Month'] = df['DateTime'].dt.month
    
    # Filter to specified months
    filtered_df = df[df['Month'].isin(months)].copy()
    
    print(f"  Filtered to months {months}: {len(filtered_df):,} trips (from {len(df):,})")
    
    return filtered_df

def filter_weekdays(df):
    """Filter to only weekdays (Monday-Friday)"""
    # Check for date column names (BIXI uses different names across years)
    date_col = None
    for col in ['start_date', 'Start date', 'Start Date', 'DATE']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        print("Warning: No date column found!")
        print(f"Available columns: {df.columns.tolist()}")
        return df
    
    # Convert to datetime
    df['DateTime'] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
    
    # Get day of week (0=Monday, 6=Sunday)
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    
    # Filter weekdays only (0-4)
    weekdays_df = df[df['DayOfWeek'] < 5].copy()
    
    print(f"Filtered to {len(weekdays_df)} weekday trips (from {len(df)} total)")
    
    return weekdays_df

def main():
    """Main function to download BIXI trip data"""
    
    # Create output directory
    output_dir = Path("../data/bixi")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DOWNLOADING BIXI MONTREAL TRIP DATA")
    print("=" * 70)
    print("Data source: https://bixi.com/en/open-data/")
    print()
    
    # Download all needed years
    years_data = {}
    
    print("--- Downloading yearly data ---")
    for year in [2017, 2018, 2019, 2020]:
        df = download_bixi_year(year, output_dir)
        if df is not None:
            years_data[year] = df
        time.sleep(2)  # Be nice to the server
    
    if not years_data:
        print("\n✗ No data downloaded!")
        return
    
    # Option 1: Base paper period (May-September 2019)
    print("\n\n--- Period 1: May-September 2019 (Base Paper) ---")
    if 2019 in years_data:
        df_2019 = years_data[2019]
        
        # Show column names
        print(f"Columns in 2019 data: {df_2019.columns.tolist()}")
        
        # Filter to May-September
        df_2019_may_sep = filter_months(df_2019, [5, 6, 7, 8, 9])
        df_2019_may_sep.to_csv(
            output_dir / "bixi_trips_2019_may_sep.csv",
            index=False
        )
        
        # Filter to weekdays only
        print("  Filtering to weekdays...")
        df_2019_weekdays = filter_weekdays(df_2019_may_sep)
        df_2019_weekdays.to_csv(
            output_dir / "bixi_trips_2019_may_sep_weekdays.csv",
            index=False
        )
        print(f"  ✓ Saved May-Sep 2019 (all days and weekdays only)")
    else:
        print("✗ 2019 data not available")
    
    # Option 2: Reference 22 period (June-August 2017-2020)
    print("\n\n--- Period 2: June-August 2017-2020 (Reference 22) ---")
    
    jun_aug_data = []
    for year in [2017, 2018, 2019, 2020]:
        if year in years_data:
            print(f"\nProcessing {year}:")
            df_year = years_data[year]
            
            # Filter to June-August
            df_jun_aug = filter_months(df_year, [6, 7, 8])
            df_jun_aug.to_csv(
                output_dir / f"bixi_trips_{year}_jun_aug.csv",
                index=False
            )
            jun_aug_data.append(df_jun_aug)
            
            if year == 2017:
                # Show column structure for reference
                print(f"  Column structure (example from {year}):")
                print(f"    Total columns: {len(df_jun_aug.columns)}")
                print(f"    Sample columns: {df_jun_aug.columns.tolist()[:5]}...")
    
    # Combine all years for Reference 22 period
    if jun_aug_data:
        print("\n--- Combining 2017-2020 June-August data ---")
        
        # Check if all dataframes have the same columns
        cols_sets = [set(df.columns) for df in jun_aug_data]
        common_cols = set.intersection(*cols_sets)
        
        if len(common_cols) < len(jun_aug_data[0].columns):
            print(f"⚠ Warning: Different column structures across years")
            print(f"  Using common columns only: {len(common_cols)} columns")
            jun_aug_data = [df[list(common_cols)] for df in jun_aug_data]
        
        combined_df = pd.concat(jun_aug_data, ignore_index=True)
        combined_df.to_csv(
            output_dir / "bixi_trips_2017_2020_jun_aug_combined.csv",
            index=False
        )
        print(f"✓ Combined dataset: {len(combined_df):,} trips")
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"\nFiles saved in: {output_dir.absolute()}")
    
    # Show typical trip data structure
    if 2019 in years_data:
        df_sample = years_data[2019]
        print("\n--- BIXI Trip Data Structure ---")
        print("\nExpected fields:")
        print("  - start_date/Start date: Departure timestamp")
        print("  - end_date/End date: Arrival timestamp")  
        print("  - start_station_code: Origin station ID")
        print("  - end_station_code: Destination station ID")
        print("  - duration_sec: Trip duration in seconds")
        print("  - is_member: User type (member/casual)")
        print("\nActual columns found in 2019 data:")
        for col in df_sample.columns:
            print(f"  - {col}")
        
        # Show sample trip
        print("\nSample trip record:")
        print(df_sample.head(1).to_dict('records')[0])

if __name__ == "__main__":
    main()
