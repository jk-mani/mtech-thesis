"""
Inspect downloaded weather data and display key statistics
"""

import pandas as pd
from pathlib import Path

def inspect_weather_data():
    """Inspect the downloaded weather data"""
    
    data_dir = Path("../data/weather")
    
    print("=" * 70)
    print("WEATHER DATA INSPECTION")
    print("=" * 70)
    
    # Base paper dataset (2019 May-Sep weekdays)
    print("\n--- Base Paper Dataset (2019 May-Sep Weekdays) ---")
    df_base = pd.read_csv(data_dir / "montreal_weather_2019_may_sep_weekdays.csv")
    
    print(f"Total records: {len(df_base)}")
    print(f"Date range: {df_base['Date/Time (LST)'].min()} to {df_base['Date/Time (LST)'].max()}")
    print(f"\nKey fields available:")
    print(f"  - Temperature: {df_base['Temp (°C)'].notna().sum()} non-null records")
    print(f"  - Humidity: {df_base['Rel Hum (%)'].notna().sum()} non-null records")
    print(f"  - Date/Time: {df_base['Date/Time (LST)'].notna().sum()} non-null records")
    
    print(f"\nTemperature statistics (°C):")
    print(f"  Min: {df_base['Temp (°C)'].min():.1f}")
    print(f"  Max: {df_base['Temp (°C)'].max():.1f}")
    print(f"  Mean: {df_base['Temp (°C)'].mean():.1f}")
    print(f"  Std: {df_base['Temp (°C)'].std():.1f}")
    
    print(f"\nHumidity statistics (%):")
    print(f"  Min: {df_base['Rel Hum (%)'].min():.1f}")
    print(f"  Max: {df_base['Rel Hum (%)'].max():.1f}")
    print(f"  Mean: {df_base['Rel Hum (%)'].mean():.1f}")
    print(f"  Std: {df_base['Rel Hum (%)'].std():.1f}")
    
    # Reference 22 dataset (2017-2020 Jun-Aug)
    print("\n\n--- Reference 22 Dataset (2017-2020 Jun-Aug Combined) ---")
    df_ref = pd.read_csv(data_dir / "montreal_weather_2017_2020_jun_aug_combined.csv")
    
    print(f"Total records: {len(df_ref)}")
    print(f"Years covered: {sorted(df_ref['Year'].unique())}")
    print(f"Months covered: {sorted(df_ref['Month'].unique())}")
    
    print(f"\nKey fields available:")
    print(f"  - Temperature: {df_ref['Temp (°C)'].notna().sum()} non-null records")
    print(f"  - Humidity: {df_ref['Rel Hum (%)'].notna().sum()} non-null records")
    
    print(f"\nTemperature statistics (°C):")
    print(f"  Min: {df_ref['Temp (°C)'].min():.1f}")
    print(f"  Max: {df_ref['Temp (°C)'].max():.1f}")
    print(f"  Mean: {df_ref['Temp (°C)'].mean():.1f}")
    print(f"  Std: {df_ref['Temp (°C)'].std():.1f}")
    
    print(f"\nHumidity statistics (%):")
    print(f"  Min: {df_ref['Rel Hum (%)'].min():.1f}")
    print(f"  Max: {df_ref['Rel Hum (%)'].max():.1f}")
    print(f"  Mean: {df_ref['Rel Hum (%)'].mean():.1f}")
    print(f"  Std: {df_ref['Rel Hum (%)'].std():.1f}")
    
    # Show all available columns
    print("\n\n--- All Available Columns ---")
    print("Columns that will be used for synthetic data generation:")
    important_cols = [
        'Date/Time (LST)',
        'Year',
        'Month', 
        'Day',
        'Time (LST)',
        'Temp (°C)',
        'Rel Hum (%)',
    ]
    print("\nIMPORTANT COLUMNS:")
    for col in important_cols:
        if col in df_base.columns:
            print(f"  ✓ {col}")
    
    print("\nOTHER AVAILABLE COLUMNS:")
    other_cols = [col for col in df_base.columns if col not in important_cols]
    for col in other_cols:
        print(f"    {col}")
    
    # Check for missing values in important fields
    print("\n\n--- Missing Value Analysis ---")
    print("Base Paper Dataset:")
    for col in important_cols:
        if col in df_base.columns:
            missing = df_base[col].isna().sum()
            pct = (missing / len(df_base)) * 100
            print(f"  {col}: {missing} missing ({pct:.1f}%)")
    
    print("\nReference 22 Dataset:")
    for col in important_cols:
        if col in df_ref.columns:
            missing = df_ref[col].isna().sum()
            pct = (missing / len(df_ref)) * 100
            print(f"  {col}: {missing} missing ({pct:.1f}%)")
    
    print("\n" + "=" * 70)
    print("DATA READY FOR PROCESSING!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Process weather data to calculate hourly changes")
    print("2. Fit normal distributions for 4 time segments")
    print("3. Download BIXI trip data")
    print("4. Train linear regression model")
    print("5. Generate synthetic trips")

if __name__ == "__main__":
    inspect_weather_data()
