# BIXI Trip Data Download for Synthetic Trip Generator

## Overview
This script downloads trip data from BIXI Montreal's open data portal to support the synthetic trip generator reproduction.

## BIXI Open Data Structure

### Trip Record Fields:
Each trip record typically contains:
- **start_date**: Departure timestamp (YYYY-MM-DD HH:MM:SS)
- **start_station_code**: Origin station ID/code
- **start_station_name**: Origin station name
- **start_station_latitude**: Origin latitude
- **start_station_longitude**: Origin longitude
- **end_date**: Arrival timestamp (YYYY-MM-DD HH:MM:SS)
- **end_station_code**: Destination station ID/code
- **end_station_name**: Destination station name
- **end_station_latitude**: Destination latitude
- **end_station_longitude**: Destination longitude
- **duration_sec**: Trip duration in seconds
- **is_member**: User type (1=member, 0=casual)

### Station Data:
Station information is embedded in trip records:
- Station IDs (codes)
- Station names
- Station locations (lat/long)
- Station capacities (available in separate station info files)

## Time Periods

### Option 1: Base Paper (May-Sep 2019)
- **Period**: May-September 2019
- **Filter**: Weekdays only
- **Expected**: ~100 weekdays of data
- **Typical volume**: ~33,300 trips/day × 100 days = ~3.3M trips

### Option 2: Reference 22 (Jun-Aug 2017-2020)
- **Period**: June-August 2017-2020 (4 summers)
- **Filter**: All days (weekdays + weekends)
- **Expected**: ~368 days of data
- **Typical volume**: ~33,300 trips/day × 368 days = ~12M trips

## Data Source
- **Official Portal**: https://bixi.com/en/open-data-2/
- **Format**: Monthly ZIP files containing CSV data
- **License**: Open data, free to use
- **Updates**: Historical data available from 2014 onwards

## Installation

```bash
cd code
pip install -r requirements.txt
```

## Usage

```bash
python download_bixi_data.py
```

This will:
1. Create `../data/bixi/` directory
2. Download monthly trip data for both time periods
3. Extract and save individual CSV files
4. Create weekday-filtered versions
5. Create combined datasets

## Output Files

```
../data/bixi/
├── bixi_trips_2019_05.csv                           # May 2019
├── bixi_trips_2019_06.csv                           # June 2019
├── bixi_trips_2019_07.csv                           # July 2019
├── bixi_trips_2019_08.csv                           # August 2019
├── bixi_trips_2019_09.csv                           # September 2019
├── bixi_trips_2019_may_sep_combined.csv            # All 5 months
├── bixi_trips_2019_may_sep_weekdays.csv            # Weekdays only
├── bixi_trips_2017_jun_aug_combined.csv            # 2017 summer
├── bixi_trips_2018_jun_aug_combined.csv            # 2018 summer
├── bixi_trips_2019_jun_aug_combined.csv            # 2019 summer
├── bixi_trips_2020_jun_aug_combined.csv            # 2020 summer
└── bixi_trips_2017_2020_jun_aug_combined.csv       # All 4 summers
```

## What We'll Extract from BIXI Data

### For Synthetic Generator (Reference 21):

1. **Trip Volume Scaling**:
   - Real BIXI: ~33,300 trips/day on 617 stations
   - Scale to: ~3,240 trips/day for 60 stations

2. **Trip Pattern Distributions**:
   - Temporal distribution of departures (fit Beta distributions)
   - Morning peak timing (6am-11am)
   - Evening peak timing (4pm-8pm)
   - Random trip timing throughout day

3. **Trip Types Proportions**:
   - Work-related trips (commuting patterns): ~60-80%
   - Random trips: ~20-40%
   - Origin-Destination patterns (inside/outside city centers)

4. **Trip Duration**:
   - Distribution of trip durations
   - Typical range: 5-30 minutes

5. **Station Network Statistics**:
   - Station capacity patterns (docks per station)
   - City center vs. regular station ratios
   - Spatial distribution patterns

### For Linear Regression (Reference 22):

1. **Hourly Demand Aggregation**:
   - Total rentals per hour across all stations
   - Group by: date, hour, weather conditions

2. **Training Data**:
   - Merge with weather data by timestamp
   - Features: temperature, humidity, hour, day of week
   - Target: hourly rental count

## Data Processing Notes

- **Date formats**: May vary across years (handle multiple formats)
- **Column names**: May differ slightly between years
- **Missing values**: Some trips may have incomplete data
- **Station IDs**: Station codes may change over years
- **Coordinates**: Use station lat/long to determine city center locations

## Next Steps After Download

1. **Analyze trip patterns** to fit Beta distributions
2. **Merge with weather data** for regression training
3. **Identify city center stations** using spatial analysis
4. **Calculate trip statistics** for scaling parameters
5. **Generate synthetic station network** (random grid)
6. **Implement trip generator** with Beta sampling
