# Weather Data Download for Synthetic Trip Generator

## Overview
This script downloads hourly weather data from Government of Canada's climate database for Montreal to support the synthetic trip generator reproduction.

## Required Fields
According to the papers, we need:

### Essential Fields:
1. **Temperature** (°C) - Temp (°C)
2. **Humidity** (%) - Rel Hum (%)
3. **Date/Time** - Date/Time (LST)
4. **Hour** - Extracted from Date/Time
5. **Day of Week** - Calculated from Date/Time
6. **Year** - Extracted from Date/Time

### Available in Environment Canada Data:
- Date/Time (LST) - Local Standard Time
- Year, Month, Day, Time (LST)
- Temp (°C) - Temperature
- Dew Point Temp (°C)
- Rel Hum (%) - Relative Humidity
- Wind Dir (10s deg)
- Wind Spd (km/h)
- Visibility (km)
- Stn Press (kPa)
- Weather conditions

## Time Periods

### Option 1: Base Paper (150 days)
- **Period**: May-September 2019
- **Filter**: Weekdays only
- **Usage**: Generate 150 days (100 train + 50 test)

### Option 2: Reference 22 (368 days)
- **Period**: June-August 2017-2020 (4 summers)
- **Filter**: Both weekdays and weekends
- **Usage**: Generate 500 days (250 train + 100 val + 150 test)

## Station Information
- **Station**: Montreal/Pierre Elliott Trudeau International Airport
- **Station ID**: 51157 (for hourly data)
- **Coordinates**: 45.47°N, 73.74°W
- **Elevation**: 36 m

## Installation

```bash
cd code
pip install -r requirements.txt
```

## Usage

```bash
python download_weather_data.py
```

This will:
1. Create `../data/weather/` directory
2. Download hourly data for both time periods
3. Save individual files per period
4. Create weekday-filtered versions
5. Create combined datasets

## Output Files

```
../data/weather/
├── montreal_weather_2019_may_sep.csv              # All days
├── montreal_weather_2019_may_sep_weekdays.csv     # Weekdays only
├── montreal_weather_2017_jun_aug.csv              # 2017 summer
├── montreal_weather_2018_jun_aug.csv              # 2018 summer
├── montreal_weather_2019_jun_aug.csv              # 2019 summer
├── montreal_weather_2020_jun_aug.csv              # 2020 summer
└── montreal_weather_2017_2020_jun_aug_combined.csv # All 4 summers
```

## Data Processing Steps (from Reference 22)

1. **Time Segmentation**: Divide each day into 4 segments
   - 0am-5am
   - 6am-11am
   - 12pm-5pm
   - 6pm-11pm

2. **Calculate Hourly Changes**: For each segment, compute temperature and humidity differences between consecutive hours

3. **Fit Distributions**: Fit normal distributions to hourly changes per segment

4. **Generate Synthetic Weather**: Sample from distributions to create new weather scenarios

## Notes

- Data is publicly available from Environment Canada
- Download is rate-limited (1 second between requests)
- Missing values may occur and need handling
- Data is in Local Standard Time (LST)
