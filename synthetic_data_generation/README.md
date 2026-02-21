# Synthetic Data Generation

This folder contains all scripts and documentation for generating synthetic bike-sharing data (GT1 and GT2 datasets) based on References 21 and 22.

## 📁 Contents

### Python Scripts (9 files)

#### Phase 1: Real Data Download
1. **`download_weather_data.py`** - Downloads Montreal weather data from Environment Canada
2. **`download_bixi_data.py`** - Downloads BIXI Montreal trip data
3. **`inspect_weather_data.py`** - Inspects downloaded weather data
4. **`inspect_bixi_data.py`** - Inspects downloaded BIXI trip data

#### Phase 2: Data Processing & Model Training
5. **`process_weather_data.py`** - Fits normal distributions to hourly weather changes
6. **`train_demand_model.py`** - Trains linear regression model (weather → demand)

#### Phase 3: Synthetic Data Generation
7. **`generate_station_network.py`** - Generates GT1 & GT2 station networks
8. **`generate_synthetic_weather.py`** - Generates 150 days of synthetic weather
9. **`generate_synthetic_trips.py`** - Generates 1.1M+ synthetic trips

### Documentation (5 files)

1. **`README_weather.md`** - Weather data download documentation
2. **`README_bixi.md`** - BIXI trip data download documentation
3. **`DATA_DOWNLOAD_SUMMARY.md`** - Summary of downloaded real-world data
4. **`PROGRESS_LOG.md`** - Detailed progress log of generation process
5. **`SYNTHETIC_DATA_COMPLETE.md`** - Final summary and usage guide

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r ../requirements.txt
```

### Run in Order

```bash
# Phase 1: Download real data
python download_weather_data.py
python download_bixi_data.py

# Phase 2: Process and train
python process_weather_data.py
python train_demand_model.py

# Phase 3: Generate synthetic data
python generate_station_network.py
python generate_synthetic_weather.py
python generate_synthetic_trips.py
```

## 📊 Output

All generated data is saved to:
```
../data/synthetic/
├── fitted_parameters/  # Models and distributions
├── GT1/               # Ground truth 1 (1 city center)
└── GT2/               # Ground truth 2 (2 city centers)
```

## ✅ Status

**COMPLETE** - All synthetic data generated successfully!
- GT1: 560,577 trips
- GT2: 545,916 trips
- Total: 1,106,493 trips

## 📚 References

Based on methodology from:
- **Reference 21:** Instance generator for bike-sharing rebalancing
- **Reference 22:** Weather-dependent synthetic data generation

## 🎯 Purpose

This synthetic data is used to reproduce results from the base paper on Reinforcement Learning for dynamic bike rebalancing. The data generation phase is now complete, and future work focuses on implementing the RL algorithm using this synthetic data.

---

**Generated:** Nov 26, 2025  
**Status:** ✅ Production-ready
