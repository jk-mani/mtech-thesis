"""
Generate synthetic weather data using fitted distributions.

Based on Reference 22 methodology:
- Start with base weather conditions from real data
- For each hour, sample change from fitted normal distributions
- Apply constraints to keep values realistic
- Generate 150 days (100 train + 50 test) for each GT

Uses the fitted distributions from process_weather_data.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Weather constraints (from Reference 22 and Montreal climate)
TEMP_MIN = 5.5   # °C
TEMP_MAX = 36.0  # °C
HUMIDITY_MIN = 15.0  # %
HUMIDITY_MAX = 99.0  # %

def load_distributions():
    """Load fitted weather change distributions"""
    dist_file = Path("../data/synthetic/fitted_parameters/weather_distributions.json")
    
    print(f"Loading fitted distributions from {dist_file}...")
    with open(dist_file, 'r') as f:
        distributions = json.load(f)
    
    print(f"  ✓ Loaded distributions for {len(distributions)} time segments")
    return distributions

def load_base_weather():
    """Load real weather data to sample initial conditions"""
    weather_file = Path("../data/weather/montreal_weather_2017_2020_jun_aug_combined.csv")
    
    print(f"\nLoading base weather data...")
    df = pd.read_csv(weather_file)
    
    # Keep only complete records
    df = df.dropna(subset=['Temp (°C)', 'Rel Hum (%)'])
    
    print(f"  ✓ Loaded {len(df)} hourly records for sampling")
    
    return df

def get_time_segment(hour):
    """Determine which time segment an hour belongs to"""
    if 0 <= hour <= 5:
        return 'segment_0_5'
    elif 6 <= hour <= 11:
        return 'segment_6_11'
    elif 12 <= hour <= 17:
        return 'segment_12_17'
    elif 18 <= hour <= 23:
        return 'segment_18_23'
    else:
        raise ValueError(f"Invalid hour: {hour}")

def sample_weather_change(distributions, hour):
    """Sample temperature and humidity changes for a given hour"""
    segment = get_time_segment(hour)
    segment_dist = distributions[segment]
    
    # Sample temperature change
    temp_change = np.random.normal(
        segment_dist['temperature']['mean'],
        segment_dist['temperature']['std']
    )
    
    # Sample humidity change
    humidity_change = np.random.normal(
        segment_dist['humidity']['mean'],
        segment_dist['humidity']['std']
    )
    
    return temp_change, humidity_change

def generate_day_weather(distributions, start_temp, start_humidity):
    """
    Generate 24 hours of weather for a single day.
    
    Start with initial conditions, apply hourly changes.
    """
    weather_data = []
    
    current_temp = start_temp
    current_humidity = start_humidity
    
    for hour in range(24):
        # Sample changes
        temp_change, humidity_change = sample_weather_change(distributions, hour)
        
        # Apply changes
        current_temp += temp_change
        current_humidity += humidity_change
        
        # Apply constraints
        current_temp = np.clip(current_temp, TEMP_MIN, TEMP_MAX)
        current_humidity = np.clip(current_humidity, HUMIDITY_MIN, HUMIDITY_MAX)
        
        weather_data.append({
            'hour': hour,
            'temperature': current_temp,
            'humidity': current_humidity
        })
    
    return weather_data

def generate_synthetic_weather(distributions, base_weather_df, num_days, start_date):
    """Generate synthetic weather for multiple days"""
    
    all_weather = []
    
    # Sample initial conditions from real data
    initial_sample = base_weather_df.sample(1).iloc[0]
    current_temp = initial_sample['Temp (°C)']
    current_humidity = initial_sample['Rel Hum (%)']
    
    print(f"  Starting conditions: Temp={current_temp:.1f}°C, Humidity={current_humidity:.1f}%")
    
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    for day_num in range(num_days):
        # Generate 24 hours for this day
        day_weather = generate_day_weather(distributions, current_temp, current_humidity)
        
        # Add date information
        for hour_data in day_weather:
            hour_data['date'] = current_date.strftime('%Y-%m-%d')
            hour_data['datetime'] = (current_date + timedelta(hours=hour_data['hour'])).strftime('%Y-%m-%d %H:%M:%S')
            hour_data['day_num'] = day_num + 1
            all_weather.append(hour_data)
        
        # Update starting conditions for next day (use last hour's values)
        current_temp = day_weather[-1]['temperature']
        current_humidity = day_weather[-1]['humidity']
        
        # Move to next day
        current_date += timedelta(days=1)
        
        # Progress update
        if (day_num + 1) % 30 == 0:
            print(f"    Generated {day_num + 1}/{num_days} days...")
    
    return pd.DataFrame(all_weather)

def plot_weather_sample(df, ground_truth, output_dir):
    """Plot sample of generated weather (first 10 days)"""
    
    # Sample first 10 days
    sample_df = df[df['day_num'] <= 10].copy()
    sample_df['datetime'] = pd.to_datetime(sample_df['datetime'])
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Temperature plot
    ax1 = axes[0]
    ax1.plot(sample_df['datetime'], sample_df['temperature'], 'r-', linewidth=1.5)
    ax1.axhline(y=TEMP_MIN, color='gray', linestyle='--', alpha=0.5, label='Constraints')
    ax1.axhline(y=TEMP_MAX, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title(f'{ground_truth} Synthetic Weather - First 10 Days', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Humidity plot
    ax2 = axes[1]
    ax2.plot(sample_df['datetime'], sample_df['humidity'], 'b-', linewidth=1.5)
    ax2.axhline(y=HUMIDITY_MIN, color='gray', linestyle='--', alpha=0.5, label='Constraints')
    ax2.axhline(y=HUMIDITY_MAX, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Humidity (%)', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    output_file = output_dir / f"{ground_truth}_weather_sample.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved sample plot to {output_file}")
    plt.close()

def save_weather_data(df, ground_truth, output_dir):
    """Split and save weather data into train/test sets"""
    
    # Split: first 100 days = train, last 50 days = test
    train_df = df[df['day_num'] <= 100].copy()
    test_df = df[df['day_num'] > 100].copy()
    
    # Select columns for output
    output_cols = ['date', 'datetime', 'hour', 'temperature', 'humidity']
    
    # Save train set
    train_file = output_dir / f"{ground_truth}_weather_train.csv"
    train_df[output_cols].to_csv(train_file, index=False)
    print(f"  ✓ Saved training set: {train_file} ({len(train_df)} records)")
    
    # Save test set
    test_file = output_dir / f"{ground_truth}_weather_test.csv"
    test_df[output_cols].to_csv(test_file, index=False)
    print(f"  ✓ Saved test set: {test_file} ({len(test_df)} records)")
    
    return train_file, test_file

def print_statistics(df, ground_truth):
    """Print statistics for generated weather"""
    print(f"\n  Weather Statistics for {ground_truth}:")
    print(f"    Days: {df['day_num'].max()}")
    print(f"    Hours: {len(df)}")
    print(f"    Temperature:")
    print(f"      Min:  {df['temperature'].min():.2f}°C")
    print(f"      Max:  {df['temperature'].max():.2f}°C")
    print(f"      Mean: {df['temperature'].mean():.2f}°C")
    print(f"      Std:  {df['temperature'].std():.2f}°C")
    print(f"    Humidity:")
    print(f"      Min:  {df['humidity'].min():.2f}%")
    print(f"      Max:  {df['humidity'].max():.2f}%")
    print(f"      Mean: {df['humidity'].mean():.2f}%")
    print(f"      Std:  {df['humidity'].std():.2f}%")

def main():
    """Generate synthetic weather for GT1 and GT2"""
    print("="*70)
    print("SYNTHETIC WEATHER GENERATION")
    print("="*70)
    print("\nObjective: Generate 150 days of weather for GT1 and GT2")
    print("Reference: Paper Reference 22 methodology")
    print("Method: Sample changes from fitted normal distributions\n")
    
    # Load fitted distributions
    distributions = load_distributions()
    
    # Load base weather for initial conditions
    base_weather = load_base_weather()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directories
    gt1_dir = Path("../data/synthetic/GT1")
    gt2_dir = Path("../data/synthetic/GT2")
    gt1_dir.mkdir(parents=True, exist_ok=True)
    gt2_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate GT1 weather
    print("\n" + "="*70)
    print("GENERATING GT1 WEATHER")
    print("="*70)
    
    weather_gt1 = generate_synthetic_weather(
        distributions=distributions,
        base_weather_df=base_weather,
        num_days=150,
        start_date='2019-05-01'
    )
    
    print_statistics(weather_gt1, "GT1")
    
    print("\nSaving GT1 weather data...")
    save_weather_data(weather_gt1, "GT1", gt1_dir)
    plot_weather_sample(weather_gt1, "GT1", gt1_dir)
    
    # Generate GT2 weather (same as GT1 - weather is independent of network)
    print("\n" + "="*70)
    print("GENERATING GT2 WEATHER")
    print("="*70)
    print("  Note: Using same weather as GT1 (weather independent of network)")
    
    # Reset seed to get same weather
    np.random.seed(42)
    weather_gt2 = generate_synthetic_weather(
        distributions=distributions,
        base_weather_df=base_weather,
        num_days=150,
        start_date='2019-05-01'
    )
    
    print_statistics(weather_gt2, "GT2")
    
    print("\nSaving GT2 weather data...")
    save_weather_data(weather_gt2, "GT2", gt2_dir)
    plot_weather_sample(weather_gt2, "GT2", gt2_dir)
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    
    print(f"\n✓ GT1 Weather:")
    print(f"  Train: 100 days × 24 hours = 2,400 records")
    print(f"  Test:  50 days × 24 hours = 1,200 records")
    print(f"  Files: {gt1_dir}/GT1_weather_train.csv")
    print(f"         {gt1_dir}/GT1_weather_test.csv")
    
    print(f"\n✓ GT2 Weather:")
    print(f"  Train: 100 days × 24 hours = 2,400 records")
    print(f"  Test:  50 days × 24 hours = 1,200 records")
    print(f"  Files: {gt2_dir}/GT2_weather_train.csv")
    print(f"         {gt2_dir}/GT2_weather_test.csv")
    
    print("\n✓ Weather characteristics:")
    print(f"  Temperature: {weather_gt1['temperature'].min():.1f}°C to {weather_gt1['temperature'].max():.1f}°C")
    print(f"  Humidity:    {weather_gt1['humidity'].min():.1f}% to {weather_gt1['humidity'].max():.1f}%")
    print(f"  Constraints: Applied (5.5-36°C, 15-99%)")
    
    print("\n✅ Ready for synthetic trip generation!")

if __name__ == "__main__":
    main()
