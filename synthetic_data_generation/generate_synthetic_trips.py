"""
Generate synthetic trip data using demand model, weather, and Beta distributions.

Based on References 21 & 22 methodology:
- Predict daily demand from weather using trained model
- Scale to 60-station network
- Generate individual trips with Beta-distributed departure times
- Assign O-D pairs based on trip type rules
- Calculate arrival times based on duration

This is the final data generation step that brings everything together!
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Beta distribution parameters from Reference 21 (lines 3146-3165)
BETA_PARAMS = {
    'OI_morning': {'alpha': 3, 'beta': 8, 'a': 530, 'b': 340},
    'OI_evening': {'alpha': 3, 'beta': 8, 'a': 530, 'b': 1000},  # Offset for evening
    'OO': {'alpha': 3, 'beta': 7, 'a': 550, 'b': 900},
    'RD': {'alpha': 3, 'beta': 7, 'a': 900, 'b': 560},
    'RN': {'alpha': 6, 'beta': 8, 'a': 1200, 'b': 750},
}

# Trip type proportions (from Reference 21, Table 6)
TRIP_MIX_GT1 = {'OI': 0.32, 'OO': 0.32, 'RD': 0.23, 'RN': 0.13}
TRIP_MIX_GT2 = {'OI': 0.55, 'OO': 0.25, 'RD': 0.15, 'RN': 0.05}

# Work trip probability (from Reference 21)
WORK_TRIP_PROB = 0.85  # 85% chance work trip happens each day

# Scaling factor: 617 stations → 60 stations
SCALING_FACTOR = 60 / 617

# Trip duration range (minutes)
DURATION_MIN = 5
DURATION_MAX = 30

def load_demand_model():
    """Load trained demand prediction model"""
    model_file = Path("../data/synthetic/fitted_parameters/demand_model.pkl")
    
    print(f"Loading demand model from {model_file}...")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    print(f"  ✓ Model loaded")
    return model

def load_weather_data(gt_dir, split='train'):
    """Load synthetic weather data"""
    # Get GT name from directory
    gt_name = gt_dir.name
    weather_file = gt_dir / f"{gt_name}_weather_{split}.csv"
    
    print(f"Loading weather data from {weather_file}...")
    df = pd.read_csv(weather_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"  ✓ Loaded {len(df)} hourly weather records")
    return df

def load_station_network(gt_dir):
    """Load station network"""
    network_file = list(gt_dir.glob("*_station_network.json"))[0]
    
    print(f"Loading station network from {network_file}...")
    with open(network_file, 'r') as f:
        network = json.load(f)
    
    # Extract station info
    stations = network['stations']
    city_center_stations = [s['id'] for s in stations if s['is_city_center']]
    regular_stations = [s['id'] for s in stations if not s['is_city_center']]
    
    print(f"  ✓ Loaded network: {len(stations)} stations")
    print(f"    City center: {len(city_center_stations)} stations")
    print(f"    Regular: {len(regular_stations)} stations")
    
    return stations, city_center_stations, regular_stations

def predict_daily_demand(model, weather_df):
    """Predict total daily trip demand from weather"""
    
    # Aggregate to daily average weather
    daily_weather = weather_df.groupby('date').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'hour': 'mean',  # Will be ~11.5 (middle of day)
    }).reset_index()
    
    # Add weekday
    daily_weather['date'] = pd.to_datetime(daily_weather['date'])
    daily_weather['weekday'] = daily_weather['date'].dt.dayofweek
    
    # Predict hourly demand
    X = daily_weather[['temperature', 'humidity', 'hour', 'weekday']].values
    hourly_demand = model.predict(X)
    
    # Convert to daily demand (24 hours)
    daily_demand = hourly_demand * 24
    
    # Scale from 617 stations to 60 stations
    scaled_demand = daily_demand * SCALING_FACTOR
    
    daily_weather['predicted_daily_trips'] = scaled_demand
    
    return daily_weather[['date', 'predicted_daily_trips']]

def sample_departure_time(trip_type, is_morning=True):
    """
    Sample departure time from Beta distribution.
    Returns minutes from midnight.
    """
    if trip_type == 'OI':
        params = BETA_PARAMS['OI_morning'] if is_morning else BETA_PARAMS['OI_evening']
    elif trip_type == 'OO':
        params = BETA_PARAMS['OO']
    elif trip_type == 'RD':
        params = BETA_PARAMS['RD']
    elif trip_type == 'RN':
        params = BETA_PARAMS['RN']
    else:
        raise ValueError(f"Unknown trip type: {trip_type}")
    
    # Sample from Beta distribution
    x = np.random.beta(params['alpha'], params['beta'])
    
    # Transform to minutes
    minutes = params['a'] * x + params['b']
    
    # Ensure within 24 hours (0-1439 minutes)
    minutes = minutes % 1440
    
    return minutes

def assign_od_pair(trip_type, city_center_stations, regular_stations):
    """Assign origin-destination pair based on trip type"""
    
    if trip_type == 'OI':
        # Outside → Inside (city center)
        origin = np.random.choice(regular_stations)
        destination = np.random.choice(city_center_stations)
    
    elif trip_type == 'OO':
        # Outside → Outside
        origin = np.random.choice(regular_stations)
        destination = np.random.choice(regular_stations)
        # Ensure different stations
        while destination == origin:
            destination = np.random.choice(regular_stations)
    
    elif trip_type in ['RD', 'RN']:
        # Random pairs (any station to any station)
        all_stations = city_center_stations + regular_stations
        origin = np.random.choice(all_stations)
        destination = np.random.choice(all_stations)
        # Ensure different stations
        while destination == origin:
            destination = np.random.choice(all_stations)
    
    else:
        raise ValueError(f"Unknown trip type: {trip_type}")
    
    return origin, destination

def generate_trips_for_day(day_num, date, target_trips, trip_mix, city_center_stations, regular_stations):
    """Generate all trips for a single day"""
    
    trips = []
    trip_id = 1
    
    # Calculate number of trips per type
    num_trips_by_type = {
        trip_type: int(target_trips * proportion)
        for trip_type, proportion in trip_mix.items()
    }
    
    # Generate trips for each type
    for trip_type, num_trips in num_trips_by_type.items():
        
        for _ in range(num_trips):
            # For work trips (OI, OO), decide morning or evening
            if trip_type in ['OI', 'OO']:
                is_morning = np.random.rand() < 0.5  # 50% morning, 50% evening
                
                # Check if trip happens (85% probability)
                if np.random.rand() > WORK_TRIP_PROB:
                    continue  # Skip this trip
            else:
                is_morning = True  # Not applicable for random trips
            
            # Sample departure time
            departure_minutes = sample_departure_time(trip_type, is_morning)
            departure_time = date + timedelta(minutes=float(departure_minutes))
            
            # Assign O-D pair
            origin, destination = assign_od_pair(trip_type, city_center_stations, regular_stations)
            
            # Sample duration
            duration_minutes = np.random.uniform(DURATION_MIN, DURATION_MAX)
            duration_seconds = int(duration_minutes * 60)
            
            # Calculate arrival time
            arrival_time = departure_time + timedelta(seconds=duration_seconds)
            
            # Create trip record
            trip = {
                'trip_id': trip_id,
                'day': day_num,
                'date': date.strftime('%Y-%m-%d'),
                'origin_station': origin,
                'destination_station': destination,
                'departure_time': departure_time.strftime('%Y-%m-%d %H:%M:%S'),
                'arrival_time': arrival_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_sec': duration_seconds,
                'trip_type': trip_type
            }
            
            trips.append(trip)
            trip_id += 1
    
    return trips

def generate_all_trips(demand_df, trip_mix, city_center_stations, regular_stations, start_date):
    """Generate trips for all days"""
    
    all_trips = []
    
    for idx, row in demand_df.iterrows():
        day_num = idx + 1
        # Convert to datetime if it's a string, otherwise use as-is
        if isinstance(row['date'], str):
            date = datetime.strptime(row['date'], '%Y-%m-%d')
        else:
            date = pd.to_datetime(row['date']).to_pydatetime()
        target_trips = int(row['predicted_daily_trips'])
        
        # Generate trips for this day
        day_trips = generate_trips_for_day(
            day_num, date, target_trips, trip_mix,
            city_center_stations, regular_stations
        )
        
        all_trips.extend(day_trips)
        
        # Progress update
        if day_num % 25 == 0:
            print(f"    Generated day {day_num}/{len(demand_df)}: {len(day_trips)} trips")
    
    return pd.DataFrame(all_trips)

def plot_trip_statistics(trips_df, ground_truth, output_dir):
    """Plot statistics of generated trips"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Daily trip counts
    ax1 = axes[0, 0]
    daily_counts = trips_df.groupby('day').size()
    ax1.plot(daily_counts.index, daily_counts.values, 'b-', linewidth=1)
    ax1.set_xlabel('Day', fontsize=11)
    ax1.set_ylabel('Number of Trips', fontsize=11)
    ax1.set_title('Daily Trip Volume', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.axhline(y=daily_counts.mean(), color='r', linestyle='--', alpha=0.7, label=f'Mean: {daily_counts.mean():.0f}')
    ax1.legend()
    
    # Plot 2: Hourly distribution
    ax2 = axes[0, 1]
    trips_df['hour'] = pd.to_datetime(trips_df['departure_time']).dt.hour
    hourly_dist = trips_df['hour'].value_counts().sort_index()
    ax2.bar(hourly_dist.index, hourly_dist.values, color='green', alpha=0.7)
    ax2.set_xlabel('Hour of Day', fontsize=11)
    ax2.set_ylabel('Number of Trips', fontsize=11)
    ax2.set_title('Hourly Departure Distribution', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # Plot 3: Trip type distribution
    ax3 = axes[1, 0]
    type_counts = trips_df['trip_type'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ax3.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('Trip Type Distribution', fontsize=12, fontweight='bold')
    
    # Plot 4: Duration distribution
    ax4 = axes[1, 1]
    durations = trips_df['duration_sec'] / 60  # Convert to minutes
    ax4.hist(durations, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Duration (minutes)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title(f'Trip Duration Distribution\nMean: {durations.mean():.1f} min', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')
    ax4.axvline(x=durations.mean(), color='r', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    
    output_file = output_dir / f"{ground_truth}_trips_statistics.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved statistics plot to {output_file}")
    plt.close()

def save_trips(trips_df, ground_truth, output_dir, split):
    """Save trip data to CSV"""
    
    output_file = output_dir / f"{ground_truth}_trips_{split}.csv"
    
    # Select columns for output
    output_cols = ['trip_id', 'day', 'date', 'origin_station', 'destination_station',
                   'departure_time', 'arrival_time', 'duration_sec', 'trip_type']
    
    trips_df[output_cols].to_csv(output_file, index=False)
    print(f"  ✓ Saved {split} trips: {output_file} ({len(trips_df):,} trips)")
    
    return output_file

def print_trip_statistics(trips_df, ground_truth, split):
    """Print statistics about generated trips"""
    
    print(f"\n  Trip Statistics for {ground_truth} ({split}):")
    print(f"    Total trips: {len(trips_df):,}")
    print(f"    Days: {trips_df['day'].nunique()}")
    print(f"    Avg trips/day: {len(trips_df) / trips_df['day'].nunique():.0f}")
    
    print(f"\n    By trip type:")
    for trip_type, count in trips_df['trip_type'].value_counts().items():
        pct = (count / len(trips_df)) * 100
        print(f"      {trip_type}: {count:,} ({pct:.1f}%)")
    
    durations = trips_df['duration_sec'] / 60
    print(f"\n    Duration (minutes):")
    print(f"      Min: {durations.min():.1f}")
    print(f"      Max: {durations.max():.1f}")
    print(f"      Mean: {durations.mean():.1f}")
    print(f"      Median: {durations.median():.1f}")

def generate_for_ground_truth(ground_truth, trip_mix):
    """Generate synthetic trips for one ground truth"""
    
    print(f"\n{'='*70}")
    print(f"GENERATING {ground_truth} TRIPS")
    print(f"{'='*70}")
    
    gt_dir = Path(f"../data/synthetic/{ground_truth}")
    
    # Load components
    demand_model = load_demand_model()
    stations, city_center_stations, regular_stations = load_station_network(gt_dir)
    
    # Generate for training set
    print(f"\nGenerating training set trips...")
    weather_train = load_weather_data(gt_dir, 'train')
    demand_train = predict_daily_demand(demand_model, weather_train)
    
    print(f"\n  Predicted daily demand: {demand_train['predicted_daily_trips'].mean():.0f} trips/day (mean)")
    print(f"  Generating trips for {len(demand_train)} days...")
    
    trips_train = generate_all_trips(
        demand_train, trip_mix, city_center_stations, regular_stations,
        start_date='2019-05-01'
    )
    
    print_trip_statistics(trips_train, ground_truth, 'train')
    save_trips(trips_train, ground_truth, gt_dir, 'train')
    
    # Generate for test set
    print(f"\nGenerating test set trips...")
    weather_test = load_weather_data(gt_dir, 'test')
    demand_test = predict_daily_demand(demand_model, weather_test)
    
    print(f"\n  Predicted daily demand: {demand_test['predicted_daily_trips'].mean():.0f} trips/day (mean)")
    print(f"  Generating trips for {len(demand_test)} days...")
    
    trips_test = generate_all_trips(
        demand_test, trip_mix, city_center_stations, regular_stations,
        start_date='2019-08-10'  # Continuing from training end
    )
    
    print_trip_statistics(trips_test, ground_truth, 'test')
    save_trips(trips_test, ground_truth, gt_dir, 'test')
    
    # Plot combined statistics
    print(f"\nGenerating statistics plots...")
    trips_combined = pd.concat([trips_train, trips_test], ignore_index=True)
    plot_trip_statistics(trips_combined, ground_truth, gt_dir)
    
    return trips_train, trips_test

def main():
    """Generate synthetic trips for GT1 and GT2"""
    
    print("="*70)
    print("SYNTHETIC TRIP GENERATION")
    print("="*70)
    print("\nObjective: Generate ~486,000 synthetic trips")
    print("Reference: Papers Reference 21 & 22 methodology")
    print("Method: Weather → Demand → Beta sampling → O-D assignment\n")
    
    # Set random seed
    np.random.seed(42)
    
    # Generate GT1
    trips_gt1_train, trips_gt1_test = generate_for_ground_truth("GT1", TRIP_MIX_GT1)
    
    # Reset seed for GT2 (but will have different trips due to different network)
    np.random.seed(42)
    
    # Generate GT2
    trips_gt2_train, trips_gt2_test = generate_for_ground_truth("GT2", TRIP_MIX_GT2)
    
    # Final summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    
    print(f"\n✓ GT1 Trips Generated:")
    print(f"  Train: {len(trips_gt1_train):,} trips (100 days)")
    print(f"  Test:  {len(trips_gt1_test):,} trips (50 days)")
    print(f"  Total: {len(trips_gt1_train) + len(trips_gt1_test):,} trips")
    print(f"  Mix: OI=32%, OO=32%, RD=23%, RN=13%")
    
    print(f"\n✓ GT2 Trips Generated:")
    print(f"  Train: {len(trips_gt2_train):,} trips (100 days)")
    print(f"  Test:  {len(trips_gt2_test):,} trips (50 days)")
    print(f"  Total: {len(trips_gt2_train) + len(trips_gt2_test):,} trips")
    print(f"  Mix: OI=55%, OO=25%, RD=15%, RN=5%")
    
    print("\n✅ Synthetic data generation complete!")
    print("✅ Ready for validation and use in RL training!")

if __name__ == "__main__":
    main()
