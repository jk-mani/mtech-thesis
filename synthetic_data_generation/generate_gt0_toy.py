"""
Generate GT0 Toy Model - Simplified bike-sharing system for faster DQN learning.

Configuration:
- 10 stations (vs 60 in GT1/GT2)
- Single city center cluster
- 15 dock capacity per station
- ~50-100 trips per 4-hour horizon
- Designed for 2 vehicles with 15 bike capacity
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt

# GT0 Configuration
NUM_STATIONS = 10
DOCK_CAPACITY = 15  # bikes per station
GRID_SIZE = 50  # Smaller grid for toy model

# Smaller area bounds (subset of Montreal)
LAT_MIN, LAT_MAX = 45.48, 45.52
LON_MIN, LON_MAX = -73.60, -73.55

# Trip generation parameters
TRIPS_PER_HOUR_MEAN = 10  # ~40 trips per 4-hour horizon (reduced for easier learning)
HORIZON_START = 7  # 7 AM
HORIZON_END = 11   # 11 AM

# Training/test periods
TRAIN_DAYS = 100
TEST_DAYS = 50

np.random.seed(42)

def grid_to_latlon(grid_x, grid_y):
    """Convert grid coordinates to latitude/longitude"""
    lat = LAT_MIN + (grid_y / GRID_SIZE) * (LAT_MAX - LAT_MIN)
    lon = LON_MIN + (grid_x / GRID_SIZE) * (LON_MAX - LON_MIN)
    return lat, lon

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points in km."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km
    return c * r

def generate_station_network():
    """Generate 10 stations in a single cluster."""
    print("Generating GT0 station network...")
    
    # Place stations in a compact cluster around center
    center_x, center_y = GRID_SIZE // 2, GRID_SIZE // 2
    
    stations = []
    station_positions = set()
    
    # Generate stations in a tight cluster
    attempts = 0
    while len(stations) < NUM_STATIONS and attempts < 1000:
        # Random offset from center (within ~5 grid units)
        offset_x = np.random.randint(-7, 8)
        offset_y = np.random.randint(-7, 8)
        
        grid_x = center_x + offset_x
        grid_y = center_y + offset_y
        
        # Ensure unique position
        pos_key = (grid_x, grid_y)
        if pos_key in station_positions:
            attempts += 1
            continue
        
        station_positions.add(pos_key)
        lat, lon = grid_to_latlon(grid_x, grid_y)
        
        stations.append({
            'id': len(stations) + 1,
            'grid_x': grid_x,
            'grid_y': grid_y,
            'latitude': lat,
            'longitude': lon,
            'capacity': DOCK_CAPACITY,
            'is_city_center': True  # All in single cluster
        })
        attempts += 1
    
    # Calculate distance matrix
    n = len(stations)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = haversine_distance(
                    stations[i]['latitude'], stations[i]['longitude'],
                    stations[j]['latitude'], stations[j]['longitude']
                )
    
    network = {
        'ground_truth': 'GT0',
        'description': 'Toy model - 10 stations, single cluster',
        'num_stations': NUM_STATIONS,
        'num_city_centers': 1,
        'city_center_locations': [{'x': center_x, 'y': center_y}],
        'grid_size': GRID_SIZE,
        'area_bounds': {
            'lat_min': LAT_MIN,
            'lat_max': LAT_MAX,
            'lon_min': LON_MIN,
            'lon_max': LON_MAX
        },
        'stations': stations,
        'distance_matrix': distances.tolist()
    }
    
    print(f"  Created {len(stations)} stations")
    print(f"  Dock capacity: {DOCK_CAPACITY} bikes each")
    print(f"  Total system capacity: {NUM_STATIONS * DOCK_CAPACITY} bikes")
    
    return network

def generate_trips(network, num_days, start_date, split_name):
    """Generate synthetic trips for the toy model."""
    print(f"Generating {split_name} trips ({num_days} days)...")
    
    trips = []
    trip_id = 1
    
    stations = network['stations']
    num_stations = len(stations)
    
    for day in range(1, num_days + 1):
        current_date = start_date + timedelta(days=day-1)
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Generate trips for 4-hour horizon (7 AM - 11 AM)
        for hour in range(HORIZON_START, HORIZON_END):
            # Random number of trips this hour (Poisson)
            num_trips_hour = np.random.poisson(TRIPS_PER_HOUR_MEAN)
            
            for _ in range(num_trips_hour):
                # Random origin and destination
                origin = np.random.randint(1, num_stations + 1)
                dest = np.random.randint(1, num_stations + 1)
                
                # Allow same station (short trips)
                # but bias toward different stations
                if origin == dest and np.random.random() < 0.7:
                    dest = np.random.randint(1, num_stations + 1)
                
                # Random time within the hour
                minute = np.random.randint(0, 60)
                second = np.random.randint(0, 60)
                
                dep_time = current_date.replace(hour=hour, minute=minute, second=second)
                
                # Trip duration: 5-30 minutes
                duration_sec = np.random.randint(300, 1800)
                arr_time = dep_time + timedelta(seconds=duration_sec)
                
                # Trip type (simplified)
                trip_type = np.random.choice(['II', 'IO', 'OI', 'OO'], p=[0.4, 0.2, 0.2, 0.2])
                
                trips.append({
                    'trip_id': trip_id,
                    'day': day,
                    'date': date_str,
                    'origin_station': origin,
                    'destination_station': dest,
                    'departure_time': dep_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'arrival_time': arr_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_sec': duration_sec,
                    'trip_type': trip_type
                })
                trip_id += 1
    
    df = pd.DataFrame(trips)
    
    # Statistics
    trips_per_day = len(df) / num_days
    print(f"  Total trips: {len(df)}")
    print(f"  Avg trips/day: {trips_per_day:.1f}")
    print(f"  Avg trips/horizon: {trips_per_day:.1f}")
    
    return df

def main():
    print("="*60)
    print("GENERATING GT0 TOY MODEL")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Stations: {NUM_STATIONS}")
    print(f"  Dock capacity: {DOCK_CAPACITY}")
    print(f"  Horizon: {HORIZON_START}:00 - {HORIZON_END}:00")
    print(f"  Expected trips/hour: {TRIPS_PER_HOUR_MEAN}")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / 'GT0'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate network
    network = generate_station_network()
    
    # Save network
    network_path = output_dir / 'GT0_station_network.json'
    with open(network_path, 'w') as f:
        json.dump(network, f, indent=2)
    print(f"\n✓ Network saved to {network_path}")
    
    # Generate trips
    train_start = datetime(2019, 5, 1)
    test_start = datetime(2019, 8, 9)
    
    train_trips = generate_trips(network, TRAIN_DAYS, train_start, 'training')
    test_trips = generate_trips(network, TEST_DAYS, test_start, 'test')
    
    # Save trips
    train_path = output_dir / 'GT0_trips_train.csv'
    test_path = output_dir / 'GT0_trips_test.csv'
    
    train_trips.to_csv(train_path, index=False)
    test_trips.to_csv(test_path, index=False)
    
    print(f"\n✓ Training trips saved to {train_path}")
    print(f"✓ Test trips saved to {test_path}")
    
    print("\n" + "="*60)
    print("GT0 TOY MODEL GENERATION COMPLETE")
    print("="*60)
    print(f"\nRecommended DQN settings:")
    print(f"  --num-stations 10")
    print(f"  --num-vehicles 2")
    print(f"  --vehicle-capacity 15")

if __name__ == "__main__":
    main()
