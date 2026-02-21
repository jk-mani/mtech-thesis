"""
Generate synthetic station networks for GT1 and GT2.

Based on Reference 21 methodology:
- 150×150 grid covering Montreal-like area
- GT1: 60 stations, 1 city center (9 stations)
- GT2: 60 stations, 2 city centers (12 stations)
- Random placement within defined regions
- Calculate distances using Haversine formula

Reference 21 specifications (lines 2914-2993):
- Grid area: lat 45.4-45.65, long -73.71 to -73.49
- City center stations: 40 docks
- Regular stations: 20 docks
"""

import numpy as np
import json
from pathlib import Path
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt

# Grid specifications (from Reference 21, lines 2920-2922)
GRID_SIZE = 150  # 150×150 grids
LAT_MIN, LAT_MAX = 45.4, 45.65
LON_MIN, LON_MAX = -73.71, -73.49

# Station specifications (from Reference 21)
NUM_STATIONS = 60
CAPACITY_CITY_CENTER = 40  # docks
CAPACITY_REGULAR = 20  # docks

# City center placement ranges (from Reference 21, lines 2932-2940)
# For 1 center: grids 53-98 on both axes
# For 2 centers: one in 30-75, another in 75-120
CC1_RANGE = (53, 98)  # Single city center
CC2_RANGE_1 = (30, 75)  # First center of two
CC2_RANGE_2 = (75, 120)  # Second center of two

# City center size (number of grids around central point)
CC_RADIUS = 10  # grids around center

def grid_to_latlon(grid_x, grid_y):
    """Convert grid coordinates to latitude/longitude"""
    lat = LAT_MIN + (grid_y / GRID_SIZE) * (LAT_MAX - LAT_MIN)
    lon = LON_MIN + (grid_x / GRID_SIZE) * (LON_MAX - LON_MIN)
    return lat, lon

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great circle distance between two points on Earth.
    Returns distance in kilometers.
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r

def generate_city_center_locations(num_centers):
    """
    Generate city center locations based on ground truth type.
    
    Reference 21, lines 2932-2940
    """
    centers = []
    
    if num_centers == 1:
        # Single city center: random in grids 53-98
        center_x = np.random.randint(CC1_RANGE[0], CC1_RANGE[1])
        center_y = np.random.randint(CC1_RANGE[0], CC1_RANGE[1])
        centers.append({'x': center_x, 'y': center_y})
        
    elif num_centers == 2:
        # Two city centers: one in 30-75, another in 75-120
        center1_x = np.random.randint(CC2_RANGE_1[0], CC2_RANGE_1[1])
        center1_y = np.random.randint(CC2_RANGE_1[0], CC2_RANGE_1[1])
        centers.append({'x': center1_x, 'y': center1_y})
        
        center2_x = np.random.randint(CC2_RANGE_2[0], CC2_RANGE_2[1])
        center2_y = np.random.randint(CC2_RANGE_2[0], CC2_RANGE_2[1])
        centers.append({'x': center2_x, 'y': center2_y})
    
    return centers

def is_in_city_center(grid_x, grid_y, centers):
    """Check if a grid location is within any city center"""
    for center in centers:
        dx = abs(grid_x - center['x'])
        dy = abs(grid_y - center['y'])
        if dx <= CC_RADIUS and dy <= CC_RADIUS:
            return True
    return False

def place_stations(num_city_center_stations, num_regular_stations, centers):
    """
    Place stations on the grid.
    
    City center stations are placed within city center areas.
    Regular stations are placed outside city center areas.
    """
    stations = []
    occupied_grids = set()
    
    station_id = 1
    
    # Place city center stations
    print(f"  Placing {num_city_center_stations} city center stations...")
    attempts = 0
    while len([s for s in stations if s['is_city_center']]) < num_city_center_stations:
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("Could not place all city center stations")
        
        # Choose a random city center
        center = centers[np.random.randint(0, len(centers))]
        
        # Random position within city center
        grid_x = center['x'] + np.random.randint(-CC_RADIUS, CC_RADIUS+1)
        grid_y = center['y'] + np.random.randint(-CC_RADIUS, CC_RADIUS+1)
        
        # Ensure within bounds and not occupied
        if (0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE and
            (grid_x, grid_y) not in occupied_grids):
            
            lat, lon = grid_to_latlon(grid_x, grid_y)
            
            station = {
                'id': station_id,
                'grid_x': int(grid_x),
                'grid_y': int(grid_y),
                'latitude': float(lat),
                'longitude': float(lon),
                'capacity': CAPACITY_CITY_CENTER,
                'is_city_center': True
            }
            
            stations.append(station)
            occupied_grids.add((grid_x, grid_y))
            station_id += 1
    
    # Place regular stations
    print(f"  Placing {num_regular_stations} regular stations...")
    attempts = 0
    while len([s for s in stations if not s['is_city_center']]) < num_regular_stations:
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("Could not place all regular stations")
        
        # Random position anywhere on grid
        grid_x = np.random.randint(0, GRID_SIZE)
        grid_y = np.random.randint(0, GRID_SIZE)
        
        # Must be outside city centers and not occupied
        if ((grid_x, grid_y) not in occupied_grids and
            not is_in_city_center(grid_x, grid_y, centers)):
            
            lat, lon = grid_to_latlon(grid_x, grid_y)
            
            station = {
                'id': station_id,
                'grid_x': int(grid_x),
                'grid_y': int(grid_y),
                'latitude': float(lat),
                'longitude': float(lon),
                'capacity': CAPACITY_REGULAR,
                'is_city_center': False
            }
            
            stations.append(station)
            occupied_grids.add((grid_x, grid_y))
            station_id += 1
    
    return stations

def calculate_distance_matrix(stations):
    """Calculate pairwise distances between all stations"""
    n = len(stations)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i, j] = 0.0
            else:
                dist = haversine_distance(
                    stations[i]['latitude'], stations[i]['longitude'],
                    stations[j]['latitude'], stations[j]['longitude']
                )
                distances[i, j] = dist
    
    return distances.tolist()

def plot_network(stations, centers, ground_truth, output_dir):
    """Visualize the station network"""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot city center regions
    for center in centers:
        rect = plt.Rectangle(
            (center['x'] - CC_RADIUS, center['y'] - CC_RADIUS),
            2 * CC_RADIUS, 2 * CC_RADIUS,
            fill=False, edgecolor='red', linewidth=2, linestyle='--',
            label='City Center Area' if center == centers[0] else ''
        )
        ax.add_patch(rect)
        
        # Mark center point
        ax.plot(center['x'], center['y'], 'r*', markersize=20, 
                label='City Center' if center == centers[0] else '')
    
    # Plot stations
    city_center_stations = [s for s in stations if s['is_city_center']]
    regular_stations = [s for s in stations if not s['is_city_center']]
    
    if city_center_stations:
        cc_x = [s['grid_x'] for s in city_center_stations]
        cc_y = [s['grid_y'] for s in city_center_stations]
        ax.scatter(cc_x, cc_y, c='blue', s=100, marker='o', 
                  label=f'City Center Stations ({len(city_center_stations)})', zorder=3)
    
    if regular_stations:
        reg_x = [s['grid_x'] for s in regular_stations]
        reg_y = [s['grid_y'] for s in regular_stations]
        ax.scatter(reg_x, reg_y, c='green', s=50, marker='s',
                  label=f'Regular Stations ({len(regular_stations)})', zorder=2)
    
    ax.set_xlim(-5, GRID_SIZE + 5)
    ax.set_ylim(-5, GRID_SIZE + 5)
    ax.set_xlabel('Grid X', fontsize=12)
    ax.set_ylabel('Grid Y', fontsize=12)
    ax.set_title(f'{ground_truth} Station Network\n{len(stations)} stations, {len(centers)} city center(s)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    
    output_file = output_dir / f"{ground_truth}_network_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization to {output_file}")
    plt.close()

def save_network(stations, distance_matrix, centers, ground_truth, output_dir):
    """Save network to JSON file"""
    network = {
        'ground_truth': ground_truth,
        'num_stations': len(stations),
        'num_city_centers': len(centers),
        'city_center_locations': centers,
        'grid_size': GRID_SIZE,
        'area_bounds': {
            'lat_min': LAT_MIN, 'lat_max': LAT_MAX,
            'lon_min': LON_MIN, 'lon_max': LON_MAX
        },
        'stations': stations,
        'distance_matrix': distance_matrix,
        'total_capacity': sum(s['capacity'] for s in stations),
        'city_center_capacity': sum(s['capacity'] for s in stations if s['is_city_center']),
        'regular_capacity': sum(s['capacity'] for s in stations if not s['is_city_center'])
    }
    
    output_file = output_dir / f"{ground_truth}_station_network.json"
    with open(output_file, 'w') as f:
        json.dump(network, f, indent=2)
    
    print(f"  ✓ Saved network to {output_file}")
    return output_file

def generate_network(ground_truth, num_centers, num_cc_stations):
    """Generate a complete station network"""
    print(f"\n{'='*70}")
    print(f"GENERATING {ground_truth} NETWORK")
    print(f"{'='*70}")
    
    # Calculate number of stations
    num_regular_stations = NUM_STATIONS - num_cc_stations
    
    print(f"\nConfiguration:")
    print(f"  Total stations: {NUM_STATIONS}")
    print(f"  City centers: {num_centers}")
    print(f"  City center stations: {num_cc_stations} ({CAPACITY_CITY_CENTER} docks each)")
    print(f"  Regular stations: {num_regular_stations} ({CAPACITY_REGULAR} docks each)")
    print(f"  Total capacity: {num_cc_stations * CAPACITY_CITY_CENTER + num_regular_stations * CAPACITY_REGULAR} docks")
    
    # Generate city center locations
    print(f"\nGenerating city center locations...")
    centers = generate_city_center_locations(num_centers)
    for i, center in enumerate(centers, 1):
        lat, lon = grid_to_latlon(center['x'], center['y'])
        print(f"  City Center {i}: Grid ({center['x']}, {center['y']}) = ({lat:.4f}°, {lon:.4f}°)")
    
    # Place stations
    print(f"\nPlacing stations on grid...")
    stations = place_stations(num_cc_stations, num_regular_stations, centers)
    print(f"  ✓ Placed {len(stations)} stations")
    
    # Calculate distances
    print(f"\nCalculating distance matrix...")
    distance_matrix = calculate_distance_matrix(stations)
    
    # Statistics
    distances_flat = [d for row in distance_matrix for d in row if d > 0]
    print(f"  ✓ Calculated {len(stations)}×{len(stations)} distances")
    print(f"  Distance statistics:")
    print(f"    Min: {min(distances_flat):.3f} km")
    print(f"    Max: {max(distances_flat):.3f} km")
    print(f"    Mean: {np.mean(distances_flat):.3f} km")
    
    return stations, distance_matrix, centers

def main():
    """Generate both GT1 and GT2 networks"""
    print("="*70)
    print("STATION NETWORK GENERATION")
    print("="*70)
    print("\nObjective: Generate synthetic 60-station networks")
    print("Reference: Paper Reference 21 methodology\n")
    
    # Create output directories
    output_dir = Path("../data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gt1_dir = output_dir / "GT1"
    gt2_dir = output_dir / "GT2"
    gt1_dir.mkdir(exist_ok=True)
    gt2_dir.mkdir(exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate GT1 (1 city center, 9 CC stations)
    stations_gt1, distances_gt1, centers_gt1 = generate_network(
        ground_truth="GT1",
        num_centers=1,
        num_cc_stations=9
    )
    
    # Save GT1
    print(f"\nSaving GT1 network...")
    save_network(stations_gt1, distances_gt1, centers_gt1, "GT1", gt1_dir)
    plot_network(stations_gt1, centers_gt1, "GT1", gt1_dir)
    
    # Generate GT2 (2 city centers, 12 CC stations)
    stations_gt2, distances_gt2, centers_gt2 = generate_network(
        ground_truth="GT2",
        num_centers=2,
        num_cc_stations=12
    )
    
    # Save GT2
    print(f"\nSaving GT2 network...")
    save_network(stations_gt2, distances_gt2, centers_gt2, "GT2", gt2_dir)
    plot_network(stations_gt2, centers_gt2, "GT2", gt2_dir)
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    
    print(f"\n✓ GT1 Network:")
    print(f"  Stations: {len(stations_gt1)}")
    print(f"  City centers: {len(centers_gt1)}")
    print(f"  City center stations: {len([s for s in stations_gt1 if s['is_city_center']])}")
    print(f"  Files: {gt1_dir}/GT1_station_network.json")
    
    print(f"\n✓ GT2 Network:")
    print(f"  Stations: {len(stations_gt2)}")
    print(f"  City centers: {len(centers_gt2)}")
    print(f"  City center stations: {len([s for s in stations_gt2 if s['is_city_center']])}")
    print(f"  Files: {gt2_dir}/GT2_station_network.json")
    
    print("\n✅ Ready for synthetic data generation!")

if __name__ == "__main__":
    main()
