"""
Bike-Sharing Simulation Environment.

Main simulation environment that orchestrates:
- Station network
- Trip processing
- Rebalancing operations
- Metrics tracking

Based on Base Paper Section 3.2 - Simulation Model
"""

import json
import numpy as np
from datetime import datetime, timedelta

from .station import Station
from .trip_generator import TripGenerator
from .rebalancing_fleet import RebalancingFleet
from .metrics import SimulationMetrics


class BikeShareEnvironment:
    """Bike-sharing simulation environment."""
    
    def __init__(self, network_file, trips_file, num_vehicles=10, vehicle_capacity=20):
        """
        Initialize simulation environment.
        
        Args:
            network_file: Path to station network JSON file
            trips_file: Path to trips CSV file
            num_vehicles: Number of rebalancing vehicles (default: 10)
            vehicle_capacity: Capacity of each vehicle (default: 20 bikes)
        """
        self.network_file = network_file
        self.trips_file = trips_file
        
        # Load network
        self.network_data = self._load_network(network_file)
        self.stations = self._create_stations()
        self.distance_matrix = np.array(self.network_data['distance_matrix'])
        
        # Load trips
        self.trip_generator = TripGenerator(trips_file)
        
        # Initialize fleet
        self.fleet = RebalancingFleet(num_vehicles, vehicle_capacity)
        
        # Metrics
        self.metrics = SimulationMetrics()
        
        # Simulation state
        self.current_time = None
        self.current_day = None
        
        print(f"\n✅ Environment initialized:")
        print(f"   Stations: {len(self.stations)}")
        print(f"   Trips: {len(self.trip_generator):,}")
        print(f"   Vehicles: {num_vehicles} (capacity: {vehicle_capacity})")
    
    def _load_network(self, network_file):
        """Load station network from JSON file."""
        print(f"Loading network from {network_file}...")
        with open(network_file, 'r') as f:
            network = json.load(f)
        print(f"  Loaded {len(network['stations'])} stations")
        return network
    
    def _create_stations(self):
        """Create Station objects from network data."""
        stations = {}
        for station_data in self.network_data['stations']:
            station = Station(
                station_id=station_data['id'],
                capacity=station_data['capacity'],
                latitude=station_data['latitude'],
                longitude=station_data['longitude'],
                is_city_center=station_data['is_city_center']
            )
            stations[station.id] = station
        return stations
    
    def _load_static_inventory(self):
        """
        Load static initial inventory from file.
        
        Returns:
            dict: {station_id: inventory}. If file doesn't exist, returns empty dict.
        """
        from pathlib import Path
        
        # Construct path to static inventory file
        network_path = Path(self.network_file)
        gt_name = network_path.parent.name  # e.g., 'GT1' or 'GT2'
        static_file = network_path.parent / f"{gt_name}_static_initial_inventory.json"
        
        if static_file.exists():
            with open(static_file) as f:
                data = json.load(f)
                # Convert string keys to integers
                initial_inventory = {int(k): v for k, v in data['initial_inventory'].items()}
                print(f"\n✓ Loaded static initial inventory from {static_file.name}")
                print(f"  Total bikes: {sum(initial_inventory.values())}")
                return initial_inventory
        else:
            # Return 50% capacity for all stations if static file doesn't exist
            print(f"\n⚠ Static inventory file not found, using 50% capacity")
            return {sid: s.capacity // 2 for sid, s in self.stations.items()}
    
    def reset(self, day=None, initial_inventory=None):
        """
        Reset environment for a new episode.
        
        Args:
            day: Day number or date string for the episode
            initial_inventory: Dict {station_id: inventory}. If None, loads from
                             static rebalancing solution. If that doesn't exist,
                             defaults to 50% capacity.
        
        Returns:
            dict: Initial state
        """
        # Load initial inventory if not provided
        if initial_inventory is None:
            initial_inventory = self._load_static_inventory()
        
        # Reset all stations with specified inventory
        for station_id, station in self.stations.items():
            if station_id in initial_inventory:
                station.inventory = initial_inventory[station_id]
            else:
                station.reset_inventory()  # Fallback to 50%
            station.reset_metrics()
        
        # Reset fleet
        self.fleet.reset()
        
        # Reset metrics
        self.metrics.reset()
        
        # Set episode day
        self.current_day = day
        
        # Set initial time (7am)
        if day is not None:
            days = self.trip_generator.get_days()
            if isinstance(day, int):
                date_str = days[day - 1]
            else:
                date_str = day
            self.current_time = datetime.strptime(f"{date_str} 07:00:00", "%Y-%m-%d %H:%M:%S")
        
        return self.get_state()
    
    def get_state(self):
        """
        Get current environment state.
        
        Returns:
            dict: Current state including station inventories and time
        """
        return {
            'station_inventories': {sid: s.inventory for sid, s in self.stations.items()},
            'station_occupancies': {sid: s.inventory / s.capacity for sid, s in self.stations.items()},
            'current_time': self.current_time,
            'current_day': self.current_day,
            'num_stations': len(self.stations),
        }
    
    def step(self, day, rebalancing_policy=None):
        """
        Run a complete episode (one day, 7am-11am).
        
        Args:
            day: Day number or date string
            rebalancing_policy: Optional policy function that takes (state, env) and returns actions
        
        Returns:
            tuple: (final_state, metrics, info)
        """
        # Reset for new episode
        self.reset(day)
        
        # Get trips for this episode
        trips = self.trip_generator.get_episode_trips(day)
        
        print(f"\n🎬 Running episode: Day {day}")
        print(f"   Processing {len(trips)} trips (7am-11am)")
        
        # Process trips chronologically
        for trip in trips:
            self._process_trip(trip)
            
            # Optional: Apply rebalancing policy at decision points
            # For now, we'll apply rebalancing at the start (simplified version)
        
        # Apply rebalancing if policy provided
        if rebalancing_policy is not None:
            state = self.get_state()
            actions = rebalancing_policy(state, self)
            if actions:
                bikes_moved = self.fleet.execute_rebalancing(
                    actions, self.stations, self.distance_matrix, self.current_time
                )
                self.metrics.record_rebalancing(bikes_moved, len(actions))
        
        # Record final station states
        self.metrics.record_station_state(self.stations)
        
        # Get final metrics
        final_metrics = self.metrics.get_summary()
        
        print(f"   ✓ Episode complete: {final_metrics['total_lost_demand']} lost demand "
              f"({final_metrics['total_lost_demand_rate']:.2f}%)")
        
        return self.get_state(), final_metrics, {'trips_processed': len(trips)}
    
    def _process_trip(self, trip):
        """
        Process a single trip (rental and return).
        
        Args:
            trip: Trip dictionary
        """
        origin_id = trip['origin_station']
        dest_id = trip['destination_station']
        
        # Process rental at origin
        origin_station = self.stations[origin_id]
        rental_success = origin_station.rent_bike()
        self.metrics.record_rental_attempt(rental_success)
        
        # If rental successful, process return at destination
        if rental_success:
            dest_station = self.stations[dest_id]
            return_success = dest_station.return_bike()
            self.metrics.record_return_attempt(return_success)
        else:
            # Rental failed, no return to process
            pass
    
    def run_multiple_episodes(self, days, rebalancing_policy=None, verbose=True):
        """
        Run simulation for multiple days.
        
        Args:
            days: List of day numbers or date strings
            rebalancing_policy: Optional rebalancing policy
            verbose: Whether to print progress
        
        Returns:
            list: List of metrics dictionaries for each episode
        """
        all_metrics = []
        
        print(f"\n{'='*70}")
        print(f"Running {len(days)} episodes")
        print(f"{'='*70}")
        
        for i, day in enumerate(days, 1):
            if verbose and i % 10 == 0:
                print(f"Progress: {i}/{len(days)} episodes completed")
            
            _, metrics, _ = self.step(day, rebalancing_policy)
            all_metrics.append(metrics)
        
        # Print aggregate statistics
        if verbose:
            self._print_aggregate_stats(all_metrics)
        
        return all_metrics
    
    def _print_aggregate_stats(self, all_metrics):
        """Print aggregate statistics across all episodes."""
        print(f"\n{'='*70}")
        print("AGGREGATE RESULTS")
        print(f"{'='*70}")
        
        avg_lost_rentals = np.mean([m['lost_rental_rate'] for m in all_metrics])
        avg_lost_returns = np.mean([m['lost_return_rate'] for m in all_metrics])
        avg_total_lost = np.mean([m['total_lost_demand_rate'] for m in all_metrics])
        
        total_bikes_moved = sum(m['bikes_rebalanced'] for m in all_metrics)
        total_operations = sum(m['rebalancing_operations'] for m in all_metrics)
        
        print(f"\n📊 Average Lost Demand:")
        print(f"   Lost rentals:     {avg_lost_rentals:.2f}%")
        print(f"   Lost returns:     {avg_lost_returns:.2f}%")
        print(f"   Total lost:       {avg_total_lost:.2f}%")
        
        print(f"\n🚚 Total Rebalancing:")
        print(f"   Bikes moved:      {total_bikes_moved:,}")
        print(f"   Operations:       {total_operations:,}")
        
        print(f"\n{'='*70}\n")
    
    def get_station_info(self, station_id):
        """Get detailed information about a station."""
        station = self.stations[station_id]
        return station.get_state()
    
    def get_all_station_states(self):
        """Get current state of all stations."""
        return {sid: s.get_state() for sid, s in self.stations.items()}
    
    def __repr__(self):
        return (f"BikeShareEnvironment(stations={len(self.stations)}, "
                f"trips={len(self.trip_generator):,})")
