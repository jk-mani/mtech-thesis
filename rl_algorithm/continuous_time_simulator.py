"""
Continuous Time Event-Driven Simulator.

Based on Base Paper Section 4.2 (Algorithm 1).

Key features:
- Event-driven (not time-slotted)
- Vehicles act asynchronously upon arrival
- Processes customer trips and vehicle movements
- Calculates rewards between decision epochs
"""

import heapq
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from copy import deepcopy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.station import Station
from .vehicle import Vehicle
from .events import Event, VehicleArrival, CustomerRental, CustomerReturn
from .events import create_rental_event, create_return_event, create_vehicle_arrival_event

# Fixed seed for reproducibility
np.random.seed(42)


class ContinuousTimeSimulator:
    """
    Continuous-time event-driven simulator for multi-agent rebalancing.
    
    Based on Base Paper Algorithm 1.
    """
    
    def __init__(self, network_file, trips_file, num_vehicles=4, vehicle_capacity=40, fill_levels=None):
        """
        Initialize simulator.
        
        Args:
            network_file: Path to station network JSON
            trips_file: Path to trips CSV  
            num_vehicles: Number of rebalancing vehicles
            vehicle_capacity: Capacity of each vehicle
            fill_levels: List of fill levels for rebalancing (default: [0.10, 0.50, 0.90])
        """
        # Store fill levels (used in _rebalance_station)
        self.fill_levels = fill_levels if fill_levels is not None else [0.10, 0.50, 0.90]
        # Load network
        with open(network_file) as f:
            self.network_data = json.load(f)
        
        self.stations = self._create_stations()
        self.distance_matrix = np.array(self.network_data['distance_matrix'])
        
        # Load trips
        self.trips_df = pd.read_csv(trips_file)
        self.trips_df['departure_time'] = pd.to_datetime(self.trips_df['departure_time'])
        self.trips_df['arrival_time'] = pd.to_datetime(self.trips_df['arrival_time'])
        
        # Create vehicles
        self.num_vehicles = num_vehicles
        self.vehicles = {i: Vehicle(i, vehicle_capacity) for i in range(1, num_vehicles + 1)}
        
        # Event queue (priority queue)
        self.event_queue = []
        
        # Simulation state
        self.current_time = None
        self.episode_start_time = None
        self.episode_end_time = None
        
        # Metrics
        self.total_lost_rentals = 0
        self.total_lost_returns = 0
        self.total_successful_rentals = 0
        self.total_successful_returns = 0
        
        # For reward calculation
        self.last_decision_time = None
        self.lost_demand_since_last_decision = 0
        
        print(f"ContinuousTimeSimulator initialized:")
        print(f"  Stations: {len(self.stations)}")
        print(f"  Vehicles: {num_vehicles}")
        print(f"  Trips: {len(self.trips_df):,}")
    
    def _create_stations(self):
        """Create station objects from network data."""
        stations = {}
        for s_data in self.network_data['stations']:
            station = Station(
                station_id=s_data['id'],
                capacity=s_data['capacity'],
                latitude=s_data['latitude'],
                longitude=s_data['longitude'],
                is_city_center=s_data['is_city_center']
            )
            stations[station.id] = station
        return stations
    
    def reset(self, day, start_hour=7, end_hour=11):
        """
        Reset simulator for new episode.
        
        Args:
            day: Day number (1-indexed) or date string
            start_hour: Planning horizon start hour (default 7am)
            end_hour: Planning horizon end hour (default 11am)
        
        Returns:
            dict: Initial state
        """
        self.decision_count = 0  # Track number of decisions for debugging
        # Load static initial inventory
        self._load_static_inventory()
        
        # Reset all stations
        for station in self.stations.values():
            station.reset_metrics()
        
        # Get trips for this day
        if isinstance(day, int):
            days = sorted(self.trips_df['departure_time'].dt.date.unique())
            if len(days) == 0:
                raise ValueError(f"No dates found in trips data")
            if day > len(days):
                raise ValueError(f"Day {day} out of range. Only {len(days)} days available.")
            date = days[day - 1]
        else:
            date = pd.to_datetime(day).date()
        
        day_trips = self.trips_df[
            self.trips_df['departure_time'].dt.date == date
        ]
        
        # Filter for planning horizon
        day_trips = day_trips[
            (day_trips['departure_time'].dt.hour >= start_hour) &
            (day_trips['departure_time'].dt.hour < end_hour)
        ].sort_values('departure_time')
        
        # Set episode times
        self.episode_start_time = datetime.combine(date, datetime.min.time().replace(hour=start_hour))
        self.episode_end_time = datetime.combine(date, datetime.min.time().replace(hour=end_hour))
        self.current_time = self.episode_start_time
        
        print(f"📅 Episode reset: {date}, {len(day_trips)} trips in horizon")
        print(f"  Start: {self.episode_start_time}")
        print(f"  End:   {self.episode_end_time}")
        
        # Initialize event queue
        self.event_queue = []
        
        # Add all customer rental events
        for _, trip in day_trips.iterrows():
            event = create_rental_event(trip)
            heapq.heappush(self.event_queue, event)
        
        # Initialize vehicles at random stations
        for vehicle in self.vehicles.values():
            initial_station = np.random.choice(list(self.stations.keys()))
            vehicle.reset(initial_station)
            
            # Schedule initial vehicle arrival (triggers first decision)
            arrival_event = create_vehicle_arrival_event(
                vehicle.id, initial_station, self.episode_start_time
            )
            heapq.heappush(self.event_queue, arrival_event)
        
        # Reset metrics
        self.total_lost_rentals = 0
        self.total_lost_returns = 0
        self.total_successful_rentals = 0
        self.total_successful_returns = 0
        self.last_decision_time = self.episode_start_time
        self.lost_demand_since_last_decision = 0
        
        print(f"\n📅 Episode reset: {date}, {len(day_trips)} trips in horizon")
        
        return self.get_state()
    
    def _load_static_inventory(self):
        """Load static initial inventory from file."""
        gt_name = Path(self.network_data.get('name', 'GT1'))
        static_file = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / str(gt_name) / f"{gt_name}_static_initial_inventory.json"
        
        if static_file.exists():
            with open(static_file) as f:
                data = json.load(f)
                for sid_str, inventory in data['initial_inventory'].items():
                    sid = int(sid_str)
                    if sid in self.stations:
                        self.stations[sid].inventory = inventory
        else:
            # Fallback to 50%
            for station in self.stations.values():
                station.inventory = station.capacity // 2
    
    def get_next_decision_epoch(self):
        """
        Get next vehicle arrival event (decision epoch).
        
        Returns:
            tuple: (vehicle_id, event) or (None, None) if episode done
        """
        self.decision_count += 1
        
        # Debug: Log every 1000 decisions
        if self.decision_count % 1000 == 0:
            print(f"    DEBUG: Decision {self.decision_count}, Current time: {self.current_time}, End time: {self.episode_end_time}")
        # Process events until we hit a vehicle arrival or end of episode
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            
            # Check if episode ended
            if event.time >= self.episode_end_time:
                return None, None
            
            self.current_time = event.time
            
            # Process event
            if isinstance(event, VehicleArrival):
                # This is a decision epoch!
                vehicle = self.vehicles[event.vehicle_id]
                vehicle.arrive_at_station(event.station_id)
                
                # Calculate reward since last decision
                reward = -self.lost_demand_since_last_decision
                self.lost_demand_since_last_decision = 0
                self.last_decision_time = self.current_time
                
                return event.vehicle_id, reward
            
            elif isinstance(event, CustomerRental):
                self._process_rental(event)
            
            elif isinstance(event, CustomerReturn):
                self._process_return(event)
        
        # No more events
        return None, None
    
    def _process_rental(self, event):
        """Process customer rental attempt."""
        station = self.stations[event.origin_station]
        
        if station.can_rent():
            # Successful rental
            station.rent_bike()
            self.total_successful_rentals += 1
            
            # Schedule return event
            return_event = create_return_event(
                event.trip_id,
                event.destination_station,
                event.arrival_time,
                event.origin_station
            )
            heapq.heappush(self.event_queue, return_event)
        else:
            # Lost rental
            station.lost_rentals += 1
            self.total_lost_rentals += 1
            self.lost_demand_since_last_decision += 1
    
    def _process_return(self, event):
        """Process customer return attempt."""
        station = self.stations[event.station_id]
        
        if station.can_return():
            # Successful return
            station.return_bike()
            self.total_successful_returns += 1
        else:
            # Lost return
            station.lost_returns += 1
            self.total_lost_returns += 1
            self.lost_demand_since_last_decision += 1
    
    def execute_action(self, vehicle_id, action):
        """
        Execute vehicle action and schedule next arrival.
        
        Args:
            vehicle_id: Which vehicle is acting
            action: tuple (next_station_id, fill_level_idx)
        
        Returns:
            None (state will be obtained at next decision epoch)
        """
        vehicle = self.vehicles[vehicle_id]
        next_station_id, fill_level_idx = action
        
        # Perform rebalancing at current station
        self._rebalance_station(vehicle, fill_level_idx)
        
        # Calculate travel distance
        current_station = vehicle.current_station
        distance_km = self.distance_matrix[current_station - 1][next_station_id - 1]
        
        # FIX: If staying at same station (distance=0), add minimum wait time
        # to prevent infinite loop of instant arrivals
        if distance_km == 0:
            distance_km = 0.25  # 0.25 km = ~1 minute travel time at 15 km/h
        
        # Start trip to next station
        arrival_time = vehicle.start_trip(next_station_id, distance_km, self.current_time)
        
        # Schedule vehicle arrival event
        arrival_event = create_vehicle_arrival_event(vehicle_id, next_station_id, arrival_time)
        heapq.heappush(self.event_queue, arrival_event)
    
    def _rebalance_station(self, vehicle, fill_level_idx):
        """
        Rebalance current station to target fill level.
        
        Implements Equation (4) from paper.
        
        Args:
            vehicle: Vehicle object
            fill_level_idx: Index into self.fill_levels (0, 1, or 2)
        """
        # Use configured fill levels
        target_fill = self.fill_levels[fill_level_idx]
        
        station = self.stations[vehicle.current_station]
        target_inventory = int(target_fill * station.capacity)
        current_inventory = station.inventory
        
        if current_inventory > target_inventory:
            # Need to pick up bikes (Equation 4, case 1)
            bikes_to_pickup = min(
                vehicle.capacity - vehicle.inventory,  # Vehicle capacity constraint
                current_inventory - target_inventory   # Amount available to pickup
            )
            if bikes_to_pickup > 0:
                actual_removed = station.remove_bikes(bikes_to_pickup)
                vehicle.load_bikes(actual_removed)
        
        elif current_inventory < target_inventory:
            # Need to drop off bikes (Equation 4, case 2)
            bikes_to_dropoff = min(
                vehicle.inventory,                     # Bikes in vehicle
                target_inventory - current_inventory   # Space available at station
            )
            if bikes_to_dropoff > 0:
                actual_delivered = station.add_bikes(bikes_to_dropoff)
                vehicle.unload_bikes(actual_delivered)
        
        # Else: no rebalancing needed (Equation 4, case 3)
    
    def get_state(self):
        """
        Get current global state for all agents.
        
        State includes:
        - All station inventories
        - All vehicle states (location, inventory)
        - Current time
        - Upcoming trips in queue (limited lookahead)
        """
        # Station inventories
        station_inventories = {sid: s.inventory for sid, s in self.stations.items()}
        station_occupancies = {sid: s.inventory / s.capacity for sid, s in self.stations.items()}
        
        # Vehicle states
        vehicle_states = {vid: v.get_state() for vid, v in self.vehicles.items()}
        
        # Upcoming trips (next N trips for state awareness)
        upcoming_trips = []
        temp_queue = list(self.event_queue)
        heapq.heapify(temp_queue)
        
        count = 0
        while temp_queue and count < 20:  # Look ahead at next 20 events
            event = heapq.heappop(temp_queue)
            if isinstance(event, CustomerRental):
                upcoming_trips.append({
                    'time': event.time,
                    'origin': event.origin_station,
                    'destination': event.destination_station
                })
                count += 1
        
        return {
            'station_inventories': station_inventories,
            'station_occupancies': station_occupancies,
            'vehicle_states': vehicle_states,
            'upcoming_trips': upcoming_trips,
            'current_time': self.current_time,
            'num_stations': len(self.stations),
            'num_vehicles': len(self.vehicles)
        }
    
    def is_done(self):
        """Check if episode is complete."""
        return (not self.event_queue) or (self.current_time >= self.episode_end_time)
    
    def get_metrics(self):
        """Get episode metrics."""
        total_rentals = self.total_successful_rentals + self.total_lost_rentals
        total_returns = self.total_successful_returns + self.total_lost_returns
        total_demand = total_rentals + total_returns
        total_lost = self.total_lost_rentals + self.total_lost_returns
        
        return {
            'total_rentals': total_rentals,
            'successful_rentals': self.total_successful_rentals,
            'lost_rentals': self.total_lost_rentals,
            'lost_rental_rate': (self.total_lost_rentals / total_rentals * 100) if total_rentals > 0 else 0,
            
            'total_returns': total_returns,
            'successful_returns': self.total_successful_returns,
            'lost_returns': self.total_lost_returns,
            'lost_return_rate': (self.total_lost_returns / total_returns * 100) if total_returns > 0 else 0,
            
            'total_lost_demand': total_lost,
            'total_lost_demand_rate': (total_lost / total_demand * 100) if total_demand > 0 else 0,
        }
    
    def __repr__(self):
        return f"ContinuousTimeSimulator(stations={len(self.stations)}, vehicles={len(self.vehicles)})"
