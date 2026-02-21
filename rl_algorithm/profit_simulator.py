"""
Profit-Based Continuous Time Simulator.

Wraps the base ContinuousTimeSimulator to provide profit-based rewards
instead of simple lost-demand counting. This enables training agents
that optimize for economic efficiency rather than just service quality.

Key differences from base simulator:
- Reward = Profit (Revenue - Costs) instead of -LostDemand
- Tracks additional metrics: distance traveled, time spent, bikes handled
- Provides detailed economic breakdown for analysis
"""

import json
import heapq
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

from rl_algorithm.vehicle import Vehicle
from rl_algorithm.events import (
    Event, create_rental_event, create_return_event, 
    create_vehicle_arrival_event, CustomerRental, CustomerReturn, VehicleArrival
)
from rl_algorithm.profit_reward import ProfitRewardCalculator, ProfitParameters


# Set random seed for reproducibility
np.random.seed(42)


class Station:
    """Station with inventory tracking."""
    
    def __init__(self, station_id, capacity, initial_inventory=None):
        self.id = station_id
        self.capacity = capacity
        self.inventory = initial_inventory if initial_inventory is not None else capacity // 2
        
        self.lost_rentals = 0
        self.lost_returns = 0
        self.successful_rentals = 0
        self.successful_returns = 0
    
    def can_rent(self):
        return self.inventory > 0
    
    def can_return(self):
        return self.inventory < self.capacity
    
    def remove_bikes(self, count):
        actual = min(count, self.inventory)
        self.inventory -= actual
        return actual
    
    def add_bikes(self, count):
        space = self.capacity - self.inventory
        actual = min(count, space)
        self.inventory += actual
        return actual
    
    def reset(self, initial_inventory=None):
        self.inventory = initial_inventory if initial_inventory is not None else self.capacity // 2
        self.lost_rentals = 0
        self.lost_returns = 0
        self.successful_rentals = 0
        self.successful_returns = 0


class ProfitSimulator:
    """
    Continuous-time simulator with profit-based rewards.
    
    This simulator extends the base event-driven simulation with:
    - Profit calculation based on revenue and operational costs
    - Tracking of vehicle movements and rebalancing operations
    - Economic metrics for policy evaluation
    """
    
    def __init__(
        self,
        network_file: str,
        trips_file: str,
        num_vehicles: int = 2,
        vehicle_capacity: int = 15,
        profit_params: Optional[ProfitParameters] = None
    ):
        """
        Initialize profit-based simulator.
        
        Args:
            network_file: Path to station network JSON
            trips_file: Path to trips CSV
            num_vehicles: Number of rebalancing vehicles
            vehicle_capacity: Bike capacity per vehicle
            profit_params: Economic parameters for profit calculation
        """
        # Load network
        with open(network_file) as f:
            self.network_data = json.load(f)
        
        # Initialize stations
        self.stations = {}
        for s in self.network_data['stations']:
            self.stations[s['id']] = Station(s['id'], s['capacity'])
        
        # Distance matrix
        self.distance_matrix = np.array(self.network_data['distance_matrix'])
        
        # Load trips
        import pandas as pd
        self.trips_df = pd.read_csv(trips_file)
        self.trips_df['departure_time'] = pd.to_datetime(self.trips_df['departure_time'])
        self.trips_df['arrival_time'] = pd.to_datetime(self.trips_df['arrival_time'])
        
        # Get unique dates for episode selection
        self.trips_df['date'] = self.trips_df['departure_time'].dt.date
        self.available_dates = sorted(self.trips_df['date'].unique())
        
        # Initialize vehicles
        self.vehicles = {}
        for i in range(1, num_vehicles + 1):
            self.vehicles[i] = Vehicle(i, capacity=vehicle_capacity)
        
        # Profit calculator
        self.profit_calculator = ProfitRewardCalculator(profit_params)
        self.profit_params = profit_params or ProfitParameters()
        
        # Episode state
        self.event_queue = []
        self.current_time = None
        self.episode_start_time = None
        self.episode_end_time = None
        
        # Tracking
        self.total_lost_rentals = 0
        self.total_lost_returns = 0
        self.total_successful_rentals = 0
        self.total_successful_returns = 0
        
        # For profit calculation between decisions
        self.last_decision_time = None
        self.last_decision_stats = None
        self.last_vehicle_stations = {}
        self.bikes_moved_this_action = 0
        
        print(f"ProfitSimulator initialized:")
        print(f"  Stations: {len(self.stations)}")
        print(f"  Vehicles: {num_vehicles}")
        print(f"  Trip revenue: ${self.profit_params.trip_revenue:.2f}")
        print(f"  Lost demand penalty: ${self.profit_params.lost_demand_penalty:.2f}")
    
    def _load_static_inventory(self):
        """Load static initial inventory from MIP solution if available."""
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
            for station in self.stations.values():
                station.inventory = station.capacity // 2
    
    def reset(self, date=None):
        """
        Reset simulator for new episode.
        
        Args:
            date: Specific date to use (random if None)
            
        Returns:
            dict: Initial state
        """
        # Select date
        if date is None:
            date = np.random.choice(self.available_dates)
        
        # Get trips for this date
        day_trips = self.trips_df[self.trips_df['date'] == date].copy()
        
        # Clear event queue
        self.event_queue = []
        
        # Set time bounds
        self.episode_start_time = day_trips['departure_time'].min()
        self.episode_end_time = day_trips['arrival_time'].max() + timedelta(minutes=30)
        self.current_time = self.episode_start_time
        
        # Reset stations
        for station in self.stations.values():
            station.reset()
        self._load_static_inventory()
        
        # Reset vehicles at random stations
        station_ids = list(self.stations.keys())
        for vehicle in self.vehicles.values():
            initial_station = np.random.choice(station_ids)
            vehicle.reset(initial_station)
            self.last_vehicle_stations[vehicle.id] = initial_station
        
        # Schedule all rental events (returns scheduled upon successful rental)
        for _, trip in day_trips.iterrows():
            rental_event = create_rental_event(trip)
            heapq.heappush(self.event_queue, rental_event)
        
        # Schedule initial vehicle arrivals
        for vehicle in self.vehicles.values():
            arrival_event = create_vehicle_arrival_event(
                vehicle.id, vehicle.current_station,
                self.episode_start_time + timedelta(seconds=1)
            )
            heapq.heappush(self.event_queue, arrival_event)
        
        # Reset counters
        self.total_lost_rentals = 0
        self.total_lost_returns = 0
        self.total_successful_rentals = 0
        self.total_successful_returns = 0
        
        # Reset profit calculator
        self.profit_calculator.reset()
        self.last_decision_time = self.episode_start_time
        self.last_decision_stats = self._get_current_stats()
        self.bikes_moved_this_action = 0
        
        print(f"\n📅 Episode reset: {date}, {len(day_trips)} trips")
        
        return self.get_state()
    
    def _get_current_stats(self) -> Dict:
        """Get current trip statistics."""
        return {
            'successful_rentals': self.total_successful_rentals,
            'successful_returns': self.total_successful_returns,
            'lost_rentals': self.total_lost_rentals,
            'lost_returns': self.total_lost_returns
        }
    
    def step(self) -> Optional[Tuple[int, float]]:
        """
        Process events until a vehicle needs a decision.
        
        Returns:
            tuple: (vehicle_id, profit_reward) or None if episode done
        """
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            
            if event.time > self.episode_end_time:
                return None
            
            self.current_time = event.time
            
            if isinstance(event, VehicleArrival):
                vehicle = self.vehicles[event.vehicle_id]
                vehicle.arrive_at_station(event.station_id)
                
                # Calculate profit reward since last decision
                profit_reward = self._calculate_step_profit(event.vehicle_id)
                
                return event.vehicle_id, profit_reward
            
            elif isinstance(event, CustomerRental):
                self._process_rental(event)
            
            elif isinstance(event, CustomerReturn):
                self._process_return(event)
        
        return None
    
    def _calculate_step_profit(self, vehicle_id: int) -> float:
        """Calculate profit-based reward for this decision step."""
        current_stats = self._get_current_stats()
        
        # Delta since last decision
        delta = {
            'successful_rentals': current_stats['successful_rentals'] - self.last_decision_stats['successful_rentals'],
            'successful_returns': current_stats['successful_returns'] - self.last_decision_stats['successful_returns'],
            'lost_rentals': current_stats['lost_rentals'] - self.last_decision_stats['lost_rentals'],
            'lost_returns': current_stats['lost_returns'] - self.last_decision_stats['lost_returns']
        }
        
        # Time elapsed
        time_delta = self.current_time - self.last_decision_time
        time_elapsed_hours = time_delta.total_seconds() / 3600
        
        # Distance traveled
        vehicle = self.vehicles[vehicle_id]
        current_station = vehicle.current_station
        last_station = self.last_vehicle_stations.get(vehicle_id, current_station)
        
        if last_station != current_station:
            distance_km = self.distance_matrix[last_station - 1][current_station - 1]
        else:
            distance_km = 0.0
        
        # Calculate profit
        profit = self.profit_calculator.calculate_step_reward(
            successful_rentals=delta['successful_rentals'],
            successful_returns=delta['successful_returns'],
            lost_rentals=delta['lost_rentals'],
            lost_returns=delta['lost_returns'],
            distance_traveled_km=distance_km,
            time_elapsed_hours=time_elapsed_hours,
            bikes_loaded=0,  # Will add after execute_action
            bikes_unloaded=0,
            made_stop=(distance_km > 0)
        )
        
        # Update tracking
        self.last_decision_time = self.current_time
        self.last_decision_stats = current_stats.copy()
        self.last_vehicle_stations[vehicle_id] = current_station
        self.bikes_moved_this_action = 0
        
        return profit
    
    def _process_rental(self, event):
        """Process rental attempt."""
        station = self.stations[event.origin_station]
        
        if station.can_rent():
            station.remove_bikes(1)
            station.successful_rentals += 1
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
            station.lost_rentals += 1
            self.total_lost_rentals += 1
    
    def _process_return(self, event):
        """Process return attempt."""
        station = self.stations[event.station_id]
        
        if station.can_return():
            station.add_bikes(1)
            station.successful_returns += 1
            self.total_successful_returns += 1
        else:
            station.lost_returns += 1
            self.total_lost_returns += 1
    
    def execute_action(self, vehicle_id: int, action: Tuple[int, int]):
        """
        Execute vehicle action and schedule next arrival.
        
        Args:
            vehicle_id: Which vehicle is acting
            action: (next_station_id, fill_level_idx)
        """
        vehicle = self.vehicles[vehicle_id]
        next_station_id, fill_level_idx = action
        
        # Perform rebalancing at current station
        bikes_loaded, bikes_unloaded = self._rebalance_station(vehicle, fill_level_idx)
        
        # Track bikes moved for profit calculation
        self.bikes_moved_this_action = bikes_loaded + bikes_unloaded
        
        # Add handling cost to profit calculator
        handling_cost = self.bikes_moved_this_action * self.profit_params.handling_cost_per_bike
        self.profit_calculator.episode_handling_cost += handling_cost
        self.profit_calculator.total_bikes_moved += self.bikes_moved_this_action
        
        # Calculate travel distance
        current_station = vehicle.current_station
        distance_km = self.distance_matrix[current_station - 1][next_station_id - 1]
        
        # Minimum travel time if staying at same station
        if distance_km == 0:
            distance_km = 0.25
        
        # Start trip
        arrival_time = vehicle.start_trip(next_station_id, distance_km, self.current_time)
        
        # Schedule arrival event
        arrival_event = create_vehicle_arrival_event(vehicle_id, next_station_id, arrival_time)
        heapq.heappush(self.event_queue, arrival_event)
    
    def _rebalance_station(self, vehicle: Vehicle, fill_level_idx: int) -> Tuple[int, int]:
        """
        Rebalance station to target fill level.
        
        Returns:
            tuple: (bikes_loaded, bikes_unloaded)
        """
        fill_levels = [0.10, 0.50, 0.90]
        target_fill = fill_levels[fill_level_idx]
        
        station = self.stations[vehicle.current_station]
        target_inventory = int(target_fill * station.capacity)
        current_inventory = station.inventory
        
        bikes_loaded = 0
        bikes_unloaded = 0
        
        if current_inventory > target_inventory:
            # Pick up bikes
            bikes_to_pickup = min(
                vehicle.capacity - vehicle.inventory,
                current_inventory - target_inventory
            )
            if bikes_to_pickup > 0:
                actual = station.remove_bikes(bikes_to_pickup)
                vehicle.load_bikes(actual)
                bikes_loaded = actual
        
        elif current_inventory < target_inventory:
            # Drop off bikes
            bikes_to_dropoff = min(
                vehicle.inventory,
                target_inventory - current_inventory
            )
            if bikes_to_dropoff > 0:
                actual = station.add_bikes(bikes_to_dropoff)
                vehicle.unload_bikes(actual)
                bikes_unloaded = actual
        
        return bikes_loaded, bikes_unloaded
    
    def get_state(self) -> Dict:
        """Get current state for all agents (matches ContinuousTimeSimulator format)."""
        # Station inventories
        station_inventories = {sid: s.inventory for sid, s in self.stations.items()}
        station_occupancies = {sid: s.inventory / s.capacity for sid, s in self.stations.items()}
        
        # Vehicle states
        vehicle_states = {vid: v.get_state() for vid, v in self.vehicles.items()}
        
        # Upcoming trips (simplified - just count)
        upcoming_trips = []
        
        return {
            'station_inventories': station_inventories,
            'station_occupancies': station_occupancies,
            'vehicle_states': vehicle_states,
            'upcoming_trips': upcoming_trips,
            'current_time': self.current_time,
            'num_stations': len(self.stations),
            'num_vehicles': len(self.vehicles)
        }
    
    def get_metrics(self) -> Dict:
        """Get combined metrics including profit summary."""
        total_demand = (self.total_successful_rentals + self.total_lost_rentals +
                       self.total_successful_returns + self.total_lost_returns)
        total_lost = self.total_lost_rentals + self.total_lost_returns
        
        base_metrics = {
            'total_trips': self.total_successful_rentals,
            'successful_rentals': self.total_successful_rentals,
            'successful_returns': self.total_successful_returns,
            'lost_rentals': self.total_lost_rentals,
            'lost_returns': self.total_lost_returns,
            'total_lost_demand': total_lost,
            'lost_demand_rate': (total_lost / total_demand * 100) if total_demand > 0 else 0
        }
        
        # Add profit metrics
        profit_summary = self.profit_calculator.get_episode_summary()
        base_metrics.update(profit_summary)
        
        return base_metrics


if __name__ == "__main__":
    # Test the profit simulator
    print("Testing ProfitSimulator...")
    
    sim = ProfitSimulator(
        network_file='../data/synthetic/GT0/GT0_station_network.json',
        trips_file='../data/synthetic/GT0/GT0_trips_train.csv',
        num_vehicles=2,
        vehicle_capacity=15
    )
    
    state = sim.reset()
    print(f"Initial state: {len(state['stations'])} stations")
    
    total_profit = 0
    steps = 0
    
    while True:
        result = sim.step()
        if result is None:
            break
        
        vehicle_id, profit = result
        total_profit += profit
        steps += 1
        
        # Take random action
        action = (np.random.randint(1, 11), np.random.randint(0, 3))
        sim.execute_action(vehicle_id, action)
    
    print(f"\nEpisode complete:")
    print(f"  Steps: {steps}")
    print(f"  Total profit: ${total_profit:.2f}")
    
    metrics = sim.get_metrics()
    print(f"\nMetrics:")
    print(f"  Lost demand rate: {metrics['lost_demand_rate']:.2f}%")
    print(f"  Net profit: ${metrics['net_profit']:.2f}")
    print(f"  Revenue: ${metrics['revenue']:.2f}")
    print(f"  Operational cost: ${metrics['total_operational_cost']:.2f}")
