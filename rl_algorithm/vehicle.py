"""
Vehicle class for multi-agent rebalancing.

Each vehicle is an independent agent that:
- Travels between stations
- Performs rebalancing operations
- Makes decisions upon arrival at stations
"""

from datetime import datetime, timedelta


class Vehicle:
    """Represents a rebalancing vehicle (agent)."""
    
    def __init__(self, vehicle_id, capacity=20, speed_kmh=15):
        """
        Initialize vehicle.
        
        Args:
            vehicle_id: Unique vehicle identifier (1-10)
            capacity: Maximum bikes the vehicle can carry
            speed_kmh: Average travel speed in km/h
        """
        self.id = vehicle_id
        self.capacity = capacity
        self.speed_kmh = speed_kmh
        
        # Current state
        self.current_station = None  # Station ID where vehicle is located
        self.inventory = 0  # Number of bikes currently in vehicle
        self.is_traveling = False
        self.destination_station = None
        self.arrival_time = None
        
        # Statistics
        self.total_bikes_moved = 0
        self.total_trips_made = 0
        self.stations_visited = []
    
    def is_at_station(self):
        """Check if vehicle is currently at a station (not traveling)."""
        return not self.is_traveling
    
    def can_load_bikes(self, num_bikes):
        """Check if vehicle can load given number of bikes."""
        return self.inventory + num_bikes <= self.capacity
    
    def can_unload_bikes(self, num_bikes):
        """Check if vehicle has enough bikes to unload."""
        return self.inventory >= num_bikes
    
    def load_bikes(self, num_bikes):
        """
        Load bikes onto vehicle.
        
        Args:
            num_bikes: Number of bikes to load (positive)
        
        Returns:
            int: Actual number loaded
        """
        available_capacity = self.capacity - self.inventory
        actual_loaded = min(num_bikes, available_capacity)
        self.inventory += actual_loaded
        self.total_bikes_moved += actual_loaded
        return actual_loaded
    
    def unload_bikes(self, num_bikes):
        """
        Unload bikes from vehicle.
        
        Args:
            num_bikes: Number of bikes to unload (positive)
        
        Returns:
            int: Actual number unloaded
        """
        actual_unloaded = min(num_bikes, self.inventory)
        self.inventory -= actual_unloaded
        self.total_bikes_moved += actual_unloaded
        return actual_unloaded
    
    def start_trip(self, destination_station, distance_km, current_time):
        """
        Start traveling to destination station.
        
        Args:
            destination_station: Station ID to travel to
            distance_km: Distance to station
            current_time: Current simulation time
        
        Returns:
            datetime: Arrival time at destination
        """
        self.is_traveling = True
        self.destination_station = destination_station
        
        # Calculate travel time
        travel_hours = distance_km / self.speed_kmh
        travel_time = timedelta(hours=travel_hours)
        self.arrival_time = current_time + travel_time
        
        self.total_trips_made += 1
        
        return self.arrival_time
    
    def arrive_at_station(self, station_id):
        """
        Complete trip and arrive at station.
        
        Args:
            station_id: Station ID where vehicle arrived
        """
        self.is_traveling = False
        self.current_station = station_id
        self.destination_station = None
        self.arrival_time = None
        
        self.stations_visited.append(station_id)
    
    def reset(self, initial_station=None):
        """
        Reset vehicle to initial state.
        
        Args:
            initial_station: Station where vehicle starts (random if None)
        """
        self.current_station = initial_station
        self.inventory = 0
        self.is_traveling = False
        self.destination_station = None
        self.arrival_time = None
        
        self.total_bikes_moved = 0
        self.total_trips_made = 0
        self.stations_visited = [initial_station] if initial_station else []
    
    def get_state(self):
        """Get current vehicle state."""
        return {
            'id': self.id,
            'current_station': self.current_station,
            'inventory': self.inventory,
            'capacity': self.capacity,
            'is_traveling': self.is_traveling,
            'destination': self.destination_station,
            'arrival_time': self.arrival_time,
            'occupancy': self.inventory / self.capacity if self.capacity > 0 else 0
        }
    
    def __repr__(self):
        if self.is_traveling:
            return (f"Vehicle {self.id}: Traveling to Station {self.destination_station} "
                   f"(arrives {self.arrival_time.strftime('%H:%M')}), {self.inventory} bikes")
        else:
            return (f"Vehicle {self.id}: At Station {self.current_station}, "
                   f"{self.inventory}/{self.capacity} bikes")
