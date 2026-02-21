"""
Rebalancing Fleet for bike-sharing simulation.

Manages fleet of vehicles that move bikes between stations.
"""

from datetime import timedelta


class Vehicle:
    """Represents a single rebalancing vehicle."""
    
    def __init__(self, vehicle_id, capacity=20):
        """
        Initialize a vehicle.
        
        Args:
            vehicle_id: Unique vehicle identifier
            capacity: Maximum number of bikes the vehicle can carry
        """
        self.id = vehicle_id
        self.capacity = capacity
        self.current_load = 0
        self.is_busy = False
        self.current_location = None  # Station ID where vehicle is located
        self.destination = None
        self.arrival_time = None
    
    def load_bikes(self, count):
        """
        Load bikes onto vehicle.
        
        Args:
            count: Number of bikes to load
        
        Returns:
            int: Actual number loaded (limited by capacity)
        """
        available_space = self.capacity - self.current_load
        actual_loaded = min(count, available_space)
        self.current_load += actual_loaded
        return actual_loaded
    
    def unload_bikes(self, count):
        """
        Unload bikes from vehicle.
        
        Args:
            count: Number of bikes to unload
        
        Returns:
            int: Actual number unloaded (limited by current load)
        """
        actual_unloaded = min(count, self.current_load)
        self.current_load -= actual_unloaded
        return actual_unloaded
    
    def start_trip(self, destination, arrival_time):
        """Start a trip to a destination."""
        self.is_busy = True
        self.destination = destination
        self.arrival_time = arrival_time
    
    def complete_trip(self):
        """Complete current trip."""
        self.is_busy = False
        self.current_location = self.destination
        self.destination = None
        self.arrival_time = None
    
    def is_available(self, current_time):
        """Check if vehicle is available at current time."""
        if not self.is_busy:
            return True
        return current_time >= self.arrival_time
    
    def __repr__(self):
        return f"Vehicle(id={self.id}, load={self.current_load}/{self.capacity}, busy={self.is_busy})"


class RebalancingFleet:
    """Manages fleet of rebalancing vehicles."""
    
    def __init__(self, num_vehicles=10, vehicle_capacity=20, avg_speed_kmh=15):
        """
        Initialize rebalancing fleet.
        
        Args:
            num_vehicles: Number of vehicles in fleet
            vehicle_capacity: Capacity of each vehicle (bikes)
            avg_speed_kmh: Average travel speed in km/h
        """
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.avg_speed_kmh = avg_speed_kmh
        
        # Create fleet
        self.vehicles = [Vehicle(i, vehicle_capacity) for i in range(num_vehicles)]
        
        # Statistics
        self.total_bikes_moved = 0
        self.total_trips_made = 0
    
    def get_available_vehicles(self, current_time):
        """
        Get list of currently available vehicles.
        
        Args:
            current_time: Current simulation time
        
        Returns:
            list: Available vehicles
        """
        return [v for v in self.vehicles if v.is_available(current_time)]
    
    def calculate_travel_time(self, distance_km):
        """
        Calculate travel time between stations.
        
        Args:
            distance_km: Distance in kilometers
        
        Returns:
            timedelta: Travel time
        """
        hours = distance_km / self.avg_speed_kmh
        minutes = hours * 60
        return timedelta(minutes=minutes)
    
    def execute_rebalancing(self, actions, stations, distance_matrix, current_time):
        """
        Execute rebalancing operations.
        
        Args:
            actions: List of (origin_station_id, dest_station_id, num_bikes) tuples
            stations: Dictionary of Station objects {station_id: Station}
            distance_matrix: 2D array of distances between stations
            current_time: Current simulation time
        
        Returns:
            int: Number of bikes successfully moved
        """
        available_vehicles = self.get_available_vehicles(current_time)
        
        if not available_vehicles or not actions:
            return 0
        
        bikes_moved = 0
        
        for action in actions:
            if not available_vehicles:
                break  # No more vehicles available
            
            origin_id, dest_id, num_bikes = action
            
            # Skip invalid actions
            if num_bikes <= 0:
                continue
            
            # Get stations
            origin_station = stations[origin_id]
            dest_station = stations[dest_id]
            
            # Get vehicle
            vehicle = available_vehicles.pop(0)
            
            # Step 1: Travel to origin (if not already there)
            if vehicle.current_location != origin_id:
                # For simplicity, assume vehicles start at origin
                # In more complex version, would need to travel to origin first
                pass
            
            # Step 2: Load bikes from origin
            bikes_to_move = min(num_bikes, origin_station.inventory, vehicle.capacity)
            actual_removed = origin_station.remove_bikes(bikes_to_move)
            actual_loaded = vehicle.load_bikes(actual_removed)
            
            if actual_loaded == 0:
                continue  # Nothing to move
            
            # Step 3: Calculate travel time to destination
            # Note: distance_matrix indices may need mapping from station IDs
            # For now, assume station IDs are 1-indexed and match matrix indices
            distance = distance_matrix[origin_id - 1][dest_id - 1]
            travel_time = self.calculate_travel_time(distance)
            
            # Step 4: Start trip
            vehicle.start_trip(dest_id, current_time + travel_time)
            
            # Step 5: Immediate delivery (simplified version)
            # In reality, bikes would be delivered when vehicle arrives
            # For this version, we assume instant delivery for simplicity
            actual_delivered = dest_station.add_bikes(actual_loaded)
            vehicle.unload_bikes(actual_delivered)
            vehicle.complete_trip()
            
            # Update statistics
            bikes_moved += actual_delivered
            self.total_bikes_moved += actual_delivered
            self.total_trips_made += 1
        
        return bikes_moved
    
    def reset(self):
        """Reset fleet to initial state."""
        for vehicle in self.vehicles:
            vehicle.current_load = 0
            vehicle.is_busy = False
            vehicle.current_location = None
            vehicle.destination = None
            vehicle.arrival_time = None
        
        self.total_bikes_moved = 0
        self.total_trips_made = 0
    
    def get_statistics(self):
        """Get fleet statistics."""
        return {
            'num_vehicles': self.num_vehicles,
            'vehicle_capacity': self.vehicle_capacity,
            'total_bikes_moved': self.total_bikes_moved,
            'total_trips_made': self.total_trips_made,
            'avg_bikes_per_trip': self.total_bikes_moved / max(1, self.total_trips_made)
        }
    
    def __repr__(self):
        available = sum(1 for v in self.vehicles if not v.is_busy)
        return (f"RebalancingFleet(vehicles={self.num_vehicles}, "
                f"available={available}, bikes_moved={self.total_bikes_moved})")
