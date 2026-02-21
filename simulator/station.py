"""
Station class for bike-sharing simulation.

Represents a single bike station with:
- Current inventory (number of bikes)
- Capacity (total docks)
- Location information
- Tracking of lost demand
"""

class Station:
    """Represents a bike-sharing station."""
    
    def __init__(self, station_id, capacity, latitude, longitude, is_city_center=False):
        """
        Initialize a station.
        
        Args:
            station_id: Unique station identifier
            capacity: Total number of docks at the station
            latitude: Station latitude
            longitude: Station longitude
            is_city_center: Whether this is a city center station
        """
        self.id = station_id
        self.capacity = capacity
        self.latitude = latitude
        self.longitude = longitude
        self.is_city_center = is_city_center
        
        # Current state
        self.inventory = capacity // 2  # Start at 50% capacity (base paper default)
        
        # Metrics tracking
        self.lost_rentals = 0  # Customers couldn't rent (no bikes)
        self.lost_returns = 0  # Customers couldn't return (no docks)
        
    def can_rent(self):
        """Check if a bike is available for rental."""
        return self.inventory > 0
    
    def can_return(self):
        """Check if there is space to return a bike."""
        return self.inventory < self.capacity
    
    def rent_bike(self):
        """
        Process a bike rental.
        
        Returns:
            bool: True if rental successful, False if lost (no bikes available)
        """
        if self.can_rent():
            self.inventory -= 1
            return True
        else:
            self.lost_rentals += 1
            return False
    
    def return_bike(self):
        """
        Process a bike return.
        
        Returns:
            bool: True if return successful, False if lost (no docks available)
        """
        if self.can_return():
            self.inventory += 1
            return True
        else:
            self.lost_returns += 1
            return False
    
    def add_bikes(self, count):
        """
        Add bikes to station (rebalancing operation).
        
        Args:
            count: Number of bikes to add (can be negative to remove)
        
        Returns:
            int: Actual number of bikes added (may be less due to capacity constraints)
        """
        # Calculate how many bikes can actually be added
        if count > 0:
            # Adding bikes - limited by available docks
            available_docks = self.capacity - self.inventory
            actual_added = min(count, available_docks)
        else:
            # Removing bikes - limited by available bikes
            actual_removed = min(-count, self.inventory)
            actual_added = -actual_removed
        
        self.inventory += actual_added
        return actual_added
    
    def remove_bikes(self, count):
        """
        Remove bikes from station (rebalancing operation).
        
        Args:
            count: Number of bikes to remove
        
        Returns:
            int: Actual number of bikes removed (may be less due to inventory)
        """
        return -self.add_bikes(-count)
    
    def reset_metrics(self):
        """Reset lost demand counters (call at start of episode)."""
        self.lost_rentals = 0
        self.lost_returns = 0
    
    def reset_inventory(self, inventory=None):
        """
        Reset station inventory.
        
        Args:
            inventory: Desired inventory. If None, resets to 50% capacity.
        """
        if inventory is None:
            self.inventory = self.capacity // 2
        else:
            self.inventory = max(0, min(inventory, self.capacity))
    
    def get_state(self):
        """
        Get current station state.
        
        Returns:
            dict: Station state information
        """
        return {
            'id': self.id,
            'inventory': self.inventory,
            'capacity': self.capacity,
            'occupancy_rate': self.inventory / self.capacity,
            'available_bikes': self.inventory,
            'available_docks': self.capacity - self.inventory,
            'lost_rentals': self.lost_rentals,
            'lost_returns': self.lost_returns
        }
    
    def __repr__(self):
        return (f"Station(id={self.id}, inventory={self.inventory}/{self.capacity}, "
                f"lost_rentals={self.lost_rentals}, lost_returns={self.lost_returns})")
