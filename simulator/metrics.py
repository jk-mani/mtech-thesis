"""
Metrics calculation for bike-sharing simulation.

Tracks and calculates performance metrics for evaluation.
"""

import numpy as np


class SimulationMetrics:
    """Tracks and calculates simulation metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        # Lost demand tracking
        self.total_rentals_attempted = 0
        self.total_rentals_successful = 0
        self.total_rentals_lost = 0
        
        self.total_returns_attempted = 0
        self.total_returns_successful = 0
        self.total_returns_lost = 0
        
        # Station-level metrics
        self.station_metrics = {}
        
        # Rebalancing metrics
        self.bikes_rebalanced = 0
        self.rebalancing_operations = 0
        
        # Time series
        self.hourly_lost_rentals = []
        self.hourly_lost_returns = []
        self.hourly_inventories = []
    
    def record_rental_attempt(self, success):
        """Record a rental attempt."""
        self.total_rentals_attempted += 1
        if success:
            self.total_rentals_successful += 1
        else:
            self.total_rentals_lost += 1
    
    def record_return_attempt(self, success):
        """Record a return attempt."""
        self.total_returns_attempted += 1
        if success:
            self.total_returns_successful += 1
        else:
            self.total_returns_lost += 1
    
    def record_rebalancing(self, bikes_moved, num_operations):
        """Record rebalancing operations."""
        self.bikes_rebalanced += bikes_moved
        self.rebalancing_operations += num_operations
    
    def record_station_state(self, stations):
        """
        Record current state of all stations.
        
        Args:
            stations: Dictionary of Station objects
        """
        for station_id, station in stations.items():
            if station_id not in self.station_metrics:
                self.station_metrics[station_id] = {
                    'lost_rentals': 0,
                    'lost_returns': 0,
                    'inventory_history': []
                }
            
            self.station_metrics[station_id]['lost_rentals'] = station.lost_rentals
            self.station_metrics[station_id]['lost_returns'] = station.lost_returns
            self.station_metrics[station_id]['inventory_history'].append(station.inventory)
    
    def record_hourly_snapshot(self, stations):
        """Take hourly snapshot of system state."""
        lost_rentals = sum(s.lost_rentals for s in stations.values())
        lost_returns = sum(s.lost_returns for s in stations.values())
        inventories = [s.inventory for s in stations.values()]
        
        self.hourly_lost_rentals.append(lost_rentals)
        self.hourly_lost_returns.append(lost_returns)
        self.hourly_inventories.append(inventories.copy())
    
    def get_lost_rental_rate(self):
        """Calculate percentage of lost rentals."""
        if self.total_rentals_attempted == 0:
            return 0.0
        return (self.total_rentals_lost / self.total_rentals_attempted) * 100
    
    def get_lost_return_rate(self):
        """Calculate percentage of lost returns."""
        if self.total_returns_attempted == 0:
            return 0.0
        return (self.total_returns_lost / self.total_returns_attempted) * 100
    
    def get_total_lost_demand(self):
        """Calculate total lost demand (rentals + returns)."""
        return self.total_rentals_lost + self.total_returns_lost
    
    def get_total_lost_demand_rate(self):
        """Calculate percentage of total lost demand."""
        total_attempted = self.total_rentals_attempted + self.total_returns_attempted
        if total_attempted == 0:
            return 0.0
        return (self.get_total_lost_demand() / total_attempted) * 100
    
    def get_station_statistics(self):
        """Get per-station statistics."""
        stats = {}
        for station_id, metrics in self.station_metrics.items():
            stats[station_id] = {
                'lost_rentals': metrics['lost_rentals'],
                'lost_returns': metrics['lost_returns'],
                'avg_inventory': np.mean(metrics['inventory_history']) if metrics['inventory_history'] else 0,
                'min_inventory': np.min(metrics['inventory_history']) if metrics['inventory_history'] else 0,
                'max_inventory': np.max(metrics['inventory_history']) if metrics['inventory_history'] else 0,
            }
        return stats
    
    def get_summary(self):
        """
        Get summary of all metrics.
        
        Returns:
            dict: Summary statistics
        """
        return {
            # Demand metrics
            'total_rentals_attempted': self.total_rentals_attempted,
            'total_rentals_successful': self.total_rentals_successful,
            'total_rentals_lost': self.total_rentals_lost,
            'lost_rental_rate': self.get_lost_rental_rate(),
            
            'total_returns_attempted': self.total_returns_attempted,
            'total_returns_successful': self.total_returns_successful,
            'total_returns_lost': self.total_returns_lost,
            'lost_return_rate': self.get_lost_return_rate(),
            
            'total_lost_demand': self.get_total_lost_demand(),
            'total_lost_demand_rate': self.get_total_lost_demand_rate(),
            
            # Rebalancing metrics
            'bikes_rebalanced': self.bikes_rebalanced,
            'rebalancing_operations': self.rebalancing_operations,
            
            # System-level
            'num_stations': len(self.station_metrics),
        }
    
    def print_summary(self, title="Simulation Results"):
        """Print formatted summary."""
        print("\n" + "="*70)
        print(f"{title:^70}")
        print("="*70)
        
        summary = self.get_summary()
        
        print("\n📊 Demand Metrics:")
        print(f"  Rental attempts:     {summary['total_rentals_attempted']:>6}")
        print(f"  Rentals successful:  {summary['total_rentals_successful']:>6}")
        print(f"  Rentals lost:        {summary['total_rentals_lost']:>6} ({summary['lost_rental_rate']:.2f}%)")
        
        print(f"\n  Return attempts:     {summary['total_returns_attempted']:>6}")
        print(f"  Returns successful:  {summary['total_returns_successful']:>6}")
        print(f"  Returns lost:        {summary['total_returns_lost']:>6} ({summary['lost_return_rate']:.2f}%)")
        
        print(f"\n  Total lost demand:   {summary['total_lost_demand']:>6} ({summary['total_lost_demand_rate']:.2f}%)")
        
        print("\n🚚 Rebalancing:")
        print(f"  Bikes moved:         {summary['bikes_rebalanced']:>6}")
        print(f"  Operations:          {summary['rebalancing_operations']:>6}")
        
        print("\n" + "="*70 + "\n")
    
    def __repr__(self):
        return (f"SimulationMetrics(lost_rentals={self.total_rentals_lost}, "
                f"lost_returns={self.total_returns_lost}, "
                f"lost_rate={self.get_total_lost_demand_rate():.2f}%)")
