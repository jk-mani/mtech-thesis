"""
Bike-Sharing Simulation Package

Discrete-event simulator for bike-sharing systems with rebalancing.

Main components:
- BikeShareEnvironment: Main simulation environment
- Station: Individual station management
- TripGenerator: Load and provide trip data
- RebalancingFleet: Vehicle fleet management
- SimulationMetrics: Performance tracking

Usage:
    from simulator import BikeShareEnvironment
    
    env = BikeShareEnvironment(
        network_file='../data/synthetic/GT1/GT1_station_network.json',
        trips_file='../data/synthetic/GT1/GT1_trips_train.csv'
    )
    
    # Run single episode
    state, metrics, info = env.step(day=1)
    
    # Run multiple episodes
    results = env.run_multiple_episodes(days=range(1, 11))
"""

from .environment import BikeShareEnvironment
from .station import Station
from .trip_generator import TripGenerator
from .rebalancing_fleet import RebalancingFleet, Vehicle
from .metrics import SimulationMetrics

__version__ = '1.0.0'

__all__ = [
    'BikeShareEnvironment',
    'Station',
    'TripGenerator',
    'RebalancingFleet',
    'Vehicle',
    'SimulationMetrics',
]
