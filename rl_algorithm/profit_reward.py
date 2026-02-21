"""
Profit-Based Reward Calculator for Bike-Sharing Rebalancing.

Extends beyond the base paper's lost-demand minimization by introducing
a profit-based reward function that balances:
- Revenue from successful trips
- Operational costs (vehicle travel, driver time, bike handling)
- Opportunity cost of lost demand

This enables the RL agent to learn economically optimal rebalancing policies.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from datetime import timedelta


@dataclass
class ProfitParameters:
    """
    Economic parameters for profit calculation.
    
    Default values are based on typical bike-sharing economics:
    - Trip revenue: Base fare + distance-based charge (like real bike-sharing)
    - Vehicle operating cost: ~$0.50/km (fuel, maintenance, depreciation)
    - Driver hourly wage: ~$20/hour
    - Bike handling: ~$0.10 per bike loaded/unloaded
    - Lost demand penalty: Represents both lost revenue and customer churn
    """
    
    # Revenue parameters - Distance-based pricing (more realistic)
    trip_base_fare: float = 1.00        # $ base fare per trip unlock
    trip_per_km_rate: float = 0.75      # $ per km of trip distance
    trip_revenue: float = 3.50          # $ flat rate (used if distance unknown)
    
    # Operational cost parameters
    cost_per_km: float = 0.50           # $ per km traveled (fuel, maintenance)
    cost_per_hour: float = 20.00        # $ per hour (driver wages)
    handling_cost_per_bike: float = 0.10  # $ per bike loaded/unloaded
    
    # Fixed costs per rebalancing operation
    stop_cost: float = 0.50             # $ per station stop (time overhead)
    
    # Opportunity cost
    lost_demand_penalty: float = 5.00   # $ per lost rental/return (lost revenue + goodwill)
    
    # Discount for future rewards (for episode-level calculations)
    gamma: float = 0.99
    
    def __post_init__(self):
        """Validate parameters."""
        assert self.trip_base_fare >= 0, "Trip base fare must be non-negative"
        assert self.trip_per_km_rate >= 0, "Trip per km rate must be non-negative"
        assert self.trip_revenue >= 0, "Trip revenue must be non-negative"
        assert self.cost_per_km >= 0, "Cost per km must be non-negative"
        assert self.cost_per_hour >= 0, "Cost per hour must be non-negative"
        assert self.handling_cost_per_bike >= 0, "Handling cost must be non-negative"
        assert self.lost_demand_penalty >= 0, "Lost demand penalty must be non-negative"
    
    def calculate_trip_revenue(self, trip_distance_km: float = None) -> float:
        """
        Calculate revenue for a single trip.
        
        Args:
            trip_distance_km: Distance of the bike trip (user riding).
                             If None, uses flat trip_revenue.
        
        Returns:
            float: Revenue for this trip
        """
        if trip_distance_km is not None:
            # Distance-based pricing: base_fare + (distance × per_km_rate)
            return self.trip_base_fare + (trip_distance_km * self.trip_per_km_rate)
        else:
            # Fallback to flat rate
            return self.trip_revenue


class ProfitRewardCalculator:
    """
    Calculates profit-based rewards for bike-sharing rebalancing.
    
    Profit = Revenue - Operational Costs - Opportunity Cost
    
    Where:
    - Revenue = successful_trips * trip_revenue
    - Operational Costs = travel_cost + time_cost + handling_cost + stop_cost
    - Opportunity Cost = lost_demand * lost_demand_penalty
    
    This reward function encourages the agent to:
    1. Maximize successful trips (keep bikes available where needed)
    2. Minimize unnecessary vehicle movements
    3. Optimize route efficiency
    4. Balance proactive vs reactive rebalancing
    """
    
    def __init__(self, params: Optional[ProfitParameters] = None):
        """
        Initialize profit calculator.
        
        Args:
            params: Economic parameters (uses defaults if None)
        """
        self.params = params or ProfitParameters()
        
        # Tracking for current episode
        self.reset()
    
    def reset(self):
        """Reset episode tracking."""
        self.episode_revenue = 0.0
        self.episode_travel_cost = 0.0
        self.episode_time_cost = 0.0
        self.episode_handling_cost = 0.0
        self.episode_stop_cost = 0.0
        self.episode_lost_demand_cost = 0.0
        
        # Counts for reporting
        self.successful_trips = 0
        self.lost_rentals = 0
        self.lost_returns = 0
        self.total_distance_km = 0.0
        self.total_time_hours = 0.0
        self.total_bikes_moved = 0
        self.total_stops = 0
    
    def calculate_step_reward(
        self,
        successful_rentals: int,
        successful_returns: int,
        lost_rentals: int,
        lost_returns: int,
        distance_traveled_km: float,
        time_elapsed_hours: float,
        bikes_loaded: int,
        bikes_unloaded: int,
        made_stop: bool = True,
        trip_distances: list = None
    ) -> float:
        """
        Calculate profit-based reward for a single decision step.
        
        Args:
            successful_rentals: Number of successful rentals since last decision
            successful_returns: Number of successful returns since last decision
            lost_rentals: Number of lost rentals since last decision
            lost_returns: Number of lost returns since last decision
            distance_traveled_km: Distance traveled by rebalancing vehicle (truck)
            time_elapsed_hours: Time elapsed since last decision
            bikes_loaded: Number of bikes loaded onto vehicle
            bikes_unloaded: Number of bikes unloaded from vehicle
            made_stop: Whether vehicle made a rebalancing stop
            trip_distances: List of trip distances (km) for distance-based revenue.
                           If None, uses flat rate.
            
        Returns:
            float: Profit-based reward (can be positive or negative)
        """
        # Revenue from successful trips
        total_successful = successful_rentals + successful_returns
        
        if trip_distances is not None and len(trip_distances) > 0:
            # Distance-based revenue: sum of (base_fare + distance × per_km_rate)
            revenue = sum(self.params.calculate_trip_revenue(d) for d in trip_distances)
        else:
            # Flat rate fallback
            revenue = total_successful * self.params.trip_revenue
        
        # Operational costs
        travel_cost = distance_traveled_km * self.params.cost_per_km
        time_cost = time_elapsed_hours * self.params.cost_per_hour
        handling_cost = (bikes_loaded + bikes_unloaded) * self.params.handling_cost_per_bike
        stop_cost = self.params.stop_cost if made_stop else 0.0
        
        # Opportunity cost of lost demand
        total_lost = lost_rentals + lost_returns
        lost_demand_cost = total_lost * self.params.lost_demand_penalty
        
        # Total operational cost
        operational_cost = travel_cost + time_cost + handling_cost + stop_cost
        
        # Profit = Revenue - Costs
        profit = revenue - operational_cost - lost_demand_cost
        
        # Update episode tracking
        self.episode_revenue += revenue
        self.episode_travel_cost += travel_cost
        self.episode_time_cost += time_cost
        self.episode_handling_cost += handling_cost
        self.episode_stop_cost += stop_cost
        self.episode_lost_demand_cost += lost_demand_cost
        
        self.successful_trips += total_successful
        self.lost_rentals += lost_rentals
        self.lost_returns += lost_returns
        self.total_distance_km += distance_traveled_km
        self.total_time_hours += time_elapsed_hours
        self.total_bikes_moved += bikes_loaded + bikes_unloaded
        if made_stop:
            self.total_stops += 1
        
        return profit
    
    def get_episode_summary(self) -> Dict:
        """
        Get summary of episode economics.
        
        Returns:
            dict: Economic summary with revenue, costs, and profit
        """
        total_operational_cost = (
            self.episode_travel_cost +
            self.episode_time_cost +
            self.episode_handling_cost +
            self.episode_stop_cost
        )
        
        total_cost = total_operational_cost + self.episode_lost_demand_cost
        net_profit = self.episode_revenue - total_cost
        
        return {
            # Revenue
            'revenue': self.episode_revenue,
            'successful_trips': self.successful_trips,
            'revenue_per_trip': self.episode_revenue / max(1, self.successful_trips),
            
            # Operational costs breakdown
            'travel_cost': self.episode_travel_cost,
            'time_cost': self.episode_time_cost,
            'handling_cost': self.episode_handling_cost,
            'stop_cost': self.episode_stop_cost,
            'total_operational_cost': total_operational_cost,
            
            # Lost demand
            'lost_demand_cost': self.episode_lost_demand_cost,
            'lost_rentals': self.lost_rentals,
            'lost_returns': self.lost_returns,
            'total_lost_demand': self.lost_rentals + self.lost_returns,
            
            # Profit metrics
            'net_profit': net_profit,
            'profit_margin': (net_profit / self.episode_revenue * 100) if self.episode_revenue > 0 else 0,
            
            # Operational metrics
            'total_distance_km': self.total_distance_km,
            'total_time_hours': self.total_time_hours,
            'total_bikes_moved': self.total_bikes_moved,
            'total_stops': self.total_stops,
            'cost_per_km_actual': self.episode_travel_cost / max(0.01, self.total_distance_km),
        }
    
    def get_normalized_reward(self, profit: float, scale: float = 10.0) -> float:
        """
        Normalize profit reward for stable training.
        
        Args:
            profit: Raw profit value
            scale: Scaling factor (typical episode profit magnitude)
            
        Returns:
            float: Normalized reward in roughly [-1, 1] range
        """
        return profit / scale


class ProfitRewardWrapper:
    """
    Wrapper to integrate profit rewards with the ContinuousTimeSimulator.
    
    Tracks additional metrics needed for profit calculation that aren't
    in the base simulator (distance traveled, bikes moved, etc.)
    """
    
    def __init__(self, simulator, params: Optional[ProfitParameters] = None):
        """
        Initialize wrapper around simulator.
        
        Args:
            simulator: ContinuousTimeSimulator instance
            params: Economic parameters
        """
        self.simulator = simulator
        self.calculator = ProfitRewardCalculator(params)
        
        # Track vehicle-level metrics
        self.vehicle_distances = {}  # vehicle_id -> total distance
        self.last_vehicle_positions = {}  # vehicle_id -> last station
        self.last_decision_stats = {}  # Track stats at last decision
        
    def reset(self, date=None):
        """Reset wrapper and simulator."""
        self.calculator.reset()
        self.vehicle_distances = {v: 0.0 for v in self.simulator.vehicles}
        self.last_vehicle_positions = {}
        
        # Store baseline stats
        self.last_decision_stats = {
            'successful_rentals': 0,
            'successful_returns': 0,
            'lost_rentals': 0,
            'lost_returns': 0,
            'time': None
        }
        
        return self.simulator.reset(date)
    
    def step(self):
        """
        Execute one step and return profit-based reward.
        
        Returns:
            tuple: (vehicle_id, profit_reward) or (None, 0) if episode done
        """
        result = self.simulator.step()
        
        if result is None:
            return None, 0.0
        
        vehicle_id, base_reward = result
        
        # Calculate metrics since last decision
        current_stats = {
            'successful_rentals': self.simulator.total_successful_rentals,
            'successful_returns': self.simulator.total_successful_returns,
            'lost_rentals': self.simulator.total_lost_rentals,
            'lost_returns': self.simulator.total_lost_returns,
            'time': self.simulator.current_time
        }
        
        # Delta since last decision
        delta_successful_rentals = current_stats['successful_rentals'] - self.last_decision_stats['successful_rentals']
        delta_successful_returns = current_stats['successful_returns'] - self.last_decision_stats['successful_returns']
        delta_lost_rentals = current_stats['lost_rentals'] - self.last_decision_stats['lost_rentals']
        delta_lost_returns = current_stats['lost_returns'] - self.last_decision_stats['lost_returns']
        
        # Calculate time elapsed
        if self.last_decision_stats['time'] is not None:
            time_delta = current_stats['time'] - self.last_decision_stats['time']
            time_elapsed_hours = time_delta.total_seconds() / 3600
        else:
            time_elapsed_hours = 0.0
        
        # Get vehicle travel distance
        vehicle = self.simulator.vehicles[vehicle_id]
        current_station = vehicle.current_station
        
        distance_km = 0.0
        if vehicle_id in self.last_vehicle_positions:
            last_station = self.last_vehicle_positions[vehicle_id]
            if last_station is not None and last_station != current_station:
                # Get distance from distance matrix
                distance_km = self.simulator.distance_matrix[last_station - 1][current_station - 1]
        
        self.last_vehicle_positions[vehicle_id] = current_station
        self.vehicle_distances[vehicle_id] = self.vehicle_distances.get(vehicle_id, 0) + distance_km
        
        # Note: bikes_loaded/unloaded would need to be tracked in execute_action
        # For now, we estimate based on station changes (simplified)
        bikes_moved = 0  # Will be set properly when integrating with execute_action
        
        # Calculate profit reward
        profit_reward = self.calculator.calculate_step_reward(
            successful_rentals=delta_successful_rentals,
            successful_returns=delta_successful_returns,
            lost_rentals=delta_lost_rentals,
            lost_returns=delta_lost_returns,
            distance_traveled_km=distance_km,
            time_elapsed_hours=time_elapsed_hours,
            bikes_loaded=0,  # Updated separately
            bikes_unloaded=0,  # Updated separately
            made_stop=(distance_km > 0)
        )
        
        # Update last decision stats
        self.last_decision_stats = current_stats.copy()
        
        return vehicle_id, profit_reward
    
    def record_rebalancing(self, bikes_loaded: int, bikes_unloaded: int):
        """
        Record bikes moved during rebalancing operation.
        
        Call this after execute_action to properly track handling costs.
        
        Args:
            bikes_loaded: Bikes loaded onto vehicle
            bikes_unloaded: Bikes unloaded from vehicle
        """
        # Add handling cost retroactively
        handling_cost = (bikes_loaded + bikes_unloaded) * self.calculator.params.handling_cost_per_bike
        self.calculator.episode_handling_cost += handling_cost
        self.calculator.total_bikes_moved += bikes_loaded + bikes_unloaded
    
    def get_episode_summary(self) -> Dict:
        """Get profit summary for episode."""
        summary = self.calculator.get_episode_summary()
        summary['vehicle_distances'] = dict(self.vehicle_distances)
        summary['total_vehicle_distance'] = sum(self.vehicle_distances.values())
        return summary


# Predefined parameter sets for different scenarios
CONSERVATIVE_PARAMS = ProfitParameters(
    trip_base_fare=0.75,
    trip_per_km_rate=0.50,
    trip_revenue=3.00,
    cost_per_km=0.60,
    cost_per_hour=25.00,
    handling_cost_per_bike=0.15,
    stop_cost=0.75,
    lost_demand_penalty=4.00
)

AGGRESSIVE_PARAMS = ProfitParameters(
    trip_base_fare=1.50,
    trip_per_km_rate=1.00,
    trip_revenue=4.00,
    cost_per_km=0.40,
    cost_per_hour=15.00,
    handling_cost_per_bike=0.05,
    stop_cost=0.25,
    lost_demand_penalty=6.00
)

HIGH_DEMAND_PENALTY_PARAMS = ProfitParameters(
    trip_base_fare=1.00,
    trip_per_km_rate=0.75,
    trip_revenue=3.50,
    cost_per_km=0.50,
    cost_per_hour=20.00,
    handling_cost_per_bike=0.10,
    stop_cost=0.50,
    lost_demand_penalty=10.00  # Heavy penalty for lost customers
)

# High travel cost scenario - shows difference between profit vs lost-demand DQN
HIGH_TRAVEL_COST_PARAMS = ProfitParameters(
    trip_base_fare=1.00,
    trip_per_km_rate=0.75,
    trip_revenue=3.50,
    cost_per_km=2.50,           # High truck travel cost
    cost_per_hour=20.00,
    handling_cost_per_bike=0.10,
    stop_cost=1.00,
    lost_demand_penalty=5.00
)


if __name__ == "__main__":
    # Example usage
    print("Profit Reward Calculator - Example")
    print("="*50)
    
    calc = ProfitRewardCalculator()
    
    # Simulate a few decision steps
    rewards = []
    
    # Step 1: Vehicle travels 2km, 1 lost rental, 3 successful trips
    r1 = calc.calculate_step_reward(
        successful_rentals=2, successful_returns=1,
        lost_rentals=1, lost_returns=0,
        distance_traveled_km=2.0, time_elapsed_hours=0.15,
        bikes_loaded=3, bikes_unloaded=0, made_stop=True
    )
    rewards.append(r1)
    print(f"Step 1 reward: ${r1:.2f}")
    
    # Step 2: Vehicle travels 1km, no lost demand, 2 successful trips
    r2 = calc.calculate_step_reward(
        successful_rentals=1, successful_returns=1,
        lost_rentals=0, lost_returns=0,
        distance_traveled_km=1.0, time_elapsed_hours=0.08,
        bikes_loaded=0, bikes_unloaded=2, made_stop=True
    )
    rewards.append(r2)
    print(f"Step 2 reward: ${r2:.2f}")
    
    # Step 3: Vehicle stays, processes customers
    r3 = calc.calculate_step_reward(
        successful_rentals=3, successful_returns=2,
        lost_rentals=0, lost_returns=1,
        distance_traveled_km=0.0, time_elapsed_hours=0.20,
        bikes_loaded=0, bikes_unloaded=0, made_stop=False
    )
    rewards.append(r3)
    print(f"Step 3 reward: ${r3:.2f}")
    
    print("\nEpisode Summary:")
    summary = calc.get_episode_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:.2f}" if 'cost' in key or 'revenue' in key or 'profit' in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
