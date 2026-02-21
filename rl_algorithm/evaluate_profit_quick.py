"""
Quick profit evaluation of DQN model on TEST data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np

from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator
from rl_algorithm.multi_agent_dqn import MultiAgentDQN


def evaluate_profit_test(
    model_path,
    gt_name='GT0',
    num_episodes=50,
    trip_revenue=3.50,
    cost_per_km=0.50,
    lost_demand_penalty=5.00,
    num_stations=10,
    num_vehicles=2,
    vehicle_capacity=15
):
    """Evaluate DQN on TEST data with profit metrics."""
    
    print(f"\n{'='*60}")
    print(f"PROFIT EVALUATION ON TEST DATA: {gt_name}")
    print(f"{'='*60}")
    
    # Paths - USE TEST FILE
    base_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / gt_name
    network_file = str(base_dir / f'{gt_name}_station_network.json')
    trips_file = str(base_dir / f'{gt_name}_trips_test.csv')  # TEST DATA
    
    print(f"Using TEST data: {trips_file}")
    
    # Load distance matrix
    with open(network_file) as f:
        network_data = json.load(f)
    distance_matrix = np.array(network_data['distance_matrix'])
    
    # Initialize simulator with TEST data
    simulator = ContinuousTimeSimulator(
        network_file=network_file,
        trips_file=trips_file,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity
    )
    
    # Initialize and load agent
    agent = MultiAgentDQN(
        num_stations=num_stations,
        num_vehicles=num_vehicles,
        fill_levels=[0.10, 0.50, 0.90]
    )
    agent.load(model_path)
    print(f"✓ Model loaded: {model_path}")
    
    results = []
    
    print(f"\nRunning {num_episodes} TEST episodes...")
    for ep in range(num_episodes):
        day = (ep % 50) + 1
        simulator.reset(day)
        
        last_positions = {vid: v.current_station for vid, v in simulator.vehicles.items()}
        episode_profit = 0
        episode_revenue = 0
        episode_cost = 0
        
        last_rentals = simulator.total_successful_rentals
        last_returns = simulator.total_successful_returns
        last_lost = simulator.total_lost_rentals + simulator.total_lost_returns
        
        while not simulator.is_done():
            vehicle_id, _ = simulator.get_next_decision_epoch()
            if vehicle_id is None:
                break
            
            # Calculate profit components
            new_trips = (simulator.total_successful_rentals - last_rentals) + \
                       (simulator.total_successful_returns - last_returns)
            new_lost = (simulator.total_lost_rentals + simulator.total_lost_returns) - last_lost
            
            revenue = new_trips * trip_revenue
            
            vehicle = simulator.vehicles[vehicle_id]
            last_station = last_positions.get(vehicle_id, vehicle.current_station)
            if last_station != vehicle.current_station:
                dist = distance_matrix[last_station - 1][vehicle.current_station - 1]
                cost = dist * cost_per_km
            else:
                cost = 0
            
            lost_cost = new_lost * lost_demand_penalty
            
            episode_profit += revenue - cost - lost_cost
            episode_revenue += revenue
            episode_cost += cost + lost_cost
            
            last_positions[vehicle_id] = vehicle.current_station
            last_rentals = simulator.total_successful_rentals
            last_returns = simulator.total_successful_returns
            last_lost = simulator.total_lost_rentals + simulator.total_lost_returns
            
            # Greedy action selection (epsilon=0)
            state_dict = simulator.get_state()
            action_idx = agent.select_action(state_dict, vehicle_id, epsilon=0.0)
            action = agent.action_space.get_action(action_idx)
            simulator.execute_action(vehicle_id, action)
        
        metrics = simulator.get_metrics()
        results.append({
            'profit': episode_profit,
            'revenue': episode_revenue,
            'cost': episode_cost,
            'lost_demand_rate': metrics['total_lost_demand_rate']
        })
        
        if (ep + 1) % 10 == 0:
            print(f"  Ep {ep+1}: Profit=${episode_profit:.2f}, Lost={metrics['total_lost_demand_rate']:.1f}%")
    
    # Summary
    profits = [r['profit'] for r in results]
    lost_rates = [r['lost_demand_rate'] for r in results]
    
    print(f"\n{'='*60}")
    print("DQN TEST RESULTS (50 test episodes)")
    print(f"{'='*60}")
    print(f"Avg Profit: ${np.mean(profits):.2f} ± ${np.std(profits):.2f}")
    print(f"Avg Lost Demand: {np.mean(lost_rates):.2f}% ± {np.std(lost_rates):.2f}%")
    print(f"Best: ${np.max(profits):.2f}, Worst: ${np.min(profits):.2f}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', default='GT0')
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--num-stations', type=int, default=10)
    parser.add_argument('--num-vehicles', type=int, default=2)
    parser.add_argument('--vehicle-capacity', type=int, default=15)
    args = parser.parse_args()
    
    evaluate_profit_test(
        model_path=args.model,
        gt_name=args.gt,
        num_episodes=args.episodes,
        num_stations=args.num_stations,
        num_vehicles=args.num_vehicles,
        vehicle_capacity=args.vehicle_capacity
    )
