"""
Compare Profit-DQN vs Lost-Demand-DQN under different economic scenarios.
Shows how profit-based training leads to more efficient policies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np

from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator
from rl_algorithm.multi_agent_dqn import MultiAgentDQN


def evaluate_model_with_profit(
    model_path,
    model_name,
    gt_name='GT0',
    num_episodes=50,
    # Economic parameters
    trip_base_fare=1.00,
    trip_per_km_rate=0.75,
    cost_per_km=0.50,
    lost_demand_penalty=5.00,
    # Network params
    num_stations=10,
    num_vehicles=2,
    vehicle_capacity=15
):
    """Evaluate a model using profit metrics."""
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / gt_name
    network_file = str(base_dir / f'{gt_name}_station_network.json')
    trips_file = str(base_dir / f'{gt_name}_trips_test.csv')
    
    # Load distance matrix
    with open(network_file) as f:
        network_data = json.load(f)
    distance_matrix = np.array(network_data['distance_matrix'])
    
    # Initialize simulator
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
    
    results = []
    
    for ep in range(num_episodes):
        day = (ep % 50) + 1
        simulator.reset(day)
        
        last_positions = {vid: v.current_station for vid, v in simulator.vehicles.items()}
        episode_revenue = 0
        episode_truck_cost = 0
        episode_lost_cost = 0
        total_truck_km = 0
        
        last_rentals = simulator.total_successful_rentals
        last_returns = simulator.total_successful_returns
        last_lost = simulator.total_lost_rentals + simulator.total_lost_returns
        
        while not simulator.is_done():
            vehicle_id, _ = simulator.get_next_decision_epoch()
            if vehicle_id is None:
                break
            
            # Calculate metrics since last step
            new_rentals = simulator.total_successful_rentals - last_rentals
            new_returns = simulator.total_successful_returns - last_returns
            new_lost = (simulator.total_lost_rentals + simulator.total_lost_returns) - last_lost
            
            # Revenue from trips (use distance between origin/destination)
            # For simplicity, use average trip distance from distance matrix
            avg_trip_dist = np.mean(distance_matrix[distance_matrix > 0])
            trip_revenue = (new_rentals + new_returns) * (trip_base_fare + avg_trip_dist * trip_per_km_rate)
            episode_revenue += trip_revenue
            
            # Truck travel cost
            vehicle = simulator.vehicles[vehicle_id]
            last_station = last_positions.get(vehicle_id, vehicle.current_station)
            if last_station != vehicle.current_station:
                truck_dist = distance_matrix[last_station - 1][vehicle.current_station - 1]
                episode_truck_cost += truck_dist * cost_per_km
                total_truck_km += truck_dist
            
            # Lost demand penalty
            episode_lost_cost += new_lost * lost_demand_penalty
            
            last_positions[vehicle_id] = vehicle.current_station
            last_rentals = simulator.total_successful_rentals
            last_returns = simulator.total_successful_returns
            last_lost = simulator.total_lost_rentals + simulator.total_lost_returns
            
            # Greedy action
            state_dict = simulator.get_state()
            action_idx = agent.select_action(state_dict, vehicle_id, epsilon=0.0)
            action = agent.action_space.get_action(action_idx)
            simulator.execute_action(vehicle_id, action)
        
        metrics = simulator.get_metrics()
        profit = episode_revenue - episode_truck_cost - episode_lost_cost
        
        results.append({
            'profit': profit,
            'revenue': episode_revenue,
            'truck_cost': episode_truck_cost,
            'lost_cost': episode_lost_cost,
            'truck_km': total_truck_km,
            'lost_demand_rate': metrics['total_lost_demand_rate']
        })
    
    return {
        'model_name': model_name,
        'avg_profit': np.mean([r['profit'] for r in results]),
        'std_profit': np.std([r['profit'] for r in results]),
        'avg_revenue': np.mean([r['revenue'] for r in results]),
        'avg_truck_cost': np.mean([r['truck_cost'] for r in results]),
        'avg_lost_cost': np.mean([r['lost_cost'] for r in results]),
        'avg_truck_km': np.mean([r['truck_km'] for r in results]),
        'avg_lost_demand': np.mean([r['lost_demand_rate'] for r in results]),
        'episodes': results
    }


def run_comparison(
    profit_model_path,
    lost_demand_model_path,
    gt_name='GT0',
    num_episodes=50,
    num_stations=10,
    num_vehicles=2,
    vehicle_capacity=15
):
    """Compare two models under different economic scenarios."""
    
    print("\n" + "="*70)
    print("COMPARING: Profit-DQN vs Lost-Demand-DQN")
    print("="*70)
    
    scenarios = [
        {
            'name': 'Standard ($0.50/km truck cost)',
            'cost_per_km': 0.50,
            'lost_demand_penalty': 5.00
        },
        {
            'name': 'High Truck Cost ($2.50/km)',
            'cost_per_km': 2.50,
            'lost_demand_penalty': 5.00
        },
        {
            'name': 'Very High Truck Cost ($5.00/km)',
            'cost_per_km': 5.00,
            'lost_demand_penalty': 5.00
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'='*70}")
        
        # Evaluate Profit-DQN
        print(f"\nEvaluating Profit-DQN...")
        profit_results = evaluate_model_with_profit(
            profit_model_path, "Profit-DQN", gt_name, num_episodes,
            cost_per_km=scenario['cost_per_km'],
            lost_demand_penalty=scenario['lost_demand_penalty'],
            num_stations=num_stations,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity
        )
        
        # Evaluate Lost-Demand-DQN
        print(f"Evaluating Lost-Demand-DQN...")
        ld_results = evaluate_model_with_profit(
            lost_demand_model_path, "Lost-Demand-DQN", gt_name, num_episodes,
            cost_per_km=scenario['cost_per_km'],
            lost_demand_penalty=scenario['lost_demand_penalty'],
            num_stations=num_stations,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity
        )
        
        # Print comparison
        print(f"\n{'Metric':<25} {'Profit-DQN':>15} {'Lost-Demand-DQN':>18} {'Diff':>12}")
        print("-"*70)
        
        metrics = [
            ('Avg Profit', 'avg_profit', '$'),
            ('Avg Revenue', 'avg_revenue', '$'),
            ('Avg Truck Cost', 'avg_truck_cost', '$'),
            ('Avg Lost Penalty', 'avg_lost_cost', '$'),
            ('Avg Truck km', 'avg_truck_km', 'km'),
            ('Lost Demand %', 'avg_lost_demand', '%')
        ]
        
        for name, key, unit in metrics:
            p_val = profit_results[key]
            ld_val = ld_results[key]
            diff = p_val - ld_val
            
            if unit == '$':
                print(f"{name:<25} {unit}{p_val:>14.2f} {unit}{ld_val:>17.2f} {diff:>+12.2f}")
            elif unit == '%':
                print(f"{name:<25} {p_val:>14.2f}{unit} {ld_val:>17.2f}{unit} {diff:>+11.2f}{unit}")
            else:
                print(f"{name:<25} {p_val:>14.2f} {unit} {ld_val:>14.2f} {unit} {diff:>+10.2f} {unit}")
        
        # Winner
        profit_diff_pct = ((profit_results['avg_profit'] - ld_results['avg_profit']) / abs(ld_results['avg_profit'])) * 100
        print("-"*70)
        if profit_results['avg_profit'] > ld_results['avg_profit']:
            print(f"✅ WINNER: Profit-DQN (+{profit_diff_pct:.1f}% profit)")
        else:
            print(f"⚠️ WINNER: Lost-Demand-DQN ({profit_diff_pct:.1f}% profit diff)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--profit-model', required=True, help='Path to profit-trained model')
    parser.add_argument('--ld-model', required=True, help='Path to lost-demand-trained model')
    parser.add_argument('--gt', default='GT0')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--num-stations', type=int, default=10)
    parser.add_argument('--num-vehicles', type=int, default=2)
    parser.add_argument('--vehicle-capacity', type=int, default=15)
    args = parser.parse_args()
    
    run_comparison(
        profit_model_path=args.profit_model,
        lost_demand_model_path=args.ld_model,
        gt_name=args.gt,
        num_episodes=args.episodes,
        num_stations=args.num_stations,
        num_vehicles=args.num_vehicles,
        vehicle_capacity=args.vehicle_capacity
    )
