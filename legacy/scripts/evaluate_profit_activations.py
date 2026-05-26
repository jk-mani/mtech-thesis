"""
Evaluate all profit-based DQN activation configurations on TEST data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np

from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator
from rl_algorithm.multi_agent_dqn import MultiAgentDQN


def evaluate_profit_model(
    model_path,
    output_activation,
    gt_name='GT0',
    num_episodes=50,
    trip_base_fare=1.00,
    trip_per_km_rate=0.75,
    cost_per_km=1.00,
    lost_demand_penalty=5.00,
    num_stations=10,
    num_vehicles=2,
    vehicle_capacity=15
):
    """Evaluate a single profit-DQN model on TEST data."""
    
    # Paths - USE TEST FILE
    base_dir = Path(__file__).parent.parent / 'data' / 'synthetic' / gt_name
    network_file = str(base_dir / f'{gt_name}_station_network.json')
    trips_file = str(base_dir / f'{gt_name}_trips_test.csv')
    
    # Load distance matrix
    with open(network_file) as f:
        network_data = json.load(f)
    distance_matrix = np.array(network_data['distance_matrix'])
    
    # Load station coordinates for trip distance calculation
    stations = network_data['stations']
    
    # Initialize simulator with TEST data
    simulator = ContinuousTimeSimulator(
        network_file=network_file,
        trips_file=trips_file,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity
    )
    
    # Initialize agent with correct activation
    agent = MultiAgentDQN(
        num_stations=num_stations,
        num_vehicles=num_vehicles,
        hidden_activation='relu',
        output_activation=output_activation,
        fill_levels=[0.10, 0.50, 0.90]
    )
    agent.load(model_path)
    
    results = []
    
    for ep in range(num_episodes):
        day = (ep % 50) + 1
        simulator.reset(day)
        
        last_positions = {vid: v.current_station for vid, v in simulator.vehicles.items()}
        episode_profit = 0
        episode_revenue = 0
        episode_truck_cost = 0
        episode_lost_cost = 0
        
        last_rentals = simulator.total_successful_rentals
        last_returns = simulator.total_successful_returns
        last_lost = simulator.total_lost_rentals + simulator.total_lost_returns
        
        while not simulator.is_done():
            vehicle_id, _ = simulator.get_next_decision_epoch()
            if vehicle_id is None:
                break
            
            # Calculate profit from new trips (distance-based revenue)
            new_trips = (simulator.total_successful_rentals - last_rentals) + \
                       (simulator.total_successful_returns - last_returns)
            new_lost = (simulator.total_lost_rentals + simulator.total_lost_returns) - last_lost
            
            # Approximate trip revenue using average trip distance
            avg_trip_distance = 2.0  # Approx avg km per trip
            revenue = new_trips * (trip_base_fare + trip_per_km_rate * avg_trip_distance)
            
            # Truck travel cost
            vehicle = simulator.vehicles[vehicle_id]
            last_station = last_positions.get(vehicle_id, vehicle.current_station)
            if last_station != vehicle.current_station:
                dist = distance_matrix[last_station - 1][vehicle.current_station - 1]
                truck_cost = dist * cost_per_km
            else:
                truck_cost = 0
            
            lost_cost = new_lost * lost_demand_penalty
            
            episode_profit += revenue - truck_cost - lost_cost
            episode_revenue += revenue
            episode_truck_cost += truck_cost
            episode_lost_cost += lost_cost
            
            last_positions[vehicle_id] = vehicle.current_station
            last_rentals = simulator.total_successful_rentals
            last_returns = simulator.total_successful_returns
            last_lost = simulator.total_lost_rentals + simulator.total_lost_returns
            
            # Greedy action selection
            state_dict = simulator.get_state()
            action_idx = agent.select_action(state_dict, vehicle_id, epsilon=0.0)
            action = agent.action_space.get_action(action_idx)
            simulator.execute_action(vehicle_id, action)
        
        metrics = simulator.get_metrics()
        results.append({
            'profit': episode_profit,
            'revenue': episode_revenue,
            'truck_cost': episode_truck_cost,
            'lost_cost': episode_lost_cost,
            'lost_demand_rate': metrics['total_lost_demand_rate'],
            'successful_trips': metrics['successful_rentals'] + metrics['successful_returns'],
            'lost_demand': metrics['total_lost_demand']
        })
    
    return results


def main():
    """Run evaluation for all activation configurations."""
    
    results_dir = Path(__file__).parent / 'results_profit_GT0'
    
    experiments = [
        ('None', 'activation_none', None),
        ('ELU', 'activation_elu', 'elu'),
        ('Leaky ReLU', 'activation_leaky_relu', 'leaky_relu'),
        ('PReLU', 'activation_prelu', 'prelu'),
    ]
    
    print("=" * 70)
    print("PROFIT-DQN EVALUATION ON TEST DATA (50 episodes)")
    print("=" * 70)
    print("Economic Parameters:")
    print("  Trip Revenue: $1.00 + $0.75/km")
    print("  Truck Cost: $1.00/km")
    print("  Lost Demand Penalty: $5.00/trip")
    print("=" * 70)
    
    all_results = {}
    
    for name, exp_dir, output_act in experiments:
        model_path = results_dir / exp_dir / 'GT0_profit_dqn_best.pth'
        
        if not model_path.exists():
            print(f"\n⚠ Model not found for {name}: {model_path}")
            continue
        
        print(f"\n[{name}] Evaluating {model_path.name}...")
        
        results = evaluate_profit_model(
            model_path=str(model_path),
            output_activation=output_act,
            num_episodes=50
        )
        
        profits = [r['profit'] for r in results]
        lost_rates = [r['lost_demand_rate'] for r in results]
        revenues = [r['revenue'] for r in results]
        truck_costs = [r['truck_cost'] for r in results]
        lost_costs = [r['lost_cost'] for r in results]
        
        all_results[name] = {
            'avg_profit': np.mean(profits),
            'std_profit': np.std(profits),
            'avg_lost_rate': np.mean(lost_rates),
            'std_lost_rate': np.std(lost_rates),
            'avg_revenue': np.mean(revenues),
            'avg_truck_cost': np.mean(truck_costs),
            'avg_lost_cost': np.mean(lost_costs),
            'min_profit': np.min(profits),
            'max_profit': np.max(profits)
        }
        
        print(f"  Avg Profit: ${np.mean(profits):.2f} ± ${np.std(profits):.2f}")
        print(f"  Avg Lost Demand: {np.mean(lost_rates):.2f}% ± {np.std(lost_rates):.2f}%")
        print(f"  Range: [${np.min(profits):.2f}, ${np.max(profits):.2f}]")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: TEST PERFORMANCE BY ACTIVATION FUNCTION")
    print("=" * 70)
    print(f"{'Activation':<15} {'Avg Profit':>12} {'Std':>8} {'Lost Demand':>12} {'Std':>8}")
    print("-" * 70)
    
    for name in ['None', 'ELU', 'Leaky ReLU', 'PReLU']:
        if name in all_results:
            r = all_results[name]
            print(f"{name:<15} ${r['avg_profit']:>10.2f} ${r['std_profit']:>6.2f} {r['avg_lost_rate']:>10.2f}% {r['std_lost_rate']:>6.2f}%")
    
    # Find best
    best_profit = max(all_results.items(), key=lambda x: x[1]['avg_profit'])
    best_lost = min(all_results.items(), key=lambda x: x[1]['avg_lost_rate'])
    
    print("-" * 70)
    print(f"Best Profit:      {best_profit[0]} (${best_profit[1]['avg_profit']:.2f})")
    print(f"Best Lost Demand: {best_lost[0]} ({best_lost[1]['avg_lost_rate']:.2f}%)")
    print("=" * 70)
    
    # Save results
    output_file = results_dir / 'test_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    main()
