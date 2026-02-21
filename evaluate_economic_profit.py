"""
Evaluate economic sensitivity experiments with full profit analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np

from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator
from rl_algorithm.multi_agent_dqn import MultiAgentDQN


def calculate_distance(network_data, from_station, to_station):
    """Calculate distance between two stations."""
    stations = {s['id']: s for s in network_data['stations']}
    if from_station not in stations or to_station not in stations:
        return 0
    s1, s2 = stations[from_station], stations[to_station]
    lat1, lon1 = s1['latitude'], s1['longitude']
    lat2, lon2 = s2['latitude'], s2['longitude']
    dist = np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111
    return dist


def evaluate_config_with_profit(
    model_path,
    cost_per_km,
    lost_penalty,
    gt_name='GT0',
    num_episodes=50,
    trip_base_fare=1.0,
    trip_per_km=0.75
):
    """Evaluate a config with profit tracking."""
    
    base_dir = Path(__file__).parent.parent / 'data' / 'synthetic' / gt_name
    network_file = str(base_dir / f'{gt_name}_station_network.json')
    trips_file = str(base_dir / f'{gt_name}_trips_test.csv')
    
    with open(network_file) as f:
        network_data = json.load(f)
    
    simulator = ContinuousTimeSimulator(
        network_file=network_file,
        trips_file=trips_file,
        num_vehicles=2,
        vehicle_capacity=15
    )
    
    agent = MultiAgentDQN(
        num_stations=10,
        num_vehicles=2,
        hidden_activation='relu',
        output_activation='prelu',
        fill_levels=[0.10, 0.50, 0.90]
    )
    agent.load(model_path)
    
    results = []
    
    for ep in range(num_episodes):
        day = (ep % 50) + 1
        simulator.reset(day)
        
        episode_truck_distance = 0.0
        vehicle_positions = {0: 0, 1: 0}
        
        while not simulator.is_done():
            vehicle_id, _ = simulator.get_next_decision_epoch()
            if vehicle_id is None:
                break
            
            state_dict = simulator.get_state()
            action_idx = agent.select_action(state_dict, vehicle_id, epsilon=0.0)
            action = agent.action_space.get_action(action_idx)
            
            target_station = action[0]
            current_pos = vehicle_positions.get(vehicle_id, 0)
            dist = calculate_distance(network_data, current_pos, target_station)
            episode_truck_distance += dist
            vehicle_positions[vehicle_id] = target_station
            
            simulator.execute_action(vehicle_id, action)
        
        metrics = simulator.get_metrics()
        
        successful_trips = metrics['successful_rentals'] + metrics['successful_returns']
        lost_trips = metrics['total_lost_demand']
        
        avg_trip_km = 2.5
        revenue = successful_trips * (trip_base_fare + trip_per_km * avg_trip_km)
        truck_cost = episode_truck_distance * cost_per_km
        lost_cost = lost_trips * lost_penalty
        profit = revenue - truck_cost - lost_cost
        
        results.append({
            'lost_demand_rate': metrics['total_lost_demand_rate'],
            'lost_trips': lost_trips,
            'successful_trips': successful_trips,
            'revenue': revenue,
            'truck_distance_km': episode_truck_distance,
            'truck_cost': truck_cost,
            'lost_cost': lost_cost,
            'profit': profit
        })
    
    return results


def main():
    results_dir = Path(__file__).parent / 'results_economic_sensitivity'
    
    configs = [
        ('baseline', 1.00, 5.00),
        ('low_cost', 0.50, 5.00),
        ('high_cost', 2.00, 5.00),
        ('low_penalty', 1.00, 2.00),
        ('high_penalty', 1.00, 10.00),
        ('cheap_aggressive', 0.50, 10.00),
        ('expensive_conservative', 2.00, 2.00),
    ]
    
    all_results = {}
    
    print("=" * 70)
    print("ECONOMIC SENSITIVITY: PROFIT ANALYSIS")
    print("=" * 70)
    
    for name, cost_km, penalty in configs:
        model_path = results_dir / name / 'GT0_profit_dqn_best.pth'
        if not model_path.exists():
            model_path = results_dir / name / 'GT0_profit_dqn_final.pth'
        
        if not model_path.exists():
            print(f"⚠ Model not found for {name}")
            continue
        
        print(f"\n[{name}] cost=${cost_km}/km, penalty=${penalty}...")
        
        results = evaluate_config_with_profit(
            model_path=str(model_path),
            cost_per_km=cost_km,
            lost_penalty=penalty
        )
        
        all_results[name] = {
            'cost_per_km': cost_km,
            'lost_penalty': penalty,
            'avg_profit': np.mean([r['profit'] for r in results]),
            'std_profit': np.std([r['profit'] for r in results]),
            'avg_revenue': np.mean([r['revenue'] for r in results]),
            'avg_truck_cost': np.mean([r['truck_cost'] for r in results]),
            'avg_lost_cost': np.mean([r['lost_cost'] for r in results]),
            'avg_lost_rate': np.mean([r['lost_demand_rate'] for r in results]),
            'avg_truck_distance': np.mean([r['truck_distance_km'] for r in results])
        }
        
        print(f"  Profit: ${all_results[name]['avg_profit']:.2f}")
        print(f"  Lost Demand: {all_results[name]['avg_lost_rate']:.2f}%")
    
    # Print summary table
    print("\n" + "=" * 90)
    print("ECONOMIC SENSITIVITY PROFIT SUMMARY")
    print("=" * 90)
    print(f"{'Config':<25} {'Cost/km':>8} {'Penalty':>8} {'Profit':>12} {'Lost %':>10} {'Truck $':>10}")
    print("-" * 90)
    for name in all_results:
        r = all_results[name]
        print(f"{name:<25} ${r['cost_per_km']:>6.2f} ${r['lost_penalty']:>6.2f} ${r['avg_profit']:>10.2f} {r['avg_lost_rate']:>9.2f}% ${r['avg_truck_cost']:>8.2f}")
    
    # Save
    output_file = results_dir / 'profit_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    main()
