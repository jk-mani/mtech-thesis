"""
Evaluate all lost-demand DQN activation configurations on TEST data.
Reports lost demand percentage only.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np

from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator
from rl_algorithm.multi_agent_dqn import MultiAgentDQN


def evaluate_lostdemand_model(
    model_path,
    output_activation,
    gt_name='GT0',
    num_episodes=50,
    num_stations=10,
    num_vehicles=2,
    vehicle_capacity=15
):
    """Evaluate a single lost-demand DQN model on TEST data."""
    
    # Paths - USE TEST FILE
    base_dir = Path(__file__).parent.parent / 'data' / 'synthetic' / gt_name
    network_file = str(base_dir / f'{gt_name}_station_network.json')
    trips_file = str(base_dir / f'{gt_name}_trips_test.csv')
    
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
        
        while not simulator.is_done():
            vehicle_id, _ = simulator.get_next_decision_epoch()
            if vehicle_id is None:
                break
            
            # Greedy action selection
            state_dict = simulator.get_state()
            action_idx = agent.select_action(state_dict, vehicle_id, epsilon=0.0)
            action = agent.action_space.get_action(action_idx)
            simulator.execute_action(vehicle_id, action)
        
        metrics = simulator.get_metrics()
        results.append({
            'lost_demand_rate': metrics['total_lost_demand_rate'],
            'lost_rentals': metrics['lost_rentals'],
            'lost_returns': metrics['lost_returns'],
            'total_lost': metrics['total_lost_demand'],
            'successful_rentals': metrics['successful_rentals'],
            'successful_returns': metrics['successful_returns']
        })
    
    return results


def main():
    """Run evaluation for all activation configurations."""
    
    results_dir = Path(__file__).parent / 'results_GT0'
    
    experiments = [
        ('None', 'activation_none', None),
        ('ELU', 'activation_elu', 'elu'),
        ('Leaky ReLU', 'activation_leaky_relu', 'leaky_relu'),
        ('PReLU', 'activation_prelu', 'prelu'),
    ]
    
    print("=" * 70)
    print("LOST-DEMAND DQN EVALUATION ON TEST DATA (50 episodes)")
    print("=" * 70)
    
    all_results = {}
    
    for name, exp_dir, output_act in experiments:
        model_path = results_dir / exp_dir / 'GT0_multi_agent_dqn_final.pth'
        
        if not model_path.exists():
            print(f"\n⚠ Model not found for {name}: {model_path}")
            continue
        
        print(f"\n[{name}] Evaluating {model_path.name}...")
        
        results = evaluate_lostdemand_model(
            model_path=str(model_path),
            output_activation=output_act,
            num_episodes=50
        )
        
        lost_rates = [r['lost_demand_rate'] for r in results]
        total_lost = [r['total_lost'] for r in results]
        
        all_results[name] = {
            'avg_lost_rate': np.mean(lost_rates),
            'std_lost_rate': np.std(lost_rates),
            'min_lost_rate': np.min(lost_rates),
            'max_lost_rate': np.max(lost_rates),
            'avg_total_lost': np.mean(total_lost),
        }
        
        print(f"  Avg Lost Demand: {np.mean(lost_rates):.2f}% ± {np.std(lost_rates):.2f}%")
        print(f"  Range: [{np.min(lost_rates):.2f}%, {np.max(lost_rates):.2f}%]")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: TEST PERFORMANCE BY ACTIVATION FUNCTION")
    print("=" * 70)
    print(f"{'Activation':<15} {'Avg Lost Demand':>15} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    
    for name in ['None', 'ELU', 'Leaky ReLU', 'PReLU']:
        if name in all_results:
            r = all_results[name]
            print(f"{name:<15} {r['avg_lost_rate']:>14.2f}% {r['std_lost_rate']:>9.2f}% {r['min_lost_rate']:>9.2f}% {r['max_lost_rate']:>9.2f}%")
    
    # Find best
    best_lost = min(all_results.items(), key=lambda x: x[1]['avg_lost_rate'])
    
    print("-" * 70)
    print(f"Best (Lowest Lost Demand): {best_lost[0]} ({best_lost[1]['avg_lost_rate']:.2f}%)")
    print("=" * 70)
    
    # Save results
    output_file = results_dir / 'test_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    main()
