"""
Training Steps Ablation Study
Evaluate impact of training timesteps on lost-demand DQN performance.
Tests: 0 (untrained), 10k, 20k, 50k, 100k steps
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator
from rl_algorithm.multi_agent_dqn import MultiAgentDQN


def evaluate_model(agent, gt_name, num_episodes, num_vehicles, vehicle_capacity):
    """Evaluate a model on test data."""
    base_dir = Path(__file__).parent.parent / 'data' / 'synthetic' / gt_name
    network_file = str(base_dir / f'{gt_name}_station_network.json')
    trips_file = str(base_dir / f'{gt_name}_trips_test.csv')
    
    results = []
    
    for ep in range(num_episodes):
        simulator = ContinuousTimeSimulator(
            network_file=network_file,
            trips_file=trips_file,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity
        )
        day = (ep % 50) + 1
        simulator.reset(day)
        
        while not simulator.is_done():
            vehicle_id, _ = simulator.get_next_decision_epoch()
            if vehicle_id is None:
                break
            
            state_dict = simulator.get_state()
            action_idx = agent.select_action(state_dict, vehicle_id, epsilon=0.0)
            action = agent.action_space.get_action(action_idx)
            simulator.execute_action(vehicle_id, action)
        
        metrics = simulator.get_metrics()
        results.append({
            'lost_demand_rate': metrics['total_lost_demand_rate'],
            'total_lost': metrics['total_lost_demand'],
            'total_trips': metrics['successful_rentals'] + metrics['successful_returns']
        })
    
    return {
        'avg_lost_demand': np.mean([r['lost_demand_rate'] for r in results]),
        'std_lost_demand': np.std([r['lost_demand_rate'] for r in results]),
        'avg_lost_trips': np.mean([r['total_lost'] for r in results]),
        'avg_total_trips': np.mean([r['total_trips'] for r in results])
    }


def train_and_evaluate(timesteps, gt_name, num_episodes, num_stations, num_vehicles, vehicle_capacity):
    """Train for specified timesteps and evaluate."""
    base_dir = Path(__file__).parent.parent / 'data' / 'synthetic' / gt_name
    network_file = str(base_dir / f'{gt_name}_station_network.json')
    trips_file = str(base_dir / f'{gt_name}_trips_train.csv')
    
    # Create agent
    agent = MultiAgentDQN(
        num_stations=num_stations,
        num_vehicles=num_vehicles,
        hidden_dim=128,
        learning_rate=1e-4,
        total_timesteps=max(timesteps, 1000)  # For epsilon schedule
    )
    
    if timesteps == 0:
        # Untrained - just evaluate
        print(f"\n{'='*50}")
        print(f"Evaluating UNTRAINED DQN (0 steps)")
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print(f"Training for {timesteps:,} timesteps...")
        print(f"{'='*50}")
        
        step = 0
        episode = 0
        
        while step < timesteps:
            simulator = ContinuousTimeSimulator(
                network_file=network_file,
                trips_file=trips_file,
                num_vehicles=num_vehicles,
                vehicle_capacity=vehicle_capacity
            )
            day = (episode % 100) + 1
            simulator.reset(day)
            episode += 1
            
            while not simulator.is_done() and step < timesteps:
                vehicle_id, reward = simulator.get_next_decision_epoch()
                if vehicle_id is None:
                    break
                
                state_dict = simulator.get_state()
                action_idx = agent.select_action(state_dict, vehicle_id)
                action = agent.action_space.get_action(action_idx)
                simulator.execute_action(vehicle_id, action)
                
                next_state_dict = simulator.get_state()
                agent.store_experience(state_dict, action_idx, reward, next_state_dict, False)
                
                if len(agent.replay_buffer) >= agent.batch_size:
                    agent.train_step()
                
                step += 1
            
            if step % 10000 == 0 or step == timesteps:
                print(f"  Step {step:,}/{timesteps:,}")
    
    # Evaluate
    print(f"Evaluating on {num_episodes} test episodes...")
    results = evaluate_model(agent, gt_name, num_episodes, num_vehicles, vehicle_capacity)
    
    return results


def run_ablation():
    """Run the full ablation study."""
    gt_name = 'GT0'
    num_episodes = 50
    num_stations = 10
    num_vehicles = 2
    vehicle_capacity = 15
    
    timestep_configs = [0, 10000, 20000, 50000, 100000]
    
    print("\n" + "="*60)
    print("TRAINING STEPS ABLATION STUDY")
    print("="*60)
    print(f"Configs: {timestep_configs}")
    print(f"Test episodes: {num_episodes}")
    
    all_results = {}
    
    for timesteps in timestep_configs:
        results = train_and_evaluate(
            timesteps, gt_name, num_episodes,
            num_stations, num_vehicles, vehicle_capacity
        )
        all_results[timesteps] = results
        
        print(f"\n  {timesteps:,} steps -> Lost Demand: {results['avg_lost_demand']:.2f}% ± {results['std_lost_demand']:.2f}%")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Training Steps vs Lost Demand")
    print("="*60)
    print(f"{'Steps':<12} {'Lost Demand':<15} {'Std Dev':<12} {'Lost Trips':<12}")
    print("-"*60)
    
    for timesteps in timestep_configs:
        r = all_results[timesteps]
        print(f"{timesteps:,}".ljust(12) + 
              f"{r['avg_lost_demand']:.2f}%".ljust(15) +
              f"±{r['std_lost_demand']:.2f}%".ljust(12) +
              f"{r['avg_lost_trips']:.1f}".ljust(12))
    
    # Save results
    output_file = Path(__file__).parent / 'results_ablation_steps.json'
    with open(output_file, 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    run_ablation()
