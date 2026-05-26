"""
Evaluation Script for Profit-Based Multi-Agent DQN.

Evaluates trained profit-based agents on test data, providing detailed
economic analysis including:
- Net profit per episode
- Revenue breakdown
- Operational cost breakdown
- Comparison with lost-demand baseline
- ROI analysis

Usage:
    python evaluate_profit.py --gt GT0 --model results_profit/GT0/GT0_profit_dqn_final.pth
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

from multi_agent_dqn import MultiAgentDQN
from profit_simulator import ProfitSimulator
from profit_reward import ProfitParameters


def evaluate_profit_agent(
    model_path,
    gt_name='GT0',
    num_test_episodes=50,
    profit_params=None,
    num_stations=10,
    num_vehicles=2,
    vehicle_capacity=15,
    fill_levels=None,
    hidden_activation='relu',
    output_activation=None,
    output_dir=None,
    epsilon=0.0
):
    """
    Evaluate profit-based DQN agent on test data.
    
    Args:
        model_path: Path to trained model
        gt_name: Ground truth network
        num_test_episodes: Number of test episodes
        profit_params: Economic parameters
        num_stations: Number of stations
        num_vehicles: Number of vehicles
        vehicle_capacity: Vehicle capacity
        fill_levels: Fill level options
        hidden_activation: Hidden layer activation
        output_activation: Output layer activation
        output_dir: Directory to save results
        epsilon: Exploration rate (0 for pure exploitation)
        
    Returns:
        dict: Evaluation results
    """
    print("\n" + "="*70)
    print("PROFIT-BASED DQN EVALUATION")
    print("="*70)
    
    # Setup
    if profit_params is None:
        profit_params = ProfitParameters()
    
    fill_levels_used = fill_levels if fill_levels is not None else [0.10, 0.50, 0.90]
    
    if output_dir is None:
        output_dir = Path(model_path).parent / 'evaluation'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Network: {gt_name}")
    print(f"  Test episodes: {num_test_episodes}")
    print(f"  Epsilon: {epsilon}")
    
    # Data paths
    base_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / gt_name
    network_file = str(base_dir / f'{gt_name}_station_network.json')
    trips_file = str(base_dir / f'{gt_name}_trips_test.csv')
    
    print(f"\n[1/3] Initializing...")
    
    # Initialize simulator
    simulator = ProfitSimulator(
        network_file=network_file,
        trips_file=trips_file,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        profit_params=profit_params
    )
    
    # Initialize agent
    agent = MultiAgentDQN(
        num_stations=num_stations,
        num_vehicles=num_vehicles,
        hidden_dim=1024,
        buffer_capacity=10000,
        batch_size=256,
        learning_rate=2.5e-4,
        gamma=0.99,
        fill_levels=fill_levels_used,
        hidden_activation=hidden_activation,
        output_activation=output_activation
    )
    
    # Load model
    agent.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    # Results storage
    results = {
        'episode': [],
        'profit': [],
        'revenue': [],
        'operational_cost': [],
        'travel_cost': [],
        'time_cost': [],
        'handling_cost': [],
        'stop_cost': [],
        'lost_demand_cost': [],
        'lost_demand': [],
        'lost_demand_rate': [],
        'successful_trips': [],
        'total_distance_km': [],
        'total_bikes_moved': [],
        'decisions': []
    }
    
    print(f"\n[2/3] Running {num_test_episodes} test episodes...")
    
    for ep in range(num_test_episodes):
        # Reset
        state_dict = simulator.reset()
        episode_done = False
        decisions = 0
        
        while not episode_done:
            result = simulator.step()
            
            if result is None:
                episode_done = True
                break
            
            vehicle_id, profit_reward = result
            
            # Get state
            state = agent.state_encoder.encode(simulator.get_state(), vehicle_id)
            
            # Select action (greedy or with small epsilon)
            action_idx = agent.select_action(state, epsilon)
            
            # Decode action
            station_idx = action_idx // len(fill_levels_used)
            fill_idx = action_idx % len(fill_levels_used)
            next_station = station_idx + 1
            
            # Execute
            simulator.execute_action(vehicle_id, (next_station, fill_idx))
            decisions += 1
        
        # Get episode metrics
        metrics = simulator.get_metrics()
        
        # Store results
        results['episode'].append(ep + 1)
        results['profit'].append(metrics['net_profit'])
        results['revenue'].append(metrics['revenue'])
        results['operational_cost'].append(metrics['total_operational_cost'])
        results['travel_cost'].append(metrics['travel_cost'])
        results['time_cost'].append(metrics['time_cost'])
        results['handling_cost'].append(metrics['handling_cost'])
        results['stop_cost'].append(metrics['stop_cost'])
        results['lost_demand_cost'].append(metrics['lost_demand_cost'])
        results['lost_demand'].append(metrics['total_lost_demand'])
        results['lost_demand_rate'].append(metrics['lost_demand_rate'])
        results['successful_trips'].append(metrics['successful_trips'])
        results['total_distance_km'].append(metrics['total_distance_km'])
        results['total_bikes_moved'].append(metrics['total_bikes_moved'])
        results['decisions'].append(decisions)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_test_episodes}: "
                  f"Profit=${metrics['net_profit']:.2f}, "
                  f"Lost={metrics['lost_demand_rate']:.1f}%")
    
    print(f"\n[3/3] Computing summary statistics...")
    
    # Summary statistics
    summary = {
        # Profit metrics
        'avg_profit': np.mean(results['profit']),
        'std_profit': np.std(results['profit']),
        'min_profit': np.min(results['profit']),
        'max_profit': np.max(results['profit']),
        'median_profit': np.median(results['profit']),
        
        # Revenue
        'avg_revenue': np.mean(results['revenue']),
        'total_revenue': np.sum(results['revenue']),
        
        # Costs
        'avg_operational_cost': np.mean(results['operational_cost']),
        'avg_travel_cost': np.mean(results['travel_cost']),
        'avg_time_cost': np.mean(results['time_cost']),
        'avg_handling_cost': np.mean(results['handling_cost']),
        'avg_stop_cost': np.mean(results['stop_cost']),
        'avg_lost_demand_cost': np.mean(results['lost_demand_cost']),
        
        # Service quality
        'avg_lost_demand': np.mean(results['lost_demand']),
        'avg_lost_demand_rate': np.mean(results['lost_demand_rate']),
        'std_lost_demand_rate': np.std(results['lost_demand_rate']),
        
        # Operational
        'avg_successful_trips': np.mean(results['successful_trips']),
        'avg_distance_km': np.mean(results['total_distance_km']),
        'avg_bikes_moved': np.mean(results['total_bikes_moved']),
        'avg_decisions': np.mean(results['decisions']),
        
        # Efficiency
        'profit_per_trip': np.mean(results['profit']) / max(1, np.mean(results['successful_trips'])),
        'cost_per_km': np.mean(results['travel_cost']) / max(0.1, np.mean(results['total_distance_km'])),
        'profit_margin': (np.mean(results['profit']) / max(1, np.mean(results['revenue']))) * 100
    }
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n📊 Profit Metrics ({num_test_episodes} episodes):")
    print(f"  Average Profit: ${summary['avg_profit']:.2f} ± ${summary['std_profit']:.2f}")
    print(f"  Best Episode: ${summary['max_profit']:.2f}")
    print(f"  Worst Episode: ${summary['min_profit']:.2f}")
    print(f"  Median: ${summary['median_profit']:.2f}")
    print(f"  Profit Margin: {summary['profit_margin']:.1f}%")
    
    print(f"\n💰 Revenue & Costs (averages):")
    print(f"  Revenue: ${summary['avg_revenue']:.2f}")
    print(f"  Operational Cost: ${summary['avg_operational_cost']:.2f}")
    print(f"    - Travel: ${summary['avg_travel_cost']:.2f}")
    print(f"    - Time/Labor: ${summary['avg_time_cost']:.2f}")
    print(f"    - Handling: ${summary['avg_handling_cost']:.2f}")
    print(f"    - Stops: ${summary['avg_stop_cost']:.2f}")
    print(f"  Lost Demand Cost: ${summary['avg_lost_demand_cost']:.2f}")
    
    print(f"\n🎯 Service Quality:")
    print(f"  Lost Demand Rate: {summary['avg_lost_demand_rate']:.2f}% ± {summary['std_lost_demand_rate']:.2f}%")
    print(f"  Successful Trips: {summary['avg_successful_trips']:.1f}")
    
    print(f"\n🚚 Operations:")
    print(f"  Avg Distance: {summary['avg_distance_km']:.2f} km")
    print(f"  Avg Bikes Moved: {summary['avg_bikes_moved']:.1f}")
    print(f"  Avg Decisions: {summary['avg_decisions']:.1f}")
    
    # Save results
    full_results = {
        'summary': summary,
        'episodes': results,
        'config': {
            'model_path': str(model_path),
            'gt_name': gt_name,
            'num_episodes': num_test_episodes,
            'profit_params': {
                'trip_revenue': profit_params.trip_revenue,
                'cost_per_km': profit_params.cost_per_km,
                'cost_per_hour': profit_params.cost_per_hour,
                'handling_cost_per_bike': profit_params.handling_cost_per_bike,
                'lost_demand_penalty': profit_params.lost_demand_penalty
            }
        }
    }
    
    results_file = output_dir / f'{gt_name}_profit_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")
    
    # Generate plots
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{gt_name} Profit-Based DQN Evaluation', fontsize=16, fontweight='bold')
        
        # Profit distribution
        axes[0, 0].hist(results['profit'], bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(summary['avg_profit'], color='red', linestyle='--', 
                          linewidth=2, label=f"Mean: ${summary['avg_profit']:.2f}")
        axes[0, 0].set_xlabel('Net Profit ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Profit Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cost breakdown pie chart
        cost_labels = ['Travel', 'Time/Labor', 'Handling', 'Stops', 'Lost Demand']
        cost_values = [
            summary['avg_travel_cost'],
            summary['avg_time_cost'],
            summary['avg_handling_cost'],
            summary['avg_stop_cost'],
            summary['avg_lost_demand_cost']
        ]
        axes[0, 1].pie(cost_values, labels=cost_labels, autopct='%1.1f%%', 
                       colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff6666'])
        axes[0, 1].set_title('Cost Breakdown')
        
        # Profit over episodes
        axes[1, 0].plot(results['episode'], results['profit'], 'g-', alpha=0.7)
        axes[1, 0].axhline(summary['avg_profit'], color='red', linestyle='--', label='Mean')
        axes[1, 0].fill_between(results['episode'], 
                                 summary['avg_profit'] - summary['std_profit'],
                                 summary['avg_profit'] + summary['std_profit'],
                                 alpha=0.2, color='green')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Net Profit ($)')
        axes[1, 0].set_title('Profit per Episode')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Revenue vs Cost scatter
        axes[1, 1].scatter(results['revenue'], results['operational_cost'], 
                           c=results['profit'], cmap='RdYlGn', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Revenue ($)')
        axes[1, 1].set_ylabel('Operational Cost ($)')
        axes[1, 1].set_title('Revenue vs Cost (color = profit)')
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Profit ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_dir / f'{gt_name}_profit_evaluation_results.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Plots saved to {plot_file}")
        
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    
    return full_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Profit-Based DQN')
    
    parser.add_argument('--gt', type=str, default='GT0', choices=['GT0', 'GT1', 'GT2'])
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=50, help='Number of test episodes')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--epsilon', type=float, default=0.0, help='Exploration rate')
    
    # Environment config
    parser.add_argument('--num-stations', type=int, default=10)
    parser.add_argument('--num-vehicles', type=int, default=2)
    parser.add_argument('--vehicle-capacity', type=int, default=15)
    
    # Profit parameters (should match training)
    parser.add_argument('--trip-revenue', type=float, default=3.50)
    parser.add_argument('--cost-per-km', type=float, default=0.50)
    parser.add_argument('--cost-per-hour', type=float, default=20.00)
    parser.add_argument('--handling-cost', type=float, default=0.10)
    parser.add_argument('--lost-demand-penalty', type=float, default=5.00)
    
    # Network architecture (must match training)
    parser.add_argument('--fill-levels', type=str, default='10,50,90')
    parser.add_argument('--hidden-activation', type=str, default='relu')
    parser.add_argument('--output-activation', type=str, default=None)
    
    args = parser.parse_args()
    
    # Build profit params
    profit_params = ProfitParameters(
        trip_revenue=args.trip_revenue,
        cost_per_km=args.cost_per_km,
        cost_per_hour=args.cost_per_hour,
        handling_cost_per_bike=args.handling_cost,
        lost_demand_penalty=args.lost_demand_penalty
    )
    
    # Parse fill levels
    fill_levels = [float(x)/100.0 for x in args.fill_levels.split(',')]
    
    # Parse output activation
    output_activation = None if args.output_activation in ['none', None] else args.output_activation
    
    # Evaluate
    results = evaluate_profit_agent(
        model_path=args.model,
        gt_name=args.gt,
        num_test_episodes=args.episodes,
        profit_params=profit_params,
        num_stations=args.num_stations,
        num_vehicles=args.num_vehicles,
        vehicle_capacity=args.vehicle_capacity,
        fill_levels=fill_levels,
        hidden_activation=args.hidden_activation,
        output_activation=output_activation,
        output_dir=args.output_dir,
        epsilon=args.epsilon
    )
