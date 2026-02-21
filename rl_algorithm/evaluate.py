"""
Evaluation script for trained Multi-Agent DQN.

Evaluates the trained agent on the TEST SET (50 unseen days).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator
from rl_algorithm.multi_agent_dqn import MultiAgentDQN


def evaluate_agent(
    model_path,
    gt_name='GT1',
    num_test_episodes=50,
    epsilon=0.0,  # No exploration during evaluation
    output_dir=None,
    heuristic_routing=False,
    num_stations=60,
    num_vehicles=4,
    vehicle_capacity=40,
    hidden_activation='relu',
    output_activation=None,
    fill_levels=None
):
    """
    Evaluate trained agent on TEST SET.
    
    Args:
        model_path: Path to trained model (.pth file)
        gt_name: Ground truth name ('GT0', 'GT1', or 'GT2')
        num_test_episodes: Number of test episodes (default 50)
        epsilon: Exploration rate (0.0 = pure exploitation)
        output_dir: Directory to save results
        heuristic_routing: Use simplified action space with heuristic routing
        num_stations: Number of stations (default: 60, GT0: 10)
        num_vehicles: Number of vehicles (default: 4, GT0: 2)
        vehicle_capacity: Vehicle bike capacity (default: 40, GT0: 15)
    
    Returns:
        dict: Evaluation results
    """
    print("\n" + "="*70)
    print(f"EVALUATING MULTI-AGENT DQN ON TEST SET - {gt_name}")
    print("="*70)
    print(f"\nModel: {model_path}")
    print(f"Test episodes: {num_test_episodes}")
    print(f"Epsilon: {epsilon} (0 = pure exploitation, no exploration)")
    
    # Setup output directory - auto-detect from model path
    if output_dir is None:
        # Extract the model's parent directory (e.g., results/GT1_fill_10_50_90/)
        model_dir = Path(model_path).parent
        # Save evaluation results in the same directory as the model
        output_dir = model_dir / 'evaluation'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Initialize simulator with TEST data
    print(f"\n[1/4] Initializing simulator with TEST data...")
    print(f"  Data path: data/synthetic/{gt_name}")
    base_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / gt_name
    print(f"  Network file: {base_dir / f'{gt_name}_station_network.json'}")
    print(f"  Trips file: {base_dir / f'{gt_name}_trips_test.csv'}")
    
    simulator = ContinuousTimeSimulator(
        network_file=str(base_dir / f'{gt_name}_station_network.json'),
        trips_file=str(base_dir / f'{gt_name}_trips_test.csv'),  # ← TEST DATA!
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity
    )
    print(f"✓ Simulator initialized")
    print(f"  Stations: {num_stations}, Vehicles: {num_vehicles}, Capacity: {vehicle_capacity}")
    
    # Initialize agent
    print(f"\n[2/4] Initializing agent...")
    agent = MultiAgentDQN(
        num_stations=num_stations,
        num_vehicles=num_vehicles,
        hidden_dim=1024,
        buffer_capacity=10000,
        batch_size=256,
        learning_rate=2.5e-4,
        gamma=0.99,
        heuristic_routing=heuristic_routing,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        fill_levels=fill_levels
    )
    if heuristic_routing:
        print(f"  Mode: HEURISTIC ROUTING (3 actions)")
    print(f"✓ Agent initialized")
    
    # Load trained model
    print(f"\n[3/4] Loading trained model...")
    print(f"  Model path: {model_path}")
    agent.load(str(model_path))
    print(f"✓ Model loaded successfully!")
    print(f"  Training episodes: {agent.episode_count}")
    print(f"  Training timesteps: {agent.training_step}")
    
    print(f"\n[4/4] Starting test evaluation...")
    
    # Evaluation history
    results = {
        'episode': [],
        'lost_demand': [],
        'lost_demand_rate': [],
        'lost_rentals': [],
        'lost_returns': [],
        'num_decisions': []
    }
    
    # Evaluation loop
    print(f"\n{'='*70}")
    print(f"STARTING EVALUATION - {num_test_episodes} TEST EPISODES")
    print(f"{'='*70}\n")
    
    for episode in range(1, num_test_episodes + 1):
        # Test data has days 1-50 (not 101-150)
        test_day = episode
        
        print(f"Starting Test Episode {episode}/{num_test_episodes} (Day {test_day})...")
        
        # Reset simulator for new episode
        state_dict = simulator.reset(test_day)
        
        num_decisions = 0
        same_station_decisions = 0
        print(f"  Simulating episode...")
        
        # Run episode with trained policy (no exploration, no training)
        while not simulator.is_done():
            # Get next decision epoch
            vehicle_id, reward = simulator.get_next_decision_epoch()
            
            if vehicle_id is None:
                break
            
            # Get current state
            state_dict = simulator.get_state()
            
            # Get vehicle's current location
            current_station = simulator.vehicles[vehicle_id].current_station
            
            # Agent selects action (no exploration, epsilon=0)
            action_idx = agent.select_action(state_dict, vehicle_id, epsilon=epsilon)
            
            # Convert to simulator action
            if heuristic_routing:
                # Simplified mode: DQN selects fill level, heuristic selects station
                fill_level_idx = agent.action_space.get_action(action_idx)
                next_station = agent.action_space.select_station_heuristic(state_dict, vehicle_id)
                action = (next_station, fill_level_idx)
            else:
                # Full mode: DQN selects both station and fill level
                action = agent.action_space.get_action(action_idx)
                next_station, fill_level = action
            
            # Track if staying at same station
            if next_station == current_station:
                same_station_decisions += 1
            
            # Execute action
            simulator.execute_action(vehicle_id, action)
            
            num_decisions += 1
            
            # Log progress every 50 decisions
            if num_decisions % 50 == 0:
                same_pct = (same_station_decisions / num_decisions) * 100
                print(f"    Progress: {num_decisions} decisions ({same_station_decisions} same-station = {same_pct:.1f}%)")
        
        # Get episode metrics
        metrics = simulator.get_metrics()
        
        # Record results
        results['episode'].append(episode)
        results['lost_demand'].append(metrics['total_lost_demand'])
        results['lost_demand_rate'].append(metrics['total_lost_demand_rate'])
        results['lost_rentals'].append(metrics['lost_rentals'])
        results['lost_returns'].append(metrics['lost_returns'])
        results['num_decisions'].append(num_decisions)
        
        # Print progress for ALL episodes (not just every 10)
        same_pct = (same_station_decisions / num_decisions) * 100 if num_decisions > 0 else 0
        print(f"✓ Test Episode {episode}/{num_test_episodes} complete:")
        print(f"    Lost demand: {metrics['total_lost_demand']} ({metrics['total_lost_demand_rate']:.2f}%)")
        print(f"    Decisions made: {num_decisions}")
        print(f"    Same-station choices: {same_station_decisions}/{num_decisions} ({same_pct:.1f}%)")
    
    # Calculate summary statistics
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    
    avg_lost_demand = np.mean(results['lost_demand'])
    avg_lost_demand_rate = np.mean(results['lost_demand_rate'])
    std_lost_demand_rate = np.std(results['lost_demand_rate'])
    min_lost_demand_rate = np.min(results['lost_demand_rate'])
    max_lost_demand_rate = np.max(results['lost_demand_rate'])
    
    print(f"\n📊 Test Set Performance (50 episodes):")
    print(f"  Average lost demand: {avg_lost_demand:.1f} trips")
    print(f"  Average lost demand rate: {avg_lost_demand_rate:.2f}% ± {std_lost_demand_rate:.2f}%")
    print(f"  Best episode: {min_lost_demand_rate:.2f}%")
    print(f"  Worst episode: {max_lost_demand_rate:.2f}%")
    print(f"  Median: {np.median(results['lost_demand_rate']):.2f}%")
    
    # GT-specific baselines from simulator evaluation (not MIP objective)
    # These are actual lost demand rates when running with static inventory only
    baseline = 5.21 if gt_name == 'GT1' else 12.74
    
    print(f"\n🎯 Comparison:")
    print(f"  Baseline (static, simulated): {baseline}%")
    print(f"  Target (paper): 8-10%")
    print(f"  Achieved: {avg_lost_demand_rate:.2f}%")
    
    if avg_lost_demand_rate < baseline:
        improvement = ((baseline - avg_lost_demand_rate) / baseline) * 100
        print(f"  Improvement over baseline: {improvement:.1f}% ✅")
    else:
        print(f"  Worse than baseline ❌")
    
    if avg_lost_demand_rate <= 10:
        print(f"  Target achieved! ✅")
    else:
        gap = avg_lost_demand_rate - 10
        print(f"  Gap to target: {gap:.2f}pp")
    
    # Save results
    results_path = output_dir / f'{gt_name}_evaluation_results.json'
    results_serializable = {k: [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in vs] 
                           for k, vs in results.items()}
    
    # Add summary statistics
    results_serializable['summary'] = {
        'avg_lost_demand': float(avg_lost_demand),
        'avg_lost_demand_rate': float(avg_lost_demand_rate),
        'std_lost_demand_rate': float(std_lost_demand_rate),
        'min_lost_demand_rate': float(min_lost_demand_rate),
        'max_lost_demand_rate': float(max_lost_demand_rate),
        'median_lost_demand_rate': float(np.median(results['lost_demand_rate'])),
        'baseline': baseline,
        'target': 10.0,
        'improvement_over_baseline': float(((baseline - avg_lost_demand_rate) / baseline) * 100) if avg_lost_demand_rate < baseline else 0.0
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\n✓ Evaluation results saved to {results_path}")
    
    # Plot evaluation results
    plot_evaluation_results(results, output_dir, gt_name)
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*70}")
    
    return results


def plot_evaluation_results(results, output_dir, gt_name):
    """Plot evaluation results."""
    # GT-specific baselines from simulator evaluation (not MIP objective)
    baseline = 5.21 if gt_name == 'GT1' else 12.74
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{gt_name} Evaluation on Test Set (50 Episodes)', fontsize=16, fontweight='bold')
    
    episodes = results['episode']
    
    # Lost demand rate
    ax1 = axes[0, 0]
    ax1.plot(episodes, results['lost_demand_rate'], 'b-', marker='o', markersize=4, alpha=0.6)
    ax1.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline}%)', alpha=0.7)
    ax1.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Target (10%)', alpha=0.7)
    ax1.axhline(y=np.mean(results['lost_demand_rate']), color='blue', linestyle='-', linewidth=2, 
                label=f'Mean ({np.mean(results["lost_demand_rate"]):.2f}%)', alpha=0.7)
    ax1.set_xlabel('Test Episode', fontsize=11)
    ax1.set_ylabel('Lost Demand Rate (%)', fontsize=11)
    ax1.set_title('Lost Demand Rate on Test Set', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(results['lost_demand_rate'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=baseline, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax2.axvline(x=10, color='green', linestyle='--', linewidth=2, label='Target')
    ax2.axvline(x=np.mean(results['lost_demand_rate']), color='blue', linestyle='-', linewidth=2, label='Mean')
    ax2.set_xlabel('Lost Demand Rate (%)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Lost Demand Rates', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Lost rentals vs returns
    ax3 = axes[1, 0]
    x = np.arange(len(episodes))
    width = 0.35
    ax3.bar(x - width/2, results['lost_rentals'], width, label='Lost Rentals', color='orange', alpha=0.7)
    ax3.bar(x + width/2, results['lost_returns'], width, label='Lost Returns', color='purple', alpha=0.7)
    ax3.set_xlabel('Test Episode', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Lost Rentals vs Lost Returns', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    if len(episodes) > 10:
        ax3.set_xticks(x[::5])
        ax3.set_xticklabels(episodes[::5])
    
    # Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    TEST SET EVALUATION SUMMARY
    
    Episodes evaluated: {len(episodes)}
    
    Lost Demand Rate:
      • Mean: {np.mean(results['lost_demand_rate']):.2f}%
      • Std Dev: {np.std(results['lost_demand_rate']):.2f}%
      • Median: {np.median(results['lost_demand_rate']):.2f}%
      • Min: {np.min(results['lost_demand_rate']):.2f}%
      • Max: {np.max(results['lost_demand_rate']):.2f}%
    
    Comparison:
      • Baseline: {baseline}%
      • Target: 8-10%
      • Gap to target: {max(0, np.mean(results['lost_demand_rate']) - 10):.2f}pp
    
    Performance:
      • Decisions/episode: {np.mean(results['num_decisions']):.1f}
      • Total lost demand: {np.sum(results['lost_demand']):.0f} trips
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    plot_path = output_dir / f'{gt_name}_evaluation_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Evaluation plots saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained Multi-Agent DQN on test set')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model (.pth file)')
    parser.add_argument('--gt', type=str, default='GT1', choices=['GT0', 'GT1', 'GT2'],
                      help='Ground truth to evaluate on')
    parser.add_argument('--episodes', type=int, default=50,
                      help='Number of test episodes (default: 50)')
    parser.add_argument('--epsilon', type=float, default=0.0,
                      help='Exploration rate during evaluation (default: 0.0 = pure exploitation)')
    parser.add_argument('--heuristic-routing', action='store_true',
                      help='Use simplified action space with heuristic routing')
    
    # Environment configuration (for GT0 toy model)
    parser.add_argument('--num-stations', type=int, default=60,
                      help='Number of stations (default: 60, GT0: 10)')
    parser.add_argument('--num-vehicles', type=int, default=4,
                      help='Number of vehicles (default: 4, GT0: 2)')
    parser.add_argument('--vehicle-capacity', type=int, default=40,
                      help='Vehicle bike capacity (default: 40, GT0: 15)')
    
    # Network architecture (must match training)
    parser.add_argument('--hidden-activation', type=str, default='relu',
                      choices=['relu', 'leaky_relu', 'prelu', 'elu'],
                      help='Activation function for hidden layers (default: relu)')
    parser.add_argument('--output-activation', type=str, default=None,
                      choices=['none', 'leaky_relu', 'prelu', 'elu'],
                      help='Activation function for output layer (default: none)')
    parser.add_argument('--fill-levels', type=str, default='10,50,90',
                      help='Fill levels as comma-separated percentages (default: 10,50,90)')
    
    args = parser.parse_args()
    
    # Parse fill levels
    fill_levels = [float(x)/100.0 for x in args.fill_levels.split(',')]
    
    # Parse output activation (handle 'none')
    output_activation = None if args.output_activation == 'none' or args.output_activation is None else args.output_activation
    
    # Evaluate agent
    results = evaluate_agent(
        model_path=args.model,
        gt_name=args.gt,
        num_test_episodes=args.episodes,
        epsilon=args.epsilon,
        heuristic_routing=args.heuristic_routing,
        num_stations=args.num_stations,
        num_vehicles=args.num_vehicles,
        vehicle_capacity=args.vehicle_capacity,
        hidden_activation=args.hidden_activation,
        output_activation=output_activation,
        fill_levels=fill_levels
    )
    
    print("\n🎉 Evaluation complete! Check results/ folder for detailed analysis.")
