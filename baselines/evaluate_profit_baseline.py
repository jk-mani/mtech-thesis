"""
Profit-Based Baseline Evaluation.

Evaluates the static rebalancing baseline using profit metrics, providing
a fair comparison point for the profit-based DQN agent.

The static baseline:
- Uses MIP-optimized initial inventory
- No dynamic rebalancing (vehicles don't move)
- All profit comes from trip revenue minus lost demand costs

This establishes the economic baseline that the RL agent must beat.

Usage:
    python evaluate_profit_baseline.py --gt GT0 --episodes 50
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_algorithm.profit_reward import ProfitParameters, ProfitRewardCalculator


class Station:
    """Simple station for baseline evaluation."""
    
    def __init__(self, station_id, capacity, initial_inventory=None):
        self.id = station_id
        self.capacity = capacity
        self.inventory = initial_inventory if initial_inventory is not None else capacity // 2
        
        self.lost_rentals = 0
        self.lost_returns = 0
        self.successful_rentals = 0
        self.successful_returns = 0
    
    def process_rental(self):
        if self.inventory > 0:
            self.inventory -= 1
            self.successful_rentals += 1
            return True
        else:
            self.lost_rentals += 1
            return False
    
    def process_return(self):
        if self.inventory < self.capacity:
            self.inventory += 1
            self.successful_returns += 1
            return True
        else:
            self.lost_returns += 1
            return False
    
    def reset(self, initial_inventory=None):
        self.inventory = initial_inventory if initial_inventory is not None else self.capacity // 2
        self.lost_rentals = 0
        self.lost_returns = 0
        self.successful_rentals = 0
        self.successful_returns = 0


def evaluate_static_profit_baseline(
    gt_name='GT0',
    num_episodes=50,
    profit_params=None,
    output_dir=None
):
    """
    Evaluate static baseline with profit metrics.
    
    The static baseline represents the "no dynamic rebalancing" scenario:
    - Initial inventory is optimized by MIP
    - No vehicles move during the day
    - Revenue from successful trips, cost from lost demand
    
    Args:
        gt_name: Ground truth network
        num_episodes: Number of test episodes
        profit_params: Economic parameters
        output_dir: Directory to save results
        
    Returns:
        dict: Evaluation results
    """
    print("\n" + "="*70)
    print("PROFIT-BASED STATIC BASELINE EVALUATION")
    print("="*70)
    
    # Setup
    if profit_params is None:
        profit_params = ProfitParameters()
    
    if output_dir is None:
        output_dir = Path(f'../results_profit_baseline/{gt_name}')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Network: {gt_name}")
    print(f"  Test episodes: {num_episodes}")
    print(f"  Trip revenue: ${profit_params.trip_revenue:.2f}")
    print(f"  Lost demand penalty: ${profit_params.lost_demand_penalty:.2f}")
    
    # Load network
    base_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / gt_name
    network_file = base_dir / f'{gt_name}_station_network.json'
    trips_file = base_dir / f'{gt_name}_trips_test.csv'
    static_inventory_file = base_dir / f'{gt_name}_static_initial_inventory.json'
    
    with open(network_file) as f:
        network_data = json.load(f)
    
    # Load static inventory if available
    static_inventory = None
    if static_inventory_file.exists():
        with open(static_inventory_file) as f:
            static_data = json.load(f)
            static_inventory = {int(k): v for k, v in static_data['initial_inventory'].items()}
        print(f"✓ Loaded static inventory from MIP solution")
    else:
        print("⚠ No static inventory found, using 50% fill")
    
    # Initialize stations
    stations = {}
    for s in network_data['stations']:
        initial = static_inventory.get(s['id'], s['capacity'] // 2) if static_inventory else s['capacity'] // 2
        stations[s['id']] = Station(s['id'], s['capacity'], initial)
    
    # Load trips
    trips_df = pd.read_csv(trips_file)
    trips_df['departure_time'] = pd.to_datetime(trips_df['departure_time'])
    trips_df['arrival_time'] = pd.to_datetime(trips_df['arrival_time'])
    trips_df['date'] = trips_df['departure_time'].dt.date
    available_dates = sorted(trips_df['date'].unique())
    
    print(f"✓ Loaded {len(trips_df)} trips across {len(available_dates)} days")
    
    # Results storage
    results = {
        'episode': [],
        'profit': [],
        'revenue': [],
        'lost_demand_cost': [],
        'lost_demand': [],
        'lost_demand_rate': [],
        'successful_trips': [],
        'lost_rentals': [],
        'lost_returns': []
    }
    
    print(f"\n[1/2] Running {num_episodes} baseline episodes...")
    
    # Use deterministic date selection for reproducibility
    np.random.seed(42)
    
    for ep in range(num_episodes):
        # Select date
        date = available_dates[ep % len(available_dates)]
        day_trips = trips_df[trips_df['date'] == date].copy()
        
        # Reset stations
        for sid, station in stations.items():
            initial = static_inventory.get(sid, station.capacity // 2) if static_inventory else station.capacity // 2
            station.reset(initial)
        
        # Process all trips chronologically
        all_events = []
        
        for _, trip in day_trips.iterrows():
            all_events.append(('rental', trip['departure_time'], trip['origin_station']))
            all_events.append(('return', trip['arrival_time'], trip['destination_station']))
        
        # Sort by time
        all_events.sort(key=lambda x: x[1])
        
        # Process events
        for event_type, event_time, station_id in all_events:
            station = stations[station_id]
            if event_type == 'rental':
                station.process_rental()
            else:
                station.process_return()
        
        # Calculate metrics
        total_successful_rentals = sum(s.successful_rentals for s in stations.values())
        total_successful_returns = sum(s.successful_returns for s in stations.values())
        total_lost_rentals = sum(s.lost_rentals for s in stations.values())
        total_lost_returns = sum(s.lost_returns for s in stations.values())
        
        total_successful = total_successful_rentals + total_successful_returns
        total_lost = total_lost_rentals + total_lost_returns
        total_demand = total_successful + total_lost
        
        # Calculate profit
        # Revenue: from successful trips only
        revenue = total_successful * profit_params.trip_revenue
        
        # Cost: only lost demand (no operational costs since no vehicles move)
        lost_demand_cost = total_lost * profit_params.lost_demand_penalty
        
        # Net profit (no operational costs for static baseline)
        profit = revenue - lost_demand_cost
        
        # Lost demand rate
        lost_demand_rate = (total_lost / total_demand * 100) if total_demand > 0 else 0
        
        # Store results
        results['episode'].append(ep + 1)
        results['profit'].append(profit)
        results['revenue'].append(revenue)
        results['lost_demand_cost'].append(lost_demand_cost)
        results['lost_demand'].append(total_lost)
        results['lost_demand_rate'].append(lost_demand_rate)
        results['successful_trips'].append(total_successful)
        results['lost_rentals'].append(total_lost_rentals)
        results['lost_returns'].append(total_lost_returns)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"Profit=${profit:.2f}, Lost={lost_demand_rate:.1f}%")
    
    print(f"\n[2/2] Computing summary statistics...")
    
    # Summary
    summary = {
        # Profit metrics
        'avg_profit': np.mean(results['profit']),
        'std_profit': np.std(results['profit']),
        'min_profit': np.min(results['profit']),
        'max_profit': np.max(results['profit']),
        'median_profit': np.median(results['profit']),
        
        # Revenue
        'avg_revenue': np.mean(results['revenue']),
        
        # Costs (only lost demand for static)
        'avg_lost_demand_cost': np.mean(results['lost_demand_cost']),
        'avg_operational_cost': 0.0,  # No vehicle operations
        
        # Service quality
        'avg_lost_demand': np.mean(results['lost_demand']),
        'avg_lost_demand_rate': np.mean(results['lost_demand_rate']),
        'std_lost_demand_rate': np.std(results['lost_demand_rate']),
        
        # Operational
        'avg_successful_trips': np.mean(results['successful_trips']),
        'avg_distance_km': 0.0,  # No vehicle movement
        'avg_bikes_moved': 0.0,  # No rebalancing
        
        # Efficiency
        'profit_margin': (np.mean(results['profit']) / max(1, np.mean(results['revenue']))) * 100
    }
    
    # Print results
    print("\n" + "="*70)
    print("STATIC BASELINE RESULTS")
    print("="*70)
    
    print(f"\n📊 Profit Metrics ({num_episodes} episodes):")
    print(f"  Average Profit: ${summary['avg_profit']:.2f} ± ${summary['std_profit']:.2f}")
    print(f"  Best Episode: ${summary['max_profit']:.2f}")
    print(f"  Worst Episode: ${summary['min_profit']:.2f}")
    print(f"  Profit Margin: {summary['profit_margin']:.1f}%")
    
    print(f"\n💰 Economics (averages):")
    print(f"  Revenue: ${summary['avg_revenue']:.2f}")
    print(f"  Lost Demand Cost: ${summary['avg_lost_demand_cost']:.2f}")
    print(f"  Operational Cost: ${summary['avg_operational_cost']:.2f} (no vehicle movement)")
    
    print(f"\n🎯 Service Quality:")
    print(f"  Lost Demand Rate: {summary['avg_lost_demand_rate']:.2f}% ± {summary['std_lost_demand_rate']:.2f}%")
    print(f"  Successful Trips: {summary['avg_successful_trips']:.1f}")
    
    print(f"\n⚠️ Note: Static baseline has NO operational costs")
    print(f"   DQN must achieve higher profit despite vehicle costs!")
    
    # Save results
    full_results = {
        'summary': summary,
        'episodes': results,
        'config': {
            'gt_name': gt_name,
            'num_episodes': num_episodes,
            'baseline_type': 'static_mip',
            'profit_params': {
                'trip_revenue': profit_params.trip_revenue,
                'lost_demand_penalty': profit_params.lost_demand_penalty
            }
        }
    }
    
    results_file = output_dir / f'{gt_name}_profit_baseline_results.json'
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")
    
    # Generate plots
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{gt_name} Static Baseline - Profit Analysis', fontsize=14, fontweight='bold')
        
        # Profit distribution
        axes[0].hist(results['profit'], bins=15, color='orange', alpha=0.7, edgecolor='black')
        axes[0].axvline(summary['avg_profit'], color='red', linestyle='--', linewidth=2,
                       label=f"Mean: ${summary['avg_profit']:.2f}")
        axes[0].set_xlabel('Net Profit ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Profit Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Revenue vs Lost Demand Cost
        axes[1].bar(['Revenue', 'Lost Demand Cost'], 
                   [summary['avg_revenue'], summary['avg_lost_demand_cost']],
                   color=['green', 'red'], alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Amount ($)')
        axes[1].set_title('Revenue vs Cost')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Lost demand rate
        axes[2].hist(results['lost_demand_rate'], bins=15, color='red', alpha=0.7, edgecolor='black')
        axes[2].axvline(summary['avg_lost_demand_rate'], color='darkred', linestyle='--', 
                       linewidth=2, label=f"Mean: {summary['avg_lost_demand_rate']:.1f}%")
        axes[2].set_xlabel('Lost Demand Rate (%)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Lost Demand Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_dir / f'{gt_name}_profit_baseline_results.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Plots saved to {plot_file}")
        
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    print("\n" + "="*70)
    print("✅ STATIC BASELINE EVALUATION COMPLETE!")
    print("="*70)
    
    return full_results


def compare_with_dqn(baseline_results, dqn_results_path):
    """
    Compare static baseline with DQN results.
    
    Args:
        baseline_results: Results from evaluate_static_profit_baseline
        dqn_results_path: Path to DQN evaluation results
        
    Returns:
        dict: Comparison metrics
    """
    with open(dqn_results_path) as f:
        dqn_results = json.load(f)
    
    baseline_profit = baseline_results['summary']['avg_profit']
    dqn_profit = dqn_results['summary']['avg_profit']
    
    baseline_lost = baseline_results['summary']['avg_lost_demand_rate']
    dqn_lost = dqn_results['summary']['avg_lost_demand_rate']
    
    print("\n" + "="*70)
    print("COMPARISON: Static Baseline vs Profit-Based DQN")
    print("="*70)
    
    print(f"\n💰 Profit:")
    print(f"  Static Baseline: ${baseline_profit:.2f}")
    print(f"  DQN Agent:       ${dqn_profit:.2f}")
    profit_improvement = ((dqn_profit - baseline_profit) / abs(baseline_profit)) * 100 if baseline_profit != 0 else 0
    print(f"  Improvement:     {profit_improvement:+.1f}%")
    
    print(f"\n🎯 Lost Demand Rate:")
    print(f"  Static Baseline: {baseline_lost:.2f}%")
    print(f"  DQN Agent:       {dqn_lost:.2f}%")
    lost_improvement = ((baseline_lost - dqn_lost) / baseline_lost) * 100 if baseline_lost > 0 else 0
    print(f"  Reduction:       {lost_improvement:.1f}%")
    
    print(f"\n📊 Trade-off Analysis:")
    dqn_op_cost = dqn_results['summary'].get('avg_operational_cost', 0)
    print(f"  DQN Operational Cost: ${dqn_op_cost:.2f}")
    print(f"  Extra Cost justified by: ${dqn_profit - baseline_profit:.2f} extra profit")
    
    if dqn_profit > baseline_profit:
        print(f"\n✅ DQN achieves HIGHER profit despite operational costs!")
    else:
        print(f"\n⚠️ DQN has LOWER profit - operational costs outweigh benefits")
    
    return {
        'baseline_profit': baseline_profit,
        'dqn_profit': dqn_profit,
        'profit_improvement_pct': profit_improvement,
        'baseline_lost_demand': baseline_lost,
        'dqn_lost_demand': dqn_lost,
        'lost_demand_reduction_pct': lost_improvement,
        'dqn_operational_cost': dqn_op_cost
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Static Baseline with Profit Metrics')
    
    parser.add_argument('--gt', type=str, default='GT0', choices=['GT0', 'GT1', 'GT2'])
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default=None)
    
    # Profit parameters
    parser.add_argument('--trip-revenue', type=float, default=3.50)
    parser.add_argument('--lost-demand-penalty', type=float, default=5.00)
    
    # Optional: compare with DQN
    parser.add_argument('--compare-dqn', type=str, default=None,
                       help='Path to DQN evaluation results for comparison')
    
    args = parser.parse_args()
    
    # Build profit params (only revenue and penalty matter for static)
    profit_params = ProfitParameters(
        trip_revenue=args.trip_revenue,
        lost_demand_penalty=args.lost_demand_penalty
    )
    
    # Evaluate baseline
    results = evaluate_static_profit_baseline(
        gt_name=args.gt,
        num_episodes=args.episodes,
        profit_params=profit_params,
        output_dir=args.output_dir
    )
    
    # Compare with DQN if provided
    if args.compare_dqn:
        compare_with_dqn(results, args.compare_dqn)
