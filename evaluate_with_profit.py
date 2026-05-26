"""
Evaluate policies with full profit analysis.
Tracks: revenue, truck costs, lost demand penalty, and net profit.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import matplotlib.pyplot as plt

from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator
from rl_algorithm.multi_agent_dqn import MultiAgentDQN


def calculate_distance(network_data, from_station, to_station):
    """Calculate distance between two stations."""
    stations = {s['id']: s for s in network_data['stations']}
    if from_station not in stations or to_station not in stations:
        return 0
    s1, s2 = stations[from_station], stations[to_station]
    # Haversine approximation (simplified)
    lat1, lon1 = s1['latitude'], s1['longitude']
    lat2, lon2 = s2['latitude'], s2['longitude']
    # Simple Euclidean in degrees * 111km/degree
    dist = np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111
    return dist


def evaluate_policy_with_profit(
    model_path,
    output_activation,
    gt_name='GT0',
    num_episodes=50,
    cost_per_km=1.0,
    lost_penalty=5.0,
    trip_base_fare=1.0,
    trip_per_km=0.75,
    fill_levels=None,
    hidden_activation='relu',
    num_stations=10,
    num_vehicles=2,
    vehicle_capacity=15,
):
    """Evaluate a model with full profit tracking on the GT0 test set.

    `fill_levels` and `hidden_activation` MUST match what the model was trained
    with, otherwise the network architecture won't match the saved weights.
    """
    if fill_levels is None:
        fill_levels = [0.10, 0.50, 0.90]

    base_dir = Path(__file__).parent.parent / 'data' / 'synthetic' / gt_name
    network_file = str(base_dir / f'{gt_name}_station_network.json')
    trips_file = str(base_dir / f'{gt_name}_trips_test.csv')

    with open(network_file) as f:
        network_data = json.load(f)

    simulator = ContinuousTimeSimulator(
        network_file=network_file,
        trips_file=trips_file,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        fill_levels=fill_levels,
    )

    agent = MultiAgentDQN(
        num_stations=num_stations,
        num_vehicles=num_vehicles,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        fill_levels=fill_levels,
    )
    agent.load(model_path)
    
    results = []
    
    for ep in range(num_episodes):
        day = (ep % 50) + 1
        simulator.reset(day)
        
        episode_truck_distance = 0.0
        action_count = 0
        vehicle_positions = {0: 0, 1: 0}  # Track vehicle positions (station IDs)
        
        while not simulator.is_done():
            vehicle_id, _ = simulator.get_next_decision_epoch()
            if vehicle_id is None:
                break
            
            state_dict = simulator.get_state()
            action_idx = agent.select_action(state_dict, vehicle_id, epsilon=0.0)
            action = agent.action_space.get_action(action_idx)
            
            # Track distance
            target_station = action[0]
            current_pos = vehicle_positions.get(vehicle_id, 0)
            dist = calculate_distance(network_data, current_pos, target_station)
            episode_truck_distance += dist
            vehicle_positions[vehicle_id] = target_station
            
            simulator.execute_action(vehicle_id, action)
            action_count += 1
        
        metrics = simulator.get_metrics()
        
        # Calculate profit components
        successful_trips = metrics['successful_rentals'] + metrics['successful_returns']
        lost_trips = metrics['total_lost_demand']
        
        # Revenue: estimate based on average trip distance
        avg_trip_km = 2.5
        revenue = successful_trips * (trip_base_fare + trip_per_km * avg_trip_km)
        
        # Costs
        truck_cost = episode_truck_distance * cost_per_km
        lost_cost = lost_trips * lost_penalty
        
        # Profit
        profit = revenue - truck_cost - lost_cost
        
        results.append({
            'day': day,
            'lost_demand_rate': metrics['total_lost_demand_rate'],
            'lost_trips': lost_trips,
            'successful_trips': successful_trips,
            'action_count': action_count,
            'revenue': revenue,
            'truck_distance_km': episode_truck_distance,
            'truck_cost': truck_cost,
            'lost_cost': lost_cost,
            'profit': profit
        })
    
    return results


def run_profit_comparison():
    """Compare lost-demand and profit DQN with full profit metrics."""
    
    code_dir = Path(__file__).parent
    output_dir = code_dir / 'results_policy_comparison'
    
    # Models
    ld_model = code_dir / 'results_GT0' / 'activation_elu' / 'GT0_multi_agent_dqn_final.pth'
    pf_model = code_dir / 'results_profit_GT0' / 'activation_prelu' / 'GT0_profit_dqn_best.pth'
    
    print("=" * 70)
    print("PROFIT ANALYSIS: Lost-Demand DQN vs Profit DQN")
    print("=" * 70)
    
    # Evaluate Lost-Demand DQN
    print("\n[1/2] Evaluating Lost-Demand DQN (ELU)...")
    ld_results = evaluate_policy_with_profit(
        model_path=str(ld_model),
        output_activation='elu'
    )
    
    # Evaluate Profit DQN
    print("[2/2] Evaluating Profit DQN (PReLU)...")
    pf_results = evaluate_policy_with_profit(
        model_path=str(pf_model),
        output_activation='prelu'
    )
    
    # Aggregate
    def aggregate(results):
        return {
            'avg_lost_rate': np.mean([r['lost_demand_rate'] for r in results]),
            'std_lost_rate': np.std([r['lost_demand_rate'] for r in results]),
            'avg_revenue': np.mean([r['revenue'] for r in results]),
            'avg_truck_distance': np.mean([r['truck_distance_km'] for r in results]),
            'avg_truck_cost': np.mean([r['truck_cost'] for r in results]),
            'avg_lost_cost': np.mean([r['lost_cost'] for r in results]),
            'avg_profit': np.mean([r['profit'] for r in results]),
            'std_profit': np.std([r['profit'] for r in results]),
            'total_lost': sum([r['lost_trips'] for r in results]),
            'total_successful': sum([r['successful_trips'] for r in results]),
            'avg_actions': np.mean([r['action_count'] for r in results])
        }
    
    ld_agg = aggregate(ld_results)
    pf_agg = aggregate(pf_results)
    
    # Print results
    print("\n" + "=" * 70)
    print("PROFIT COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'Lost-Demand DQN':>18} {'Profit DQN':>18}")
    print("-" * 70)
    print(f"{'Avg Profit':<25} ${ld_agg['avg_profit']:>16.2f} ${pf_agg['avg_profit']:>16.2f}")
    print(f"{'Std Profit':<25} ${ld_agg['std_profit']:>16.2f} ${pf_agg['std_profit']:>16.2f}")
    print(f"{'Avg Revenue':<25} ${ld_agg['avg_revenue']:>16.2f} ${pf_agg['avg_revenue']:>16.2f}")
    print(f"{'Avg Truck Cost':<25} ${ld_agg['avg_truck_cost']:>16.2f} ${pf_agg['avg_truck_cost']:>16.2f}")
    print(f"{'Avg Lost Penalty':<25} ${ld_agg['avg_lost_cost']:>16.2f} ${pf_agg['avg_lost_cost']:>16.2f}")
    print(f"{'Truck Distance (km)':<25} {ld_agg['avg_truck_distance']:>17.2f} {pf_agg['avg_truck_distance']:>17.2f}")
    print(f"{'Lost Demand Rate':<25} {ld_agg['avg_lost_rate']:>17.2f}% {pf_agg['avg_lost_rate']:>17.2f}%")
    print("-" * 70)
    
    profit_diff = pf_agg['avg_profit'] - ld_agg['avg_profit']
    print(f"\nProfit Difference: ${profit_diff:.2f} ({'better' if profit_diff > 0 else 'worse'} for Profit DQN)")
    
    # Save results
    output = {
        'lost_demand_dqn': {
            'model': str(ld_model),
            'activation': 'elu',
            'aggregated': ld_agg,
            'episodes': ld_results
        },
        'profit_dqn': {
            'model': str(pf_model),
            'activation': 'prelu',
            'aggregated': pf_agg,
            'episodes': pf_results
        },
        'comparison': {
            'profit_diff': profit_diff,
            'lost_rate_diff': pf_agg['avg_lost_rate'] - ld_agg['avg_lost_rate'],
            'truck_cost_diff': pf_agg['avg_truck_cost'] - ld_agg['avg_truck_cost']
        }
    }
    
    output_file = output_dir / 'profit_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Create plots
    create_profit_plots(ld_results, pf_results, ld_agg, pf_agg, output_dir)
    
    return output


def create_profit_plots(ld_results, pf_results, ld_agg, pf_agg, output_dir):
    """Create profit-focused plots."""
    
    # Plot 1: Profit Breakdown Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(3)
    width = 0.35
    
    ld_values = [ld_agg['avg_revenue'], -ld_agg['avg_truck_cost'], -ld_agg['avg_lost_cost']]
    pf_values = [pf_agg['avg_revenue'], -pf_agg['avg_truck_cost'], -pf_agg['avg_lost_cost']]
    
    bars1 = ax.bar(x - width/2, ld_values, width, label='Lost-Demand DQN', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, pf_values, width, label='Profit DQN', color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('Amount ($)', fontsize=12)
    ax.set_title('Profit Breakdown: Revenue and Costs', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Revenue', 'Truck Cost', 'Lost Penalty'], fontsize=11)
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'${abs(height):.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3 if height >= 0 else -12),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'profit_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'profit_breakdown.png'}")
    
    # Plot 2: Net Profit Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['Lost-Demand\nDQN', 'Profit\nDQN']
    profits = [ld_agg['avg_profit'], pf_agg['avg_profit']]
    stds = [ld_agg['std_profit'], pf_agg['std_profit']]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(methods, profits, yerr=stds, capsize=8, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Average Profit ($)', fontsize=12)
    ax.set_title('Net Profit Comparison', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    for bar, profit in zip(bars, profits):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'${profit:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'profit_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'profit_comparison.png'}")
    
    # Plot 3: Per-Episode Profit
    fig, ax = plt.subplots(figsize=(12, 5))
    
    episodes = range(1, len(ld_results) + 1)
    ld_profits = [r['profit'] for r in ld_results]
    pf_profits = [r['profit'] for r in pf_results]
    
    ax.plot(episodes, ld_profits, 'b-', alpha=0.7, label='Lost-Demand DQN', linewidth=1.5)
    ax.plot(episodes, pf_profits, 'r-', alpha=0.7, label='Profit DQN', linewidth=1.5)
    
    ax.axhline(y=np.mean(ld_profits), color='blue', linestyle='--', alpha=0.5, 
               label=f'LD Mean: ${np.mean(ld_profits):.2f}')
    ax.axhline(y=np.mean(pf_profits), color='red', linestyle='--', alpha=0.5,
               label=f'PF Mean: ${np.mean(pf_profits):.2f}')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Profit ($)', fontsize=12)
    ax.set_title('Per-Episode Profit Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'profit_per_episode.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'profit_per_episode.png'}")
    
    # Plot 4: Summary metrics (lost demand + profit side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Lost Demand
    ax = axes[0]
    means = [ld_agg['avg_lost_rate'], pf_agg['avg_lost_rate']]
    stds = [ld_agg['std_lost_rate'], pf_agg['std_lost_rate']]
    bars = ax.bar(methods, means, yerr=stds, capsize=8, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Lost Demand Rate (%)', fontsize=12)
    ax.set_title('Service Quality', fontsize=14, fontweight='bold')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{m:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right: Profit
    ax = axes[1]
    profits = [ld_agg['avg_profit'], pf_agg['avg_profit']]
    bars = ax.bar(methods, profits, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Average Profit ($)', fontsize=12)
    ax.set_title('Economic Performance', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    for bar, p in zip(bars, profits):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'${p:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'policy_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'policy_summary.png'}")


if __name__ == "__main__":
    run_profit_comparison()
