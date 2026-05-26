"""
Evaluate economic sensitivity experiments and create publication-quality plots.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator
from rl_algorithm.multi_agent_dqn import MultiAgentDQN


def evaluate_profit_model(
    model_path,
    output_activation='prelu',
    gt_name='GT0',
    num_episodes=50,
    cost_per_km=1.0,
    lost_penalty=5.0,
    trip_base_fare=1.0,
    trip_per_km=0.75
):
    """Evaluate a profit-based DQN model on test data with economic metrics."""
    
    base_dir = Path(__file__).parent.parent / 'data' / 'synthetic' / gt_name
    network_file = str(base_dir / f'{gt_name}_station_network.json')
    trips_file = str(base_dir / f'{gt_name}_trips_test.csv')
    
    # Load network for distance calculations
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
        output_activation=output_activation,
        fill_levels=[0.10, 0.50, 0.90]
    )
    agent.load(model_path)
    
    results = []
    
    for ep in range(num_episodes):
        day = (ep % 50) + 1
        simulator.reset(day)
        
        action_count = 0
        
        while not simulator.is_done():
            vehicle_id, _ = simulator.get_next_decision_epoch()
            if vehicle_id is None:
                break
            
            state_dict = simulator.get_state()
            action_idx = agent.select_action(state_dict, vehicle_id, epsilon=0.0)
            action = agent.action_space.get_action(action_idx)
            simulator.execute_action(vehicle_id, action)
            action_count += 1
        
        metrics = simulator.get_metrics()
        
        # Calculate profit components
        successful_trips = metrics['successful_rentals'] + metrics['successful_returns']
        lost_trips = metrics['total_lost_demand']
        
        # Estimate revenue (approximate - use average trip distance)
        avg_trip_km = 2.5  # approximate
        revenue = successful_trips * (trip_base_fare + trip_per_km * avg_trip_km)
        
        # Lost demand cost
        lost_cost = lost_trips * lost_penalty
        
        results.append({
            'day': day,
            'lost_demand_rate': metrics['total_lost_demand_rate'],
            'lost_trips': lost_trips,
            'successful_trips': successful_trips,
            'action_count': action_count,
            'revenue': revenue,
            'lost_cost': lost_cost
        })
    
    return results


def evaluate_all_economic_configs():
    """Evaluate all economic sensitivity configurations."""
    
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
    print("EVALUATING ECONOMIC SENSITIVITY CONFIGURATIONS")
    print("=" * 70)
    
    for name, cost_km, penalty in configs:
        model_path = results_dir / name / 'GT0_profit_dqn_best.pth'
        
        if not model_path.exists():
            model_path = results_dir / name / 'GT0_profit_dqn_final.pth'
        
        if not model_path.exists():
            print(f"⚠ Model not found for {name}")
            continue
        
        print(f"\n[{name}] cost=${cost_km}/km, penalty=${penalty}...")
        
        results = evaluate_profit_model(
            model_path=str(model_path),
            num_episodes=50,
            cost_per_km=cost_km,
            lost_penalty=penalty
        )
        
        # Aggregate
        all_results[name] = {
            'cost_per_km': cost_km,
            'lost_penalty': penalty,
            'avg_lost_rate': np.mean([r['lost_demand_rate'] for r in results]),
            'std_lost_rate': np.std([r['lost_demand_rate'] for r in results]),
            'avg_actions': np.mean([r['action_count'] for r in results]),
            'total_lost': sum([r['lost_trips'] for r in results]),
            'total_successful': sum([r['successful_trips'] for r in results]),
            'episodes': results
        }
        
        print(f"  Lost Demand: {all_results[name]['avg_lost_rate']:.2f}%")
        print(f"  Avg Actions: {all_results[name]['avg_actions']:.1f}")
    
    return all_results


def create_economic_sensitivity_plots(results):
    """Create publication-quality plots for economic sensitivity analysis."""
    
    output_dir = Path(__file__).parent / 'results_economic_sensitivity'
    
    # Prepare data
    configs = list(results.keys())
    lost_rates = [results[c]['avg_lost_rate'] for c in configs]
    lost_stds = [results[c]['std_lost_rate'] for c in configs]
    actions = [results[c]['avg_actions'] for c in configs]
    cost_per_km = [results[c]['cost_per_km'] for c in configs]
    penalties = [results[c]['lost_penalty'] for c in configs]
    
    # Color coding by cost_per_km
    colors = []
    for c in configs:
        if results[c]['cost_per_km'] == 0.5:
            colors.append('#2ecc71')  # green - low cost
        elif results[c]['cost_per_km'] == 1.0:
            colors.append('#3498db')  # blue - baseline cost
        else:
            colors.append('#e74c3c')  # red - high cost
    
    # Plot 1: Lost Demand Rate by Configuration
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(configs))
    bars = ax.bar(x, lost_rates, yerr=lost_stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Lost Demand Rate (%)', fontsize=12)
    ax.set_title('Economic Parameter Sensitivity: Lost Demand Rate', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in configs], fontsize=10)
    ax.axhline(y=results['baseline']['avg_lost_rate'], color='gray', linestyle='--', alpha=0.7, label='Baseline')
    
    # Add legend for colors
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Low Cost ($0.50/km)'),
        mpatches.Patch(color='#3498db', label='Baseline ($1.00/km)'),
        mpatches.Patch(color='#e74c3c', label='High Cost ($2.00/km)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'economic_lost_demand.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'economic_lost_demand.png'}")
    
    # Plot 2: Actions per Episode
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(x, actions, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Average Actions per Episode', fontsize=12)
    ax.set_title('Economic Parameter Sensitivity: Agent Activity', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in configs], fontsize=10)
    ax.axhline(y=results['baseline']['avg_actions'], color='gray', linestyle='--', alpha=0.7)
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'economic_actions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'economic_actions.png'}")
    
    # Plot 3: Cost vs Penalty Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create matrix
    cost_values = [0.5, 1.0, 2.0]
    penalty_values = [2.0, 5.0, 10.0]
    matrix = np.zeros((3, 3))
    
    config_map = {
        (0.5, 5.0): 'low_cost',
        (1.0, 5.0): 'baseline',
        (2.0, 5.0): 'high_cost',
        (1.0, 2.0): 'low_penalty',
        (1.0, 10.0): 'high_penalty',
        (0.5, 10.0): 'cheap_aggressive',
        (2.0, 2.0): 'expensive_conservative',
    }
    
    for i, cost in enumerate(cost_values):
        for j, penalty in enumerate(penalty_values):
            key = (cost, penalty)
            if key in config_map and config_map[key] in results:
                matrix[j, i] = results[config_map[key]]['avg_lost_rate']
            else:
                matrix[j, i] = np.nan
    
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=4, vmax=12)
    
    ax.set_xticks(range(3))
    ax.set_xticklabels(['$0.50', '$1.00', '$2.00'])
    ax.set_yticks(range(3))
    ax.set_yticklabels(['$2.00', '$5.00', '$10.00'])
    ax.set_xlabel('Truck Cost ($/km)', fontsize=12)
    ax.set_ylabel('Lost Demand Penalty ($)', fontsize=12)
    ax.set_title('Lost Demand Rate (%) by Economic Parameters', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            if not np.isnan(matrix[j, i]):
                ax.text(i, j, f'{matrix[j, i]:.1f}%', ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Lost Demand Rate (%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'economic_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'economic_heatmap.png'}")
    
    return output_dir


def create_policy_comparison_plots():
    """Create plots comparing lost-demand vs profit policies."""
    
    results_file = Path(__file__).parent / 'results_policy_comparison' / 'comparison_results.json'
    output_dir = Path(__file__).parent / 'results_policy_comparison'
    
    with open(results_file) as f:
        data = json.load(f)
    
    ld_episodes = data['lost_demand_dqn']['episodes']
    pf_episodes = data['profit_dqn']['episodes']
    
    # Plot 1: Lost demand comparison per episode
    fig, ax = plt.subplots(figsize=(12, 5))
    
    episodes = range(1, len(ld_episodes) + 1)
    ld_rates = [e['lost_demand_rate'] for e in ld_episodes]
    pf_rates = [e['lost_demand_rate'] for e in pf_episodes]
    
    ax.plot(episodes, ld_rates, 'b-', alpha=0.7, label='Lost-Demand DQN', linewidth=1.5)
    ax.plot(episodes, pf_rates, 'r-', alpha=0.7, label='Profit DQN', linewidth=1.5)
    
    ax.axhline(y=np.mean(ld_rates), color='blue', linestyle='--', alpha=0.5, label=f'LD Mean: {np.mean(ld_rates):.2f}%')
    ax.axhline(y=np.mean(pf_rates), color='red', linestyle='--', alpha=0.5, label=f'PF Mean: {np.mean(pf_rates):.2f}%')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Lost Demand Rate (%)', fontsize=12)
    ax.set_title('Policy Comparison: Lost Demand per Episode', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'policy_comparison_episodes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'policy_comparison_episodes.png'}")
    
    # Plot 2: Summary bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Lost demand comparison
    ax = axes[0]
    methods = ['Lost-Demand\nDQN', 'Profit\nDQN']
    means = [data['lost_demand_dqn']['aggregated']['avg_lost_rate'], 
             data['profit_dqn']['aggregated']['avg_lost_rate']]
    stds = [data['lost_demand_dqn']['aggregated']['std_lost_rate'],
            data['profit_dqn']['aggregated']['std_lost_rate']]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(methods, means, yerr=stds, capsize=8, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Lost Demand Rate (%)', fontsize=12)
    ax.set_title('Average Lost Demand', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{mean:.2f}%', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Right: Actions comparison
    ax = axes[1]
    actions = [data['lost_demand_dqn']['aggregated']['avg_actions'],
               data['profit_dqn']['aggregated']['avg_actions']]
    
    bars = ax.bar(methods, actions, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Actions per Episode', fontsize=12)
    ax.set_title('Agent Activity Level', fontsize=14, fontweight='bold')
    
    for bar, act in zip(bars, actions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{act:.0f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'policy_comparison_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'policy_comparison_summary.png'}")
    
    # Plot 3: Win/Loss chart
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ld_wins = sum(1 for ld, pf in zip(ld_rates, pf_rates) if ld < pf)
    pf_wins = sum(1 for ld, pf in zip(ld_rates, pf_rates) if pf < ld)
    ties = len(ld_rates) - ld_wins - pf_wins
    
    categories = ['Lost-Demand\nWins', 'Ties', 'Profit\nWins']
    values = [ld_wins, ties, pf_wins]
    colors = ['#3498db', '#95a5a6', '#e74c3c']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Number of Episodes', fontsize=12)
    ax.set_title(f'Head-to-Head Comparison ({len(ld_rates)} Episodes)', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), 
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'policy_comparison_wins.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'policy_comparison_wins.png'}")
    
    return output_dir


def main():
    print("=" * 70)
    print("EXPERIMENT EVALUATION AND PLOTTING")
    print("=" * 70)
    
    # 1. Evaluate economic sensitivity configs
    print("\n[1/3] Evaluating economic sensitivity experiments...")
    econ_results = evaluate_all_economic_configs()
    
    # Save results
    output_file = Path(__file__).parent / 'results_economic_sensitivity' / 'test_evaluation_results.json'
    with open(output_file, 'w') as f:
        # Convert to serializable format
        serializable = {}
        for k, v in econ_results.items():
            serializable[k] = {key: val for key, val in v.items() if key != 'episodes'}
        json.dump(serializable, f, indent=2)
    print(f"✅ Results saved to: {output_file}")
    
    # 2. Create economic sensitivity plots
    print("\n[2/3] Creating economic sensitivity plots...")
    create_economic_sensitivity_plots(econ_results)
    
    # 3. Create policy comparison plots
    print("\n[3/3] Creating policy comparison plots...")
    create_policy_comparison_plots()
    
    print("\n" + "=" * 70)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    
    # Print summary
    print("\nECONOMIC SENSITIVITY SUMMARY:")
    print("-" * 50)
    print(f"{'Config':<25} {'Lost Demand':>12} {'Actions':>10}")
    print("-" * 50)
    for name in econ_results:
        r = econ_results[name]
        print(f"{name:<25} {r['avg_lost_rate']:>11.2f}% {r['avg_actions']:>10.1f}")


if __name__ == "__main__":
    main()
