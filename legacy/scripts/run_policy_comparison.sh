#!/bin/bash
# Lost-Demand vs Profit Policy Comparison for GT0
# Evaluates both policies on same test scenarios and compares behavior

set -e

# Configuration
GT_NAME="GT0"
NUM_EPISODES=50
BASE_DIR="results_policy_comparison"

echo "=============================================="
echo "Lost-Demand vs Profit Policy Comparison"
echo "GT: $GT_NAME | Test Episodes: $NUM_EPISODES"
echo "=============================================="

# Create results directory
mkdir -p "$BASE_DIR"

# Check if models exist
# Best configurations from experiments:
# Lost-Demand: ELU with 15-50-85 fill levels (best: 2.61% lost demand)
# Profit: Leaky ReLU with 10-50-90 fill levels (best: $61.25)
LOST_DEMAND_MODEL="results_GT0/activation_elu_fill_15_50_85/GT0_multi_agent_dqn_final.pth"
PROFIT_MODEL="results_profit_GT0/activation_leaky_relu_fill_10_50_90/GT0_profit_dqn_best.pth"

# Fill levels used by each model
LOST_DEMAND_FILL="15,50,85"
PROFIT_FILL="10,50,90"

if [ ! -f "$LOST_DEMAND_MODEL" ]; then
    echo "❌ Lost-demand model not found: $LOST_DEMAND_MODEL"
    echo "   Run ./run_gt0_experiments.sh first"
    exit 1
fi

if [ ! -f "$PROFIT_MODEL" ]; then
    echo "❌ Profit model not found: $PROFIT_MODEL"
    echo "   Run ./run_profit_gt0_experiments.sh first"
    exit 1
fi

echo ""
echo "Models found:"
echo "  Lost-Demand: $LOST_DEMAND_MODEL"
echo "  Profit:      $PROFIT_MODEL"

# Create comparison evaluation script
cat > "${BASE_DIR}/compare_policies.py" << 'PYTHON_SCRIPT'
"""
Compare Lost-Demand DQN vs Profit-Based DQN policies on same test scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np

from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator
from rl_algorithm.multi_agent_dqn import MultiAgentDQN


def evaluate_policy(
    model_path,
    output_activation,
    fill_levels,
    gt_name='GT0',
    num_episodes=50,
    num_stations=10,
    num_vehicles=2,
    vehicle_capacity=15,
):
    """Evaluate a single policy on TEST data with detailed metrics."""
    
    base_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / gt_name
    network_file = str(base_dir / f'{gt_name}_station_network.json')
    trips_file = str(base_dir / f'{gt_name}_trips_test.csv')
    
    simulator = ContinuousTimeSimulator(
        network_file=network_file,
        trips_file=trips_file,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        fill_levels=fill_levels
    )
    
    agent = MultiAgentDQN(
        num_stations=num_stations,
        num_vehicles=num_vehicles,
        hidden_activation='relu',
        output_activation=output_activation,
        fill_levels=fill_levels
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
        results.append({
            'day': day,
            'lost_demand_rate': metrics['total_lost_demand_rate'],
            'lost_rentals': metrics['lost_rentals'],
            'lost_returns': metrics['lost_returns'],
            'total_lost': metrics['total_lost_demand'],
            'successful_rentals': metrics['successful_rentals'],
            'successful_returns': metrics['successful_returns'],
            'action_count': action_count
        })
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lost-demand-model', required=True)
    parser.add_argument('--profit-model', required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--output', default='comparison_results.json')
    parser.add_argument('--lost-demand-fill', default='15,50,85')
    parser.add_argument('--profit-fill', default='10,50,90')
    args = parser.parse_args()
    
    ld_fill = [float(x)/100 for x in args.lost_demand_fill.split(',')]
    pf_fill = [float(x)/100 for x in args.profit_fill.split(',')]
    
    print("=" * 70)
    print("POLICY COMPARISON: Lost-Demand DQN vs Profit-Based DQN")
    print("=" * 70)
    
    # Evaluate Lost-Demand policy
    print(f"\n[1/2] Evaluating Lost-Demand DQN (ELU, fill={args.lost_demand_fill})...")
    lost_demand_results = evaluate_policy(
        model_path=args.lost_demand_model,
        output_activation='elu',
        fill_levels=ld_fill,
        num_episodes=args.episodes
    )
    
    # Evaluate Profit policy  
    print(f"[2/2] Evaluating Profit-Based DQN (Leaky ReLU, fill={args.profit_fill})...")
    profit_results = evaluate_policy(
        model_path=args.profit_model,
        output_activation='leaky_relu',
        fill_levels=pf_fill,
        num_episodes=args.episodes
    )
    
    # Aggregate metrics
    def aggregate(results):
        return {
            'avg_lost_rate': np.mean([r['lost_demand_rate'] for r in results]),
            'std_lost_rate': np.std([r['lost_demand_rate'] for r in results]),
            'avg_actions': np.mean([r['action_count'] for r in results]),
            'total_lost': sum([r['total_lost'] for r in results]),
            'total_successful': sum([r['successful_rentals'] + r['successful_returns'] for r in results])
        }
    
    ld_agg = aggregate(lost_demand_results)
    pf_agg = aggregate(profit_results)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'Lost-Demand DQN':>20} {'Profit DQN':>20}")
    print("-" * 70)
    print(f"{'Avg Lost Demand':<25} {ld_agg['avg_lost_rate']:>19.2f}% {pf_agg['avg_lost_rate']:>19.2f}%")
    print(f"{'Std Lost Demand':<25} {ld_agg['std_lost_rate']:>19.2f}% {pf_agg['std_lost_rate']:>19.2f}%")
    print(f"{'Avg Actions/Episode':<25} {ld_agg['avg_actions']:>20.1f} {pf_agg['avg_actions']:>20.1f}")
    print(f"{'Total Lost Trips':<25} {ld_agg['total_lost']:>20d} {pf_agg['total_lost']:>20d}")
    print(f"{'Total Successful Trips':<25} {ld_agg['total_successful']:>20d} {pf_agg['total_successful']:>20d}")
    print("-" * 70)
    
    # Difference analysis
    lost_diff = pf_agg['avg_lost_rate'] - ld_agg['avg_lost_rate']
    action_diff = pf_agg['avg_actions'] - ld_agg['avg_actions']
    
    print("\nDIFFERENCE ANALYSIS (Profit - Lost-Demand):")
    print(f"  Lost Demand Rate: {lost_diff:+.2f}% {'(worse)' if lost_diff > 0 else '(better)'}")
    print(f"  Actions/Episode: {action_diff:+.1f} {'(more)' if action_diff > 0 else '(less)'}")
    
    # Per-episode comparison
    print("\nPER-EPISODE BREAKDOWN (first 10 episodes):")
    print(f"{'Day':<6} {'LD Lost%':>10} {'PF Lost%':>10} {'LD Actions':>12} {'PF Actions':>12} {'Winner':>10}")
    print("-" * 65)
    for i in range(min(10, args.episodes)):
        ld = lost_demand_results[i]
        pf = profit_results[i]
        winner = "LD" if ld['lost_demand_rate'] < pf['lost_demand_rate'] else "PF" if pf['lost_demand_rate'] < ld['lost_demand_rate'] else "TIE"
        print(f"{ld['day']:<6} {ld['lost_demand_rate']:>9.2f}% {pf['lost_demand_rate']:>9.2f}% {ld['action_count']:>12} {pf['action_count']:>12} {winner:>10}")
    
    # Save results
    output = {
        'lost_demand_dqn': {
            'model': args.lost_demand_model,
            'activation': 'elu',
            'fill_levels': args.lost_demand_fill,
            'aggregated': ld_agg,
            'episodes': lost_demand_results
        },
        'profit_dqn': {
            'model': args.profit_model,
            'activation': 'leaky_relu',
            'fill_levels': args.profit_fill,
            'aggregated': pf_agg,
            'episodes': profit_results
        },
        'comparison': {
            'lost_rate_diff': lost_diff,
            'action_diff': action_diff
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
PYTHON_SCRIPT

echo ""
echo "Running policy comparison..."
echo ""

python "${BASE_DIR}/compare_policies.py" \
    --lost-demand-model "$LOST_DEMAND_MODEL" \
    --profit-model "$PROFIT_MODEL" \
    --lost-demand-fill "$LOST_DEMAND_FILL" \
    --profit-fill "$PROFIT_FILL" \
    --episodes "$NUM_EPISODES" \
    --output "${BASE_DIR}/comparison_results.json"

echo ""
echo "=============================================="
echo "Policy comparison complete!"
echo "Results saved in: $BASE_DIR/"
echo "=============================================="
