"""
Evaluate Static Baseline using the same simulator as DQN.

This script runs the simulator with the static initial inventory
but NO dynamic rebalancing (vehicles don't move). This provides
a fair comparison baseline for the DQN agent.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator


def evaluate_static_baseline(gt_name='GT1', num_episodes=50, use_test_data=True):
    """
    Evaluate the static rebalancing baseline by running the simulator
    with NO dynamic rebalancing actions.
    
    Args:
        gt_name: 'GT1' or 'GT2'
        num_episodes: Number of episodes to evaluate
        use_test_data: If True, use test trips; else use train trips
    
    Returns:
        dict: Evaluation results
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING STATIC BASELINE FOR {gt_name}")
    print(f"{'='*70}")
    print(f"Method: Run simulator with static inventory, NO vehicle actions")
    print(f"Episodes: {num_episodes}")
    print(f"Data: {'Test' if use_test_data else 'Train'}")
    
    # Initialize simulator
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / gt_name
    network_file = data_dir / f'{gt_name}_station_network.json'
    trips_file = data_dir / f'{gt_name}_trips_{"test" if use_test_data else "train"}.csv'
    
    print(f"\nLoading data from: {data_dir}")
    
    sim = ContinuousTimeSimulator(
        network_file=str(network_file),
        trips_file=str(trips_file)
    )
    
    results = {
        'lost_demand': [],
        'lost_demand_rate': [],
        'lost_rentals': [],
        'lost_returns': [],
        'total_trips': []
    }
    
    # Get available days from trips dataframe
    available_days = sorted(sim.trips_df['departure_time'].dt.date.unique())
    num_days = len(available_days)
    print(f"Available days: {num_days}")
    
    print(f"\nRunning {num_episodes} episodes with NO rebalancing...")
    
    for episode in range(1, num_episodes + 1):
        # Reset to a specific day (1-indexed) with static initial inventory
        day_idx = ((episode - 1) % num_days) + 1
        sim.reset(day=day_idx)
        
        # Run the episode WITHOUT any vehicle actions
        # Just let the simulation run until the horizon ends
        import heapq
        from rl_algorithm.events import CustomerRental, CustomerReturn, VehicleArrival
        
        while sim.current_time < sim.episode_end_time:
            # Process all events until end of horizon
            # No vehicle actions taken - static only
            if not sim.event_queue:
                break
            
            # Get next event (peek without popping)
            event = sim.event_queue[0]
            
            if event.time >= sim.episode_end_time:
                break
            
            # Pop and process the event
            heapq.heappop(sim.event_queue)
            sim.current_time = event.time
            
            if isinstance(event, CustomerRental):
                sim._process_rental(event)
            elif isinstance(event, CustomerReturn):
                sim._process_return(event)
            elif isinstance(event, VehicleArrival):
                # Vehicle arrives but takes no action - stays idle
                pass
        
        # Get metrics for this episode
        metrics = sim.get_metrics()
        
        results['lost_demand'].append(metrics['total_lost_demand'])
        results['lost_demand_rate'].append(metrics['total_lost_demand_rate'])
        results['lost_rentals'].append(metrics['lost_rentals'])
        results['lost_returns'].append(metrics['lost_returns'])
        results['total_trips'].append(metrics['total_rentals'] + metrics['lost_rentals'])
        
        if episode % 10 == 0 or episode == 1:
            print(f"  Episode {episode}/{num_episodes}: Lost demand = {metrics['total_lost_demand']} ({metrics['total_lost_demand_rate']:.2f}%)")
    
    # Calculate summary statistics
    avg_lost_demand = np.mean(results['lost_demand'])
    avg_lost_demand_rate = np.mean(results['lost_demand_rate'])
    std_lost_demand_rate = np.std(results['lost_demand_rate'])
    
    print(f"\n{'='*70}")
    print("STATIC BASELINE RESULTS (No Dynamic Rebalancing)")
    print(f"{'='*70}")
    print(f"\nLost Demand Rate:")
    print(f"  Mean: {avg_lost_demand_rate:.2f}%")
    print(f"  Std Dev: {std_lost_demand_rate:.2f}%")
    print(f"  Min: {np.min(results['lost_demand_rate']):.2f}%")
    print(f"  Max: {np.max(results['lost_demand_rate']):.2f}%")
    print(f"  Median: {np.median(results['lost_demand_rate']):.2f}%")
    
    print(f"\nAverage per episode:")
    print(f"  Total trips: {np.mean(results['total_trips']):.1f}")
    print(f"  Lost rentals: {np.mean(results['lost_rentals']):.1f}")
    print(f"  Lost returns: {np.mean(results['lost_returns']):.1f}")
    print(f"  Total lost: {avg_lost_demand:.1f}")
    
    # Save results
    output = {
        'gt_name': gt_name,
        'num_episodes': num_episodes,
        'use_test_data': use_test_data,
        'avg_lost_demand_rate': float(avg_lost_demand_rate),
        'std_lost_demand_rate': float(std_lost_demand_rate),
        'min_lost_demand_rate': float(np.min(results['lost_demand_rate'])),
        'max_lost_demand_rate': float(np.max(results['lost_demand_rate'])),
        'median_lost_demand_rate': float(np.median(results['lost_demand_rate'])),
        'avg_lost_demand': float(avg_lost_demand),
        'episodes': results
    }
    
    output_file = data_dir / f'{gt_name}_static_baseline_evaluation.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', default='GT1', choices=['GT0', 'GT1', 'GT2'])
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--train', action='store_true', help='Use training data instead of test')
    args = parser.parse_args()
    
    evaluate_static_baseline(
        gt_name=args.gt,
        num_episodes=args.episodes,
        use_test_data=not args.train
    )
