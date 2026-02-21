"""
Training script for Multi-Agent DQN.

Trains 10 cooperative vehicle agents using continuous-time event-driven simulation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import datetime

from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator
from rl_algorithm.multi_agent_dqn import MultiAgentDQN


def train_multi_agent_dqn(
    gt_name='GT1',
    total_timesteps=300000,
    save_freq_timesteps=10000,
    optimizer='adam',
    optimizer_kwargs=None,
    fill_levels=None,
    hidden_activation='relu',
    output_activation=None,
    output_dir=None,
    exploration_fraction=0.5,
    heuristic_routing=False,
    num_stations=60,
    num_vehicles=4,
    vehicle_capacity=40
):
    """
    Train multi-agent DQN on bike rebalancing.
    
    Args:
        gt_name: Ground truth name ('GT0', 'GT1', or 'GT2')
        total_timesteps: Total training timesteps (paper: 3,000,000)
        save_freq_timesteps: Save checkpoint every N timesteps
        optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
        optimizer_kwargs: Additional optimizer parameters (dict)
        output_dir: Directory to save results
        exploration_fraction: Fraction of timesteps for epsilon decay (paper: 0.5)
        heuristic_routing: Use simplified action space with heuristic routing (Section 5.2.2)
        num_stations: Number of stations (default: 60, GT0: 10)
        num_vehicles: Number of vehicles (default: 4, GT0: 2)
        vehicle_capacity: Vehicle bike capacity (default: 40, GT0: 15)
    
    Returns:
        tuple: (agent, training_history)
    """
    print("\n" + "="*70)
    print(f"TRAINING MULTI-AGENT DQN - {gt_name}")
    print("="*70)
    print("\nContinuous-Time Event-Driven Multi-Agent Learning")
    print(f"{num_vehicles} vehicles acting asynchronously, shared DQN network")
    
    # Setup output directory with fill-level-specific naming
    if output_dir is None:
        # Create descriptive directory name based on fill levels
        if fill_levels is not None:
            fill_suffix = '_'.join([str(int(f*100)) for f in fill_levels])
            dir_name = f"{gt_name}_fill_{fill_suffix}"
        else:
            # Default fill levels [10, 50, 90]
            dir_name = f"{gt_name}_fill_10_50_90"
        output_dir = Path(__file__).parent.parent / 'results' / dir_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also save fill level config to a metadata file
    fill_levels_used = fill_levels if fill_levels is not None else [0.10, 0.50, 0.90]
    metadata = {
        'gt_name': gt_name,
        'fill_levels': fill_levels_used,
        'optimizer': optimizer,
        'hidden_activation': hidden_activation,
        'output_activation': output_activation
    }
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Fill levels: {[f'{f*100:.0f}%' for f in fill_levels_used]}")
    
    # Initialize simulator
    print(f"\nInitializing continuous-time simulator...")
    print(f"  Stations: {num_stations}, Vehicles: {num_vehicles}, Capacity: {vehicle_capacity}")
    base_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / gt_name
    simulator = ContinuousTimeSimulator(
        network_file=str(base_dir / f'{gt_name}_station_network.json'),
        trips_file=str(base_dir / f'{gt_name}_trips_train.csv'),
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity
    )
    
    # Initialize agent
    print(f"\nInitializing multi-agent DQN...")
    agent = MultiAgentDQN(
        num_stations=num_stations,
        num_vehicles=num_vehicles,
        hidden_dim=1024,             # Paper: 1024 first layer
        buffer_capacity=10000,       # Paper: 10,000
        batch_size=256,              # Paper: 256
        learning_rate=2.5e-4,        # Paper: 2.5e-4
        gamma=0.99,                  # Paper: 0.99
        epsilon_start=1.0,           # Paper: 1.0
        epsilon_end=0.05,            # Paper: 0.05
        epsilon_schedule='linear',   # Paper: Linear schedule
        exploration_fraction=exploration_fraction,  # Paper: 0.5
        total_timesteps=total_timesteps,            # Paper: 3,000,000
        optimizer=optimizer,         # Configurable: 'adam', 'sgd', 'rmsprop'
        optimizer_kwargs=optimizer_kwargs,  # E.g., {'momentum': 0.9} for SGD
        fill_levels=fill_levels,     # Configurable: [0.10, 0.50, 0.90] (default)
        hidden_activation=hidden_activation,  # 'relu', 'leaky_relu', 'prelu', 'elu'
        output_activation=output_activation,  # None, 'leaky_relu', 'prelu', 'elu'
        heuristic_routing=heuristic_routing   # Simplified action space (Section 5.2.2)
    )
    
    if heuristic_routing:
        print("  Mode: HEURISTIC ROUTING (DQN decides fill level only)")
    
    # Training history
    history = {
        'episode': [],
        'total_reward': [],
        'lost_demand': [],
        'lost_demand_rate': [],
        'epsilon': [],
        'loss': [],
        'buffer_size': [],
        'num_decisions': []
    }
    
    # Training loop - timestep-based (matching base paper)
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING - {total_timesteps:,} TIMESTEPS")
    print(f"{'='*70}\n")
    
    episode = 0
    last_save_timestep = 0
    
    while agent.timestep < total_timesteps:
        episode += 1
        
        # Cycle through training days (1-100)
        day = ((episode - 1) % 100) + 1
        
        # Reset simulator for new episode
        state_dict = simulator.reset(day)
        
        episode_reward = 0
        episode_losses = []
        num_decisions = 0
        
        # Run episode (continuous-time event-driven)
        while not simulator.is_done() and agent.timestep < total_timesteps:
            # Get next decision epoch (vehicle arrival)
            vehicle_id, reward = simulator.get_next_decision_epoch()
            
            if vehicle_id is None:
                # Episode ended
                break
            
            # Update cumulative reward
            episode_reward += reward
            
            # Get current state
            state_dict = simulator.get_state()
            
            # Agent selects action for this vehicle
            action_idx = agent.select_action(state_dict, vehicle_id)
            
            # Convert to simulator action
            if heuristic_routing:
                # Simplified mode: DQN selects fill level, heuristic selects station
                fill_level_idx = agent.action_space.get_action(action_idx)
                next_station = agent.action_space.select_station_heuristic(state_dict, vehicle_id)
                action = (next_station, fill_level_idx)
            else:
                # Full mode: DQN selects both station and fill level
                action = agent.action_space.get_action(action_idx)
            
            # Execute action in simulator
            simulator.execute_action(vehicle_id, action)
            
            # Get next state (will be available at next decision epoch)
            next_state_dict = simulator.get_state()
            
            # Store experience
            # Note: done is always False until episode ends
            agent.store_experience(state_dict, action_idx, reward, next_state_dict, False)
            
            # Train
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
            
            num_decisions += 1
        
        # End episode
        agent.end_episode()
        
        # Get episode metrics
        metrics = simulator.get_metrics()
        
        # Record history
        history['episode'].append(episode)
        history['total_reward'].append(episode_reward)
        history['lost_demand'].append(metrics['total_lost_demand'])
        history['lost_demand_rate'].append(metrics['total_lost_demand_rate'])
        history['epsilon'].append(agent.epsilon)
        history['loss'].append(np.mean(episode_losses) if episode_losses else 0)
        history['buffer_size'].append(len(agent.replay_buffer))
        history['num_decisions'].append(num_decisions)
        
        # Print progress every 10 episodes
        if episode % 10 == 0 or episode == 1:
            progress_pct = (agent.timestep / total_timesteps) * 100
            cycle = (episode - 1) // 100 + 1
            print(f"Episode {episode} [Day {day}, Cycle {cycle}] (Timestep {agent.timestep:,}/{total_timesteps:,} = {progress_pct:.1f}%):")
            print(f"  Lost demand: {metrics['total_lost_demand']} ({metrics['total_lost_demand_rate']:.2f}%)")
            print(f"  Reward: {episode_reward:.1f}")
            print(f"  Decisions: {num_decisions}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            if episode_losses:
                print(f"  Avg loss: {np.mean(episode_losses):.4f}")
            print(f"  Buffer: {len(agent.replay_buffer)}/{agent.replay_buffer.capacity}")
        
        # Save checkpoint based on timesteps
        if agent.timestep - last_save_timestep >= save_freq_timesteps:
            checkpoint_path = output_dir / f'checkpoint_ts{agent.timestep}.pth'
            agent.save(str(checkpoint_path))
            last_save_timestep = agent.timestep
    
    # Save final model
    final_model_path = output_dir / f'{gt_name}_multi_agent_dqn_final.pth'
    agent.save(str(final_model_path))
    
    # Save training history
    history_path = output_dir / f'{gt_name}_training_history.json'
    # Convert to serializable format
    history_serializable = {k: [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in vs] 
                           for k, vs in history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    print(f"\n✓ Training history saved to {history_path}")
    
    # Plot training curves
    plot_training_curves(history, output_dir, gt_name)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    
    return agent, history


def plot_training_curves(history, output_dir, gt_name):
    """
    Plot training curves matching base paper terminology.
    
    Creates 4 plots:
    1. Episodic Return (reward per episode = negative lost demand)
    2. TD Loss (temporal difference loss from Q-learning)
    3. Lost Demand Rate (performance metric)
    4. Epsilon (exploration rate)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{gt_name} Multi-Agent DQN Training Progress', fontsize=16, fontweight='bold')
    
    episodes = history['episode']
    
    # Plot 1: Episodic Return (Paper Figure 4a)
    axes[0, 0].plot(episodes, history['total_reward'], 'b-', alpha=0.5, linewidth=1, label='Raw')
    if len(episodes) > 10:
        smoothed = smooth(history['total_reward'], 10)
        axes[0, 0].plot(episodes, smoothed, 'darkblue', linewidth=2.5, label='Smoothed (10-ep MA)')
    axes[0, 0].set_xlabel('Episode (Step)', fontsize=11)
    axes[0, 0].set_ylabel('Episodic Return', fontsize=11)
    axes[0, 0].set_title('Episodic Return (Cumulative Reward)', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot 2: TD Loss (Paper Figure 4b)
    axes[0, 1].plot(episodes, history['loss'], 'orange', alpha=0.5, linewidth=1, label='Raw')
    if len(episodes) > 10:
        smoothed = smooth(history['loss'], 10)
        axes[0, 1].plot(episodes, smoothed, 'darkorange', linewidth=2.5, label='Smoothed (10-ep MA)')
    axes[0, 1].set_xlabel('Episode (Step)', fontsize=11)
    axes[0, 1].set_ylabel('TD Loss', fontsize=11)
    axes[0, 1].set_title('TD Loss (Temporal Difference Error)', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Lost Demand Rate (Performance metric)
    axes[1, 0].plot(episodes, history['lost_demand_rate'], 'red', alpha=0.5, linewidth=1, label='Raw')
    if len(episodes) > 10:
        smoothed = smooth(history['lost_demand_rate'], 10)
        axes[1, 0].plot(episodes, smoothed, 'darkred', linewidth=2.5, label='Smoothed (10-ep MA)')
    axes[1, 0].axhline(y=15.41, color='green', linestyle='--', linewidth=2, label='Baseline (15.41%)', alpha=0.7)
    axes[1, 0].axhline(y=10, color='purple', linestyle=':', linewidth=2, label='Target (10%)', alpha=0.7)
    axes[1, 0].set_xlabel('Episode (Step)', fontsize=11)
    axes[1, 0].set_ylabel('Lost Demand Rate (%)', fontsize=11)
    axes[1, 0].set_title('Lost Demand Rate (Performance)', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Epsilon (Exploration rate)
    axes[1, 1].plot(episodes, history['epsilon'], 'purple', linewidth=2)
    axes[1, 1].fill_between(episodes, history['epsilon'], alpha=0.2, color='purple')
    axes[1, 1].axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='ε_end = 0.05', alpha=0.7)
    axes[1, 1].set_xlabel('Episode (Step)', fontsize=11)
    axes[1, 1].set_ylabel('Epsilon (ε)', fontsize=11)
    axes[1, 1].set_title('Exploration Rate (Linear Schedule)', fontsize=13, fontweight='bold')
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / f'{gt_name}_training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to {plot_path}")
    plt.close()


def smooth(values, window=10):
    """Smooth values using moving average."""
    if len(values) < window:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(np.mean(values[start:end]))
    
    return smoothed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Multi-Agent DQN for bike rebalancing')
    parser.add_argument('--gt', type=str, default='GT1', choices=['GT0', 'GT1', 'GT2'],
                      help='Ground truth to train on')
    parser.add_argument('--timesteps', type=int, default=300000,
                      help='Total training timesteps (paper: 3,000,000, default: 300,000)')
    parser.add_argument('--save-freq', type=int, default=10000,
                      help='Save checkpoint every N timesteps (default: 10,000)')
    parser.add_argument('--exploration-fraction', type=float, default=0.5,
                      help='Fraction of timesteps for epsilon decay (paper: 0.5)')
    parser.add_argument('--optimizer', type=str, default='adam', 
                      choices=['adam', 'sgd', 'rmsprop'],
                      help='Optimizer type (default: adam, paper-literal: sgd)')
    parser.add_argument('--momentum', type=float, default=None,
                      help='Momentum for SGD optimizer (e.g., 0.9)')
    
    # Network architecture
    parser.add_argument('--hidden-activation', type=str, default='relu',
                      choices=['relu', 'leaky_relu', 'prelu', 'elu'],
                      help='Activation function for hidden layers (default: relu)')
    parser.add_argument('--output-activation', type=str, default=None,
                      choices=['none', 'leaky_relu', 'prelu', 'elu'],
                      help='Activation function for output layer (default: none)')
    
    # Action space
    parser.add_argument('--fill-levels', type=str, default='10,50,90',
                      help='Fill levels as comma-separated percentages (default: 10,50,90)')
    parser.add_argument('--heuristic-routing', action='store_true',
                      help='Use simplified action space with heuristic routing (Section 5.2.2)')
    
    # Environment configuration (for GT0 toy model)
    parser.add_argument('--num-stations', type=int, default=60,
                      help='Number of stations (default: 60, GT0: 10)')
    parser.add_argument('--num-vehicles', type=int, default=4,
                      help='Number of vehicles (default: 4, GT0: 2)')
    parser.add_argument('--vehicle-capacity', type=int, default=40,
                      help='Vehicle bike capacity (default: 40, GT0: 15)')
    
    # Output directory
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for results (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Build optimizer kwargs
    optimizer_kwargs = {}
    if args.momentum is not None:
        optimizer_kwargs['momentum'] = args.momentum
    
    # Parse fill levels
    fill_levels = [float(x)/100.0 for x in args.fill_levels.split(',')]
    
    # Parse output activation (handle 'none')
    output_activation = None if args.output_activation == 'none' or args.output_activation is None else args.output_activation
    
    # Train agent
    agent, history = train_multi_agent_dqn(
        gt_name=args.gt,
        total_timesteps=args.timesteps,
        save_freq_timesteps=args.save_freq,
        optimizer=args.optimizer,
        optimizer_kwargs=optimizer_kwargs if optimizer_kwargs else None,
        fill_levels=fill_levels,
        hidden_activation=args.hidden_activation,
        output_activation=output_activation,
        output_dir=args.output_dir,
        exploration_fraction=args.exploration_fraction,
        heuristic_routing=args.heuristic_routing,
        num_stations=args.num_stations,
        num_vehicles=args.num_vehicles,
        vehicle_capacity=args.vehicle_capacity
    )
    
    print("\n🎉 Training complete! Multi-agent DQN ready for evaluation.")
