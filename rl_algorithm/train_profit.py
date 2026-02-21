"""
Simplified Profit-Based Training - Uses existing ContinuousTimeSimulator
with profit reward transformation.

Profit = trips_revenue - distance_cost - lost_demand_penalty
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime

from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator
from rl_algorithm.multi_agent_dqn import MultiAgentDQN


def train_profit_simple(
    gt_name='GT0',
    total_timesteps=100000,
    trip_base_fare=1.00,
    trip_per_km_rate=0.75,
    cost_per_km=1.00,
    lost_demand_penalty=5.00,
    output_dir=None,
    num_stations=10,
    num_vehicles=2,
    vehicle_capacity=15,
    hidden_activation='relu',
    output_activation=None
):
    """Train with profit-based rewards using existing simulator."""
    
    print("\n" + "="*60)
    print("PROFIT-BASED DQN TRAINING (Distance-Based Revenue)")
    print("="*60)
    
    if output_dir is None:
        output_dir = Path(f'results_profit/{gt_name}_distrev')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfig: {gt_name}, {total_timesteps:,} timesteps")
    print(f"Trip Revenue: ${trip_base_fare} + ${trip_per_km_rate}/km")
    print(f"Truck Cost: ${cost_per_km}/km, Lost Penalty: ${lost_demand_penalty}")
    
    # Data paths
    base_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / gt_name
    network_file = str(base_dir / f'{gt_name}_station_network.json')
    trips_file = str(base_dir / f'{gt_name}_trips_train.csv')
    
    # Load distance matrix for cost calculation
    with open(network_file) as f:
        network_data = json.load(f)
    distance_matrix = np.array(network_data['distance_matrix'])
    
    # Initialize simulator
    simulator = ContinuousTimeSimulator(
        network_file=network_file,
        trips_file=trips_file,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity
    )
    
    # Initialize agent
    fill_levels = [0.10, 0.50, 0.90]
    agent = MultiAgentDQN(
        num_stations=num_stations,
        num_vehicles=num_vehicles,
        hidden_dim=1024,
        buffer_capacity=10000,
        batch_size=256,
        learning_rate=2.5e-4,
        gamma=0.99,
        total_timesteps=total_timesteps,
        exploration_fraction=0.5,
        fill_levels=fill_levels,
        hidden_activation=hidden_activation,
        output_activation=output_activation
    )
    
    print(f"Activations: hidden={hidden_activation}, output={output_activation}")
    
    # Training history
    history = {
        'episode': [], 'timestep': [], 'profit': [],
        'revenue': [], 'cost': [], 'lost_demand': [],
        'lost_demand_rate': [], 'epsilon': [], 'loss': []
    }
    
    print("\nTraining...")
    
    episode = 0
    timestep = 0
    best_profit = float('-inf')
    
    # Track vehicle positions for distance calculation
    last_positions = {}
    
    while timestep < total_timesteps:
        # Cycle through training days
        day = (episode % 100) + 1
        simulator.reset(day)
        last_positions = {vid: v.current_station for vid, v in simulator.vehicles.items()}
        
        episode_profit = 0
        episode_revenue = 0
        episode_cost = 0
        episode_lost = 0
        loss = None
        
        # Track stats at episode start
        last_rentals = simulator.total_successful_rentals
        last_returns = simulator.total_successful_returns
        last_lost_rentals = simulator.total_lost_rentals
        last_lost_returns = simulator.total_lost_returns
        
        while not simulator.is_done() and timestep < total_timesteps:
            # Get next decision epoch
            vehicle_id, base_reward = simulator.get_next_decision_epoch()
            
            if vehicle_id is None:
                break
            
            # Calculate profit components since last decision
            new_rentals = simulator.total_successful_rentals - last_rentals
            new_returns = simulator.total_successful_returns - last_returns
            new_lost_rentals = simulator.total_lost_rentals - last_lost_rentals
            new_lost_returns = simulator.total_lost_returns - last_lost_returns
            
            # Revenue from successful trips (distance-based)
            # Use average trip distance from distance matrix for estimation
            avg_trip_dist = np.mean(distance_matrix[distance_matrix > 0])
            trips_completed = new_rentals + new_returns
            revenue = trips_completed * (trip_base_fare + avg_trip_dist * trip_per_km_rate)
            
            # Distance cost
            vehicle = simulator.vehicles[vehicle_id]
            current_station = vehicle.current_station
            last_station = last_positions.get(vehicle_id, current_station)
            if last_station != current_station:
                dist = distance_matrix[last_station - 1][current_station - 1]
                cost = dist * cost_per_km
            else:
                cost = 0
            
            # Lost demand penalty
            lost = new_lost_rentals + new_lost_returns
            lost_cost = lost * lost_demand_penalty
            
            # Profit reward
            profit_reward = revenue - cost - lost_cost
            
            # Update tracking
            last_positions[vehicle_id] = current_station
            last_rentals = simulator.total_successful_rentals
            last_returns = simulator.total_successful_returns
            last_lost_rentals = simulator.total_lost_rentals
            last_lost_returns = simulator.total_lost_returns
            
            episode_profit += profit_reward
            episode_revenue += revenue
            episode_cost += cost
            episode_lost += lost
            
            # Get state and select action
            state_dict = simulator.get_state()
            action_idx = agent.select_action(state_dict, vehicle_id)
            
            # Execute action
            action = agent.action_space.get_action(action_idx)
            simulator.execute_action(vehicle_id, action)
            
            # Store experience with profit reward
            next_state_dict = simulator.get_state()
            agent.store_experience(state_dict, action_idx, profit_reward, next_state_dict, False)
            
            # Train
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.train_step()
            timestep += 1
            
            if timestep % 10000 == 0:
                print(f"  Step {timestep:,}/{total_timesteps:,}")
        
        # Episode metrics
        metrics = simulator.get_metrics()
        lost_rate = metrics['total_lost_demand_rate']
        
        history['episode'].append(episode)
        history['timestep'].append(timestep)
        history['profit'].append(episode_profit)
        history['revenue'].append(episode_revenue)
        history['cost'].append(episode_cost)
        history['lost_demand'].append(episode_lost)
        history['lost_demand_rate'].append(lost_rate)
        history['epsilon'].append(agent.epsilon)
        history['loss'].append(loss if loss else 0)
        
        if episode % 50 == 0:
            print(f"Ep {episode}: Profit=${episode_profit:.2f}, Lost={lost_rate:.1f}%")
        
        if episode_profit > best_profit:
            best_profit = episode_profit
            agent.save(str(output_dir / f'{gt_name}_profit_dqn_best.pth'))
        
        episode += 1
    
    # Save final model and history
    agent.save(str(output_dir / f'{gt_name}_profit_dqn_final.pth'))
    
    with open(output_dir / f'{gt_name}_profit_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"  Episodes: {episode}, Best profit: ${best_profit:.2f}")
    print(f"  Saved to: {output_dir}")
    
    return agent, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', default='GT0')
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--num-stations', type=int, default=10)
    parser.add_argument('--num-vehicles', type=int, default=2)
    parser.add_argument('--vehicle-capacity', type=int, default=15)
    parser.add_argument('--trip-base-fare', type=float, default=1.00)
    parser.add_argument('--trip-per-km', type=float, default=0.75)
    parser.add_argument('--cost-per-km', type=float, default=1.00)
    parser.add_argument('--lost-penalty', type=float, default=5.00)
    parser.add_argument('--hidden-activation', default='relu', choices=['relu', 'leaky_relu', 'prelu', 'elu'])
    parser.add_argument('--output-activation', default=None, choices=[None, 'none', 'leaky_relu', 'prelu', 'elu'])
    args = parser.parse_args()
    
    # Handle 'none' string as None
    output_act = None if args.output_activation in [None, 'none'] else args.output_activation
    
    train_profit_simple(
        gt_name=args.gt,
        total_timesteps=args.timesteps,
        trip_base_fare=args.trip_base_fare,
        trip_per_km_rate=args.trip_per_km,
        cost_per_km=args.cost_per_km,
        lost_demand_penalty=args.lost_penalty,
        output_dir=args.output_dir,
        num_stations=args.num_stations,
        num_vehicles=args.num_vehicles,
        vehicle_capacity=args.vehicle_capacity,
        hidden_activation=args.hidden_activation,
        output_activation=output_act
    )
