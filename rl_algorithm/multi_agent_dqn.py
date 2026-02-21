"""
Multi-Agent DQN for Cooperative Bike Rebalancing.

All vehicles (agents) share a single DQN network and cooperatively
minimize total lost demand.

Based on Base Paper Section 4.3 - MARL Methodology with Deep Q-Network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .state_encoder import MultiAgentStateEncoder
from rl_algorithm.action_space import FillLevelActionSpace, HeuristicRoutingActionSpace
from .dqn_network import DQNNetwork, ReplayBuffer


class MultiAgentDQN:
    """
    Multi-agent DQN with shared network.
    
    Key features:
    - All 10 vehicles share one network (cooperative)
    - Each vehicle acts independently upon arrival
    - Shared experience replay buffer
    - Common goal: minimize total lost demand
    """
    
    def __init__(
        self,
        num_stations=60,
        num_vehicles=4,
        hidden_dim=1024,
        buffer_capacity=10000,
        batch_size=256,
        learning_rate=2.5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_schedule='linear',
        exploration_fraction=0.5,
        total_timesteps=300000,  # Paper: 3,000,000 (default 300K for faster iteration)
        optimizer='adam',
        optimizer_kwargs=None,
        fill_levels=None,
        hidden_activation='relu',
        output_activation=None,
        heuristic_routing=False,
        device=None,
        seed=42
    ):
        """
        Initialize multi-agent DQN.
        
        Args:
            num_stations: Number of stations
            num_vehicles: Number of vehicles (agents)
            hidden_dim: Hidden layer size
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            learning_rate: Learning rate (base for adaptive optimizers)
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_schedule: 'linear' or 'exponential'
            exploration_fraction: Fraction of timesteps for epsilon decay
            total_timesteps: Total timesteps for epsilon schedule
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            optimizer_kwargs: Additional optimizer parameters (e.g., momentum for SGD)
            fill_levels: List of fill levels for action space (default: [0.10, 0.50, 0.90])
            hidden_activation: Activation for hidden layers ('relu', 'leaky_relu', 'prelu', 'elu')
            output_activation: Activation for output layer (None, 'leaky_relu', 'prelu', 'elu')
            heuristic_routing: Use simplified action space with heuristic routing (Section 5.2.2)
            device: torch device
            seed: Random seed
        """
        self.num_stations = num_stations
        self.num_vehicles = num_vehicles
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_schedule = epsilon_schedule
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps
        # Target network updates every 1/30 of total timesteps (e.g., 10K for 300K total)
        self.target_update_freq = max(1, total_timesteps // 30)
        self.last_target_update = 0
        self.optimizer_name = optimizer.lower()
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.fill_levels = fill_levels if fill_levels is not None else [0.10, 0.50, 0.90]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.heuristic_routing = heuristic_routing
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        # Timestep counter for linear epsilon schedule
        self.timestep = 0
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize components
        self.state_encoder = MultiAgentStateEncoder(num_stations, num_vehicles)
        
        # Choose action space based on routing mode
        if heuristic_routing:
            self.action_space = HeuristicRoutingActionSpace(num_stations=num_stations, fill_levels=self.fill_levels)
        else:
            self.action_space = FillLevelActionSpace(num_stations=num_stations, fill_levels=self.fill_levels)
        
        state_dim = self.state_encoder.get_state_dim()
        action_dim = self.action_space.get_num_actions()
        
        # Shared policy network (all agents use this)
        self.policy_net = DQNNetwork(
            state_dim, action_dim, hidden_dim,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation
        ).to(self.device)
        
        # Target network (for stability)
        self.target_net = DQNNetwork(
            state_dim, action_dim, hidden_dim,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network always in eval mode
        
        # Optimizer - configurable for experimentation
        self.optimizer = self._create_optimizer(learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Shared replay buffer (all vehicles contribute)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)
        
        # Training statistics
        self.episode_count = 0
        self.training_step = 0
        self.losses = []
        
        print(f"\nMultiAgentDQN initialized:")
        print(f"  Device: {self.device}")
        print(f"  Agents (vehicles): {num_vehicles}")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Shared network parameters: {self.policy_net.count_parameters():,}")
        print(f"  Optimizer: {self.optimizer_name}")
        if self.optimizer_kwargs:
            print(f"  Optimizer kwargs: {self.optimizer_kwargs}")
        print(f"  Fill levels: {[f'{f*100:.0f}%' for f in self.fill_levels]}")
        print(f"  Hidden activation: {self.hidden_activation}")
        print(f"  Output activation: {self.output_activation if self.output_activation else 'none'}")
        print(f"  Device: {self.device}")
    
    def _create_optimizer(self, learning_rate):
        """
        Create optimizer based on configuration.
        
        Args:
            learning_rate: Base learning rate
            
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        params = self.policy_net.parameters()
        
        if self.optimizer_name == 'adam':
            # Adam: Adaptive learning rate with momentum
            # Default in modern DQN, provides stability
            optimizer = optim.Adam(params, lr=learning_rate, **self.optimizer_kwargs)
            
        elif self.optimizer_name == 'sgd':
            # Vanilla SGD: Fixed learning rate
            # What the paper literally says "gradient descent"
            # Can add momentum via optimizer_kwargs={'momentum': 0.9}
            optimizer = optim.SGD(params, lr=learning_rate, **self.optimizer_kwargs)
            
        elif self.optimizer_name == 'rmsprop':
            # RMSprop: Adaptive learning rate
            # Used in original DQN paper (Mnih et al. 2015)
            optimizer = optim.RMSprop(params, lr=learning_rate, **self.optimizer_kwargs)
            
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}. "
                           f"Choose from: 'adam', 'sgd', 'rmsprop'")
        
        return optimizer
    
    def select_action(self, state_dict, vehicle_id, epsilon=None):
        """
        Select action for a specific vehicle using epsilon-greedy policy.
        
        Args:
            state_dict: Global state dictionary
            vehicle_id: Which vehicle is making decision
            epsilon: Exploration rate (uses self.epsilon if None)
        
        Returns:
            int: Action index
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Get valid actions for this vehicle
        valid_actions = self.action_space.get_valid_actions(state_dict, vehicle_id)
        
        # Epsilon-greedy
        if np.random.random() < epsilon:
            # Explore: random valid action
            return np.random.choice(valid_actions)
        else:
            # Exploit: best action according to Q-network
            return self._get_best_action(state_dict, valid_actions)
    
    def _get_best_action(self, state_dict, valid_actions):
        """Get best action according to policy network."""
        # Encode state
        state_vector = self.state_encoder.encode(state_dict)
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        
        # Mask invalid actions
        masked_q = self.action_space.mask_invalid_actions(q_values, valid_actions)
        
        # Select best valid action
        best_action = np.argmax(masked_q)
        return best_action
    
    def store_experience(self, state_dict, action, reward, next_state_dict, done):
        """
        Store experience in replay buffer.
        
        Args:
            state_dict: State dictionary
            action: Action taken
            reward: Reward received
            next_state_dict: Next state dictionary
            done: Whether episode ended
        """
        state = self.state_encoder.encode(state_dict)
        next_state = self.state_encoder.encode(next_state_dict)
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step (Q-learning update).
        
        Returns:
            float: Loss value, or None if not enough samples
        """
        # Check if enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Record statistics
        self.training_step += 1
        self.timestep += 1  # Increment timestep for epsilon schedule
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        # Update epsilon (linear schedule based on timesteps)
        self.update_epsilon()
        
        # Update target network based on timesteps (every 1/30 of total)
        if self.timestep - self.last_target_update >= self.target_update_freq:
            self.update_target_network()
            self.last_target_update = self.timestep
            print(f"  🎯 Target network updated (timestep {self.timestep:,})")
        
        return loss_value
    
    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        """Update epsilon based on schedule (linear or exponential)."""
        if self.epsilon_schedule == 'linear':
            # Linear annealing over exploration_fraction of total_timesteps
            exploration_timesteps = int(self.total_timesteps * self.exploration_fraction)
            if self.timestep < exploration_timesteps:
                # Linear decay from epsilon_start to epsilon_end
                progress = self.timestep / exploration_timesteps
                self.epsilon = self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)
            else:
                # Stay at epsilon_end after exploration phase
                self.epsilon = self.epsilon_end
        else:
            # Legacy exponential decay (kept for compatibility)
            self.epsilon = max(self.epsilon_end, self.epsilon * 0.995)
    
    def end_episode(self):
        """Call at end of episode to update counters."""
        self.episode_count += 1
        # Note: Target network and epsilon are now updated in train_step based on timesteps
    
    def save(self, filepath):
        """Save agent state."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'losses': self.losses
        }, filepath)
        print(f"✓ Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode_count = checkpoint['episode_count']
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        print(f"✓ Agent loaded from {filepath}")
    
    def get_statistics(self):
        """Get training statistics."""
        recent_losses = self.losses[-100:] if len(self.losses) > 0 else []
        
        return {
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'buffer_utilization': len(self.replay_buffer) / self.replay_buffer.capacity,
            'recent_loss': {
                'mean': np.mean(recent_losses) if recent_losses else 0,
                'std': np.std(recent_losses) if recent_losses else 0,
                'min': np.min(recent_losses) if recent_losses else 0,
                'max': np.max(recent_losses) if recent_losses else 0
            }
        }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing Multi-Agent DQN")
    print("="*70)
    
    # Create agent
    agent = MultiAgentDQN(
        num_stations=60,
        num_vehicles=10,
        batch_size=32
    )
    
    # Create dummy state
    from datetime import datetime
    
    dummy_state = {
        'station_inventories': {i: 10 for i in range(1, 61)},
        'station_occupancies': {i: 0.5 for i in range(1, 61)},
        'vehicle_states': {
            i: {
                'current_station': i * 6,
                'destination': None,
                'inventory': 5,
                'capacity': 20,
                'occupancy': 0.25
            }
            for i in range(1, 11)
        },
        'current_time': datetime(2019, 5, 1, 8, 0),
        'upcoming_trips': []
    }
    
    print(f"\n✓ Testing action selection:")
    action = agent.select_action(dummy_state, vehicle_id=1, epsilon=0.0)
    print(f"  Vehicle 1 action: {action}")
    print(f"  Description: {agent.action_space.describe_action(action)}")
    
    print(f"\n✓ Testing experience storage:")
    for i in range(50):
        state = dummy_state
        action = agent.select_action(state, vehicle_id=(i % 10) + 1)
        reward = -np.random.rand() * 10
        next_state = dummy_state
        done = False
        
        agent.store_experience(state, action, reward, next_state, done)
    
    print(f"  Stored 50 experiences")
    print(f"  Buffer size: {len(agent.replay_buffer)}")
    
    print(f"\n✓ Testing training:")
    losses = []
    for i in range(5):
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)
    
    print(f"  Training steps: {len(losses)}")
    if losses:
        print(f"  Avg loss: {np.mean(losses):.4f}")
    
    print(f"\n✓ Agent statistics:")
    stats = agent.get_statistics()
    print(f"  Episodes: {stats['episode_count']}")
    print(f"  Epsilon: {stats['epsilon']:.4f}")
    print(f"  Buffer: {stats['buffer_size']}")
    
    print("\n✅ Multi-Agent DQN test passed!")
