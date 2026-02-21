"""
DQN Network for Multi-Agent Rebalancing.

Shared network architecture used by all vehicle agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for multi-agent bike rebalancing.
    
    All vehicles share this network (cooperative learning).
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=512, 
                 hidden_activation='relu', output_activation=None):
        """
        Initialize DQN network.
        
        Args:
            state_dim: Dimension of state vector (~84 for multi-agent)
            action_dim: Number of actions (180 = 60 stations × 3 fill levels)
            hidden_dim: Size of hidden layers (default: 512)
            hidden_activation: Activation for hidden layers ('relu', 'leaky_relu', 'prelu', 'elu')
            output_activation: Activation for output layer (None, 'leaky_relu', 'prelu', 'elu')
        """
        super(DQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation.lower()
        self.output_activation = output_activation.lower() if output_activation else None
        
        # Network layers
        # Larger network for multi-agent complexity
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Activation layers (for PReLU which has learnable parameters)
        if self.hidden_activation == 'prelu':
            self.hidden_act1 = nn.PReLU()
            self.hidden_act2 = nn.PReLU()
        
        if self.output_activation == 'prelu':
            self.output_act = nn.PReLU()
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"DQNNetwork initialized:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Hidden activation: {self.hidden_activation}")
        print(f"  Output activation: {self.output_activation if self.output_activation else 'none'}")
        print(f"  Total parameters: {self.count_parameters():,}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def _apply_activation(self, x, activation_type, activation_layer=None):
        """Apply activation function."""
        if activation_type == 'relu':
            return F.relu(x)
        elif activation_type == 'leaky_relu':
            return F.leaky_relu(x, negative_slope=0.01)
        elif activation_type == 'elu':
            return F.elu(x)
        elif activation_type == 'prelu':
            return activation_layer(x)
        else:
            return x  # No activation
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: State tensor (batch_size, state_dim) or (state_dim,)
        
        Returns:
            Q-values tensor (batch_size, action_dim) or (action_dim,)
        """
        # Hidden layer 1
        x = self.fc1(state)
        x = self._apply_activation(x, self.hidden_activation, 
                                   getattr(self, 'hidden_act1', None))
        
        # Hidden layer 2
        x = self.fc2(x)
        x = self._apply_activation(x, self.hidden_activation,
                                   getattr(self, 'hidden_act2', None))
        
        # Output layer
        q_values = self.fc3(x)
        if self.output_activation:
            q_values = self._apply_activation(q_values, self.output_activation,
                                             getattr(self, 'output_act', None))
        
        return q_values
    
    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, filepath):
        """Save model weights."""
        torch.save(self.state_dict(), filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath, device='cpu'):
        """Load model weights."""
        self.load_state_dict(torch.load(filepath, map_location=device))
        print(f"✓ Model loaded from {filepath}")


class ReplayBuffer:
    """
    Experience replay buffer for multi-agent DQN.
    
    Stores experiences from all vehicles.
    """
    
    def __init__(self, capacity=10000, seed=42):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            seed: Random seed
        """
        from collections import deque
        import random
        
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.seed = seed
        random.seed(seed)
        
        print(f"ReplayBuffer initialized:")
        print(f"  Capacity: {capacity:,}")
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        import numpy as np
        
        # Ensure numpy arrays
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample random batch of experiences."""
        import random
        import numpy as np
        
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        # Unzip batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size


def test_dqn_network():
    """Test DQN network."""
    print("\n" + "="*70)
    print("Testing DQN Network")
    print("="*70)
    
    # Network parameters
    state_dim = 84  # Multi-agent state
    action_dim = 180  # 60 stations × 3 fill levels
    hidden_dim = 512
    
    # Create network
    network = DQNNetwork(state_dim, action_dim, hidden_dim)
    
    # Test forward pass with single state
    print(f"\n✓ Testing forward pass (single state):")
    state = torch.randn(state_dim)
    q_values = network(state)
    print(f"  Input shape: {state.shape}")
    print(f"  Output shape: {q_values.shape}")
    print(f"  Sample Q-values: {q_values[:5].detach().numpy()}")
    
    # Test forward pass with batch
    print(f"\n✓ Testing forward pass (batch):")
    batch_size = 64
    state_batch = torch.randn(batch_size, state_dim)
    q_values_batch = network(state_batch)
    print(f"  Input shape: {state_batch.shape}")
    print(f"  Output shape: {q_values_batch.shape}")
    print(f"  Q-value stats: min={q_values_batch.min():.3f}, "
          f"max={q_values_batch.max():.3f}, "
          f"mean={q_values_batch.mean():.3f}")
    
    # Test action selection
    print(f"\n✓ Testing action selection:")
    best_action = torch.argmax(q_values).item()
    print(f"  Best action: {best_action}")
    print(f"  Best Q-value: {q_values[best_action]:.3f}")
    
    # Test gradient flow
    print(f"\n✓ Testing gradient flow:")
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    
    # Dummy loss
    target_q = torch.randn(action_dim)
    loss = F.mse_loss(q_values, target_q)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed successfully")
    
    print("\n✅ DQN network test passed!")
    return network


if __name__ == "__main__":
    test_dqn_network()
