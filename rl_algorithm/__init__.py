"""
Multi-Agent RL Algorithm Package

Correct implementation matching Base Paper:
- Multi-agent (10 vehicles)
- Continuous time framework
- Fill-level action space (10%, 50%, 90%)
- Event-driven simulation
- Cooperative learning

Components:
- Events: Event definitions for continuous-time simulation
- Vehicle: Vehicle/agent state management
- ContinuousTimeSimulator: Event-driven simulator
- MultiAgentStateEncoder: Global state encoding
- FillLevelActionSpace: Fill-level based actions
- DQNNetwork: Shared neural network
- MultiAgentDQN: Complete multi-agent DQN agent
"""

from .events import Event, VehicleArrival, CustomerRental, CustomerReturn
from .vehicle import Vehicle
from .continuous_time_simulator import ContinuousTimeSimulator
from .state_encoder import MultiAgentStateEncoder
from .action_space import FillLevelActionSpace
from .dqn_network import DQNNetwork, ReplayBuffer
from .multi_agent_dqn import MultiAgentDQN

__all__ = [
    'Event',
    'VehicleArrival',
    'CustomerRental',
    'CustomerReturn',
    'Vehicle',
    'ContinuousTimeSimulator',
    'MultiAgentStateEncoder',
    'FillLevelActionSpace',
    'DQNNetwork',
    'ReplayBuffer',
    'MultiAgentDQN',
]
