"""
Fill-Level Action Space for Multi-Agent Rebalancing.

Based on Base Paper Section 4.1.

Action = (routing decision, loading decision)
- Routing: Which station to visit next (1-60)
- Loading: Which fill level to target (10%, 50%, 90%)

Total actions per vehicle: 60 × 3 = 180
"""

import numpy as np


class FillLevelActionSpace:
    """
    Action space based on target fill levels.
    
    From paper: "we only consider three predefined fill levels for the
    inventories, indicating the proportion of the station capacity: µ1, µ2, and µ3"
    
    Best performing configuration: µ1=10%, µ2=50%, µ3=90%
    """
    
    def __init__(self, num_stations=60, fill_levels=None):
        """
        Initialize action space.
        
        Args:
            num_stations: Number of stations in network
            fill_levels: List of fill levels (default: [0.10, 0.50, 0.90])
        """
        self.num_stations = num_stations
        self.fill_levels = fill_levels if fill_levels is not None else [0.10, 0.50, 0.90]
        self.num_fill_levels = len(self.fill_levels)
        
        # Generate all actions
        self.actions = self._generate_actions()
        self.num_actions = len(self.actions)
        
        print(f"FillLevelActionSpace initialized:")
        print(f"  Stations: {num_stations}")
        print(f"  Fill levels: {[f'{f*100:.0f}%' for f in self.fill_levels]}")
        print(f"  Total actions: {self.num_actions}")
    
    def _generate_actions(self):
        """
        Generate all possible actions.
        
        Returns:
            list: List of (station_id, fill_level_idx) tuples
        """
        actions = []
        for station_id in range(1, self.num_stations + 1):
            for fill_idx in range(self.num_fill_levels):
                actions.append((station_id, fill_idx))
        return actions
    
    def get_action(self, action_idx):
        """
        Get action tuple from action index.
        
        Args:
            action_idx: Action index (0 to num_actions-1)
        
        Returns:
            tuple: (station_id, fill_level_idx)
        """
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"Invalid action index: {action_idx}")
        return self.actions[action_idx]
    
    def get_fill_level(self, fill_level_idx):
        """Get fill level value from index."""
        return self.fill_levels[fill_level_idx]
    
    def get_num_actions(self):
        """Return total number of actions."""
        return self.num_actions
    
    def describe_action(self, action_idx):
        """
        Human-readable action description.
        
        Args:
            action_idx: Action index
        
        Returns:
            str: Action description
        """
        station_id, fill_idx = self.get_action(action_idx)
        fill_pct = self.fill_levels[fill_idx] * 100
        return f"Go to Station {station_id}, rebalance to {fill_pct:.0f}% capacity"
    
    def get_valid_actions(self, state_dict, vehicle_id):
        """
        Get list of valid actions for a specific vehicle.
        
        Constraint from paper: "no two vehicles can be directed towards the
        same station simultaneously"
        
        Args:
            state_dict: Current state dictionary
            vehicle_id: Which vehicle is making the decision
        
        Returns:
            list: List of valid action indices
        """
        valid_actions = []
        
        # Get current vehicle state
        vehicle_state = state_dict['vehicle_states'].get(vehicle_id, {})
        current_station = vehicle_state.get('current_station')
        
        # Get stations that other vehicles are heading to
        reserved_stations = set()
        for vid, v_state in state_dict['vehicle_states'].items():
            if vid != vehicle_id:
                dest = v_state.get('destination')
                if dest is not None:
                    reserved_stations.add(dest)
        
        # Check each action
        for action_idx in range(self.num_actions):
            station_id, fill_idx = self.get_action(action_idx)
            
            # Can't go to station reserved by another vehicle
            if station_id in reserved_stations:
                continue
            
            # Can go to own current station (rebalance and stay)
            # Can go to any unreserved station
            valid_actions.append(action_idx)
        
        return valid_actions
    
    def mask_invalid_actions(self, q_values, valid_actions):
        """
        Mask invalid actions by setting their Q-values to -inf.
        
        Args:
            q_values: Q-values for all actions (np.array or tensor)
            valid_actions: List of valid action indices
        
        Returns:
            Masked Q-values
        """
        import torch
        
        if isinstance(q_values, torch.Tensor):
            masked_q = q_values.clone()
            mask = torch.ones_like(q_values, dtype=torch.bool)
            mask[valid_actions] = False
            masked_q[mask] = float('-inf')
        else:
            masked_q = q_values.copy()
            for action_idx in range(self.num_actions):
                if action_idx not in valid_actions:
                    masked_q[action_idx] = float('-inf')
        
        return masked_q
    
    def sample_action(self, valid_actions=None):
        """Sample random action (for exploration)."""
        if valid_actions is not None and len(valid_actions) > 0:
            return np.random.choice(valid_actions)
        return np.random.randint(0, self.num_actions)


class HeuristicRoutingActionSpace:
    """
    Simplified action space with heuristic routing (Section 5.2.2 of paper).
    
    DQN only decides fill level (3 actions).
    Routing is determined by heuristic:
    - Empty vehicle → go to fullest station
    - Full vehicle → go to emptiest station
    """
    
    def __init__(self, num_stations=60, fill_levels=None):
        """
        Initialize simplified action space.
        
        Args:
            num_stations: Number of stations in network
            fill_levels: List of fill levels (default: [0.10, 0.50, 0.90])
        """
        self.num_stations = num_stations
        self.fill_levels = fill_levels if fill_levels is not None else [0.10, 0.50, 0.90]
        self.num_fill_levels = len(self.fill_levels)
        self.num_actions = self.num_fill_levels  # Only 3 actions!
        
        print(f"HeuristicRoutingActionSpace initialized:")
        print(f"  Stations: {num_stations}")
        print(f"  Fill levels: {[f'{f*100:.0f}%' for f in self.fill_levels]}")
        print(f"  Total actions: {self.num_actions} (fill level only)")
        print(f"  Routing: Heuristic (fullest/emptiest station)")
    
    def get_action(self, action_idx):
        """
        Get fill level from action index.
        
        Args:
            action_idx: Action index (0, 1, or 2)
        
        Returns:
            int: fill_level_idx (routing determined separately by heuristic)
        """
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"Invalid action index: {action_idx}")
        return action_idx  # Just the fill level index
    
    def get_fill_level(self, fill_level_idx):
        """Get fill level value from index."""
        return self.fill_levels[fill_level_idx]
    
    def get_num_actions(self):
        """Return total number of actions (3)."""
        return self.num_actions
    
    def describe_action(self, action_idx):
        """Human-readable action description."""
        fill_pct = self.fill_levels[action_idx] * 100
        return f"Rebalance to {fill_pct:.0f}% capacity (heuristic routing)"
    
    def get_valid_actions(self, state_dict, vehicle_id):
        """All fill level actions are always valid."""
        return list(range(self.num_actions))
    
    def mask_invalid_actions(self, q_values, valid_actions):
        """No masking needed - all actions valid."""
        return q_values
    
    def select_station_heuristic(self, state_dict, vehicle_id):
        """
        Select next station using heuristic rule from paper Section 5.2.2:
        - Assign fullest station to most empty vehicle
        - Assign emptiest station to most full vehicle
        
        Args:
            state_dict: Current state dictionary
            vehicle_id: Which vehicle is making the decision
        
        Returns:
            int: Station ID to visit
        """
        vehicle_state = state_dict['vehicle_states'].get(vehicle_id, {})
        vehicle_occupancy = vehicle_state.get('occupancy', 0.5)
        current_station = vehicle_state.get('current_station')
        
        # Get stations that other vehicles are heading to (to avoid conflicts)
        reserved_stations = set()
        for vid, v_state in state_dict['vehicle_states'].items():
            if vid != vehicle_id:
                dest = v_state.get('destination')
                if dest is not None:
                    reserved_stations.add(dest)
        
        # Get station occupancies
        station_occupancies = state_dict.get('station_occupancies', {})
        
        # Find available stations (not reserved by other vehicles)
        available_stations = []
        for sid, occ in station_occupancies.items():
            if sid not in reserved_stations:
                available_stations.append((sid, occ))
        
        if not available_stations:
            # Fallback: stay at current station
            return current_station if current_station else 1
        
        # Heuristic: 
        # - If vehicle is more empty (occupancy < 0.5), go to FULLEST station (to pick up)
        # - If vehicle is more full (occupancy >= 0.5), go to EMPTIEST station (to drop off)
        if vehicle_occupancy < 0.5:
            # Go to fullest station
            best_station = max(available_stations, key=lambda x: x[1])
        else:
            # Go to emptiest station
            best_station = min(available_stations, key=lambda x: x[1])
        
        return best_station[0]
    
    def sample_action(self, valid_actions=None):
        """Sample random fill level action."""
        return np.random.randint(0, self.num_actions)


def test_action_space():
    """Test fill-level action space."""
    print("\n" + "="*70)
    print("Testing Fill-Level Action Space")
    print("="*70)
    
    # Create action space
    action_space = FillLevelActionSpace(num_stations=60)
    
    print(f"\n✓ Action space created:")
    print(f"  Total actions: {action_space.get_num_actions()}")
    
    # Sample some actions
    print(f"\n✓ Sample actions:")
    for i in [0, 1, 2, 60, 120, 179]:
        if i < action_space.get_num_actions():
            desc = action_space.describe_action(i)
            action = action_space.get_action(i)
            print(f"  Action {i}: {action} - {desc}")
    
    # Test valid actions
    print(f"\n✓ Testing valid action filtering:")
    
    # Mock state with some vehicles traveling
    dummy_state = {
        'vehicle_states': {
            1: {'current_station': 5, 'destination': None},  # At station, not traveling
            2: {'current_station': 10, 'destination': 25},   # Traveling to 25
            3: {'current_station': 15, 'destination': 30},   # Traveling to 30
        }
    }
    
    # Vehicle 1 can't go to stations 25 or 30
    valid_actions_v1 = action_space.get_valid_actions(dummy_state, vehicle_id=1)
    print(f"  Vehicle 1 valid actions: {len(valid_actions_v1)} out of {action_space.get_num_actions()}")
    
    # Check that station 25 and 30 are blocked
    blocked = []
    for action_idx in range(action_space.get_num_actions()):
        station_id, _ = action_space.get_action(action_idx)
        if station_id in [25, 30]:
            if action_idx not in valid_actions_v1:
                blocked.append(station_id)
    
    print(f"  Correctly blocked stations: {set(blocked)}")
    assert 25 in blocked and 30 in blocked, "Stations should be blocked!"
    
    # Test masking
    print(f"\n✓ Testing Q-value masking:")
    q_values = np.random.randn(action_space.get_num_actions())
    masked_q = action_space.mask_invalid_actions(q_values, valid_actions_v1)
    
    num_masked = np.sum(np.isinf(masked_q))
    print(f"  Original Q-values: min={q_values.min():.2f}, max={q_values.max():.2f}")
    print(f"  Masked invalid actions: {num_masked}")
    print(f"  Valid actions remain: {len(valid_actions_v1)}")
    
    print("\n✅ Action space test passed!")
    return action_space


if __name__ == "__main__":
    test_action_space()
