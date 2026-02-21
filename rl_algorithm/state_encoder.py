"""
State Encoder for Multi-Agent DQN.

Encodes global system state into fixed-size vector for neural network.

State includes:
- Station inventories (all stations)
- Vehicle states (all vehicles)
- Time features
- System-level features
"""

import numpy as np


class MultiAgentStateEncoder:
    """Encodes global state for multi-agent system."""
    
    def __init__(self, num_stations=60, num_vehicles=4):
        """
        Initialize state encoder.
        
        Args:
            num_stations: Number of stations
            num_vehicles: Number of vehicles
        """
        self.num_stations = num_stations
        self.num_vehicles = num_vehicles
        self.state_dim = self._calculate_state_dim()
        
        print(f"MultiAgentStateEncoder initialized:")
        print(f"  Stations: {num_stations}")
        print(f"  Vehicles: {num_vehicles}")
        print(f"  State dimension: {self.state_dim}")
    
    def _calculate_state_dim(self):
        """Calculate total state dimension."""
        dim = 0
        dim += self.num_stations      # Station occupancy rates
        dim += self.num_vehicles      # Vehicle inventories (normalized)
        dim += self.num_vehicles      # Vehicle locations (one-hot would be large, use station ID normalized)
        dim += 2                       # Time features (hour, weekday)
        dim += 2                       # System features (avg station occupancy, avg vehicle occupancy)
        return dim
    
    def encode(self, state_dict):
        """
        Encode simulator state into feature vector.
        
        Args:
            state_dict: State dictionary from simulator
        
        Returns:
            np.array: State vector of shape (state_dim,)
        """
        features = []
        
        # 1. Station occupancy rates (60 features)
        station_occupancies = []
        for sid in range(1, self.num_stations + 1):
            occ = state_dict['station_occupancies'].get(sid, 0.5)
            station_occupancies.append(occ)
        features.extend(station_occupancies)
        
        # 2. Vehicle inventories normalized by capacity (10 features)
        vehicle_inventories = []
        for vid in range(1, self.num_vehicles + 1):
            v_state = state_dict['vehicle_states'].get(vid, {})
            inv_norm = v_state.get('occupancy', 0.0)
            vehicle_inventories.append(inv_norm)
        features.extend(vehicle_inventories)
        
        # 3. Vehicle locations normalized (10 features)
        # Encode as station_id / num_stations
        vehicle_locations = []
        for vid in range(1, self.num_vehicles + 1):
            v_state = state_dict['vehicle_states'].get(vid, {})
            location = v_state.get('current_station', 1)
            if location is None:
                location = 1
            loc_norm = location / self.num_stations
            vehicle_locations.append(loc_norm)
        features.extend(vehicle_locations)
        
        # 4. Time features (2 features)
        if state_dict.get('current_time') is not None:
            current_time = state_dict['current_time']
            hour_norm = current_time.hour / 23.0
            weekday_norm = current_time.weekday() / 6.0
            features.extend([hour_norm, weekday_norm])
        else:
            features.extend([0.0, 0.0])
        
        # 5. System-level features (2 features)
        # Average station occupancy
        avg_station_occ = np.mean(station_occupancies)
        features.append(avg_station_occ)
        
        # Average vehicle occupancy
        avg_vehicle_occ = np.mean(vehicle_inventories)
        features.append(avg_vehicle_occ)
        
        # Convert to numpy array
        state_vector = np.array(features, dtype=np.float32)
        
        # Sanity check
        assert len(state_vector) == self.state_dim, \
            f"State vector size mismatch: {len(state_vector)} != {self.state_dim}"
        
        return state_vector
    
    def get_state_dim(self):
        """Return state dimension."""
        return self.state_dim
    
    def describe_state(self, state_vector):
        """
        Human-readable description of state vector.
        
        Args:
            state_vector: Encoded state vector
        
        Returns:
            dict: Interpretable state description
        """
        idx = 0
        
        # Station occupancies
        station_occs = state_vector[idx:idx+self.num_stations]
        idx += self.num_stations
        
        # Vehicle inventories
        vehicle_invs = state_vector[idx:idx+self.num_vehicles]
        idx += self.num_vehicles
        
        # Vehicle locations
        vehicle_locs = state_vector[idx:idx+self.num_vehicles]
        idx += self.num_vehicles
        
        # Time
        hour_norm = state_vector[idx]
        weekday_norm = state_vector[idx+1]
        idx += 2
        
        # System
        avg_station_occ = state_vector[idx]
        avg_vehicle_occ = state_vector[idx+1]
        
        return {
            'stations': {
                'mean_occupancy': float(np.mean(station_occs)),
                'std_occupancy': float(np.std(station_occs)),
                'min_occupancy': float(np.min(station_occs)),
                'max_occupancy': float(np.max(station_occs)),
            },
            'vehicles': {
                'mean_inventory': float(np.mean(vehicle_invs)),
                'locations': [int(loc * self.num_stations) for loc in vehicle_locs]
            },
            'time': {
                'hour': int(hour_norm * 23),
                'weekday': int(weekday_norm * 6)
            },
            'system': {
                'avg_station_occ': float(avg_station_occ),
                'avg_vehicle_occ': float(avg_vehicle_occ)
            }
        }


def test_state_encoder():
    """Test state encoder with dummy data."""
    print("\n" + "="*70)
    print("Testing Multi-Agent State Encoder")
    print("="*70)
    
    encoder = MultiAgentStateEncoder(num_stations=60, num_vehicles=10)
    
    # Create dummy state
    from datetime import datetime
    
    dummy_state = {
        'station_inventories': {i: 10 for i in range(1, 61)},
        'station_occupancies': {i: 0.5 for i in range(1, 61)},
        'vehicle_states': {
            i: {
                'current_station': i * 6,
                'inventory': i * 2,
                'capacity': 20,
                'occupancy': (i * 2) / 20
            }
            for i in range(1, 11)
        },
        'current_time': datetime(2019, 5, 1, 8, 30),
        'upcoming_trips': []
    }
    
    # Encode
    state_vector = encoder.encode(dummy_state)
    
    print(f"\n✓ Encoded state vector:")
    print(f"  Shape: {state_vector.shape}")
    print(f"  Dtype: {state_vector.dtype}")
    print(f"  Range: [{state_vector.min():.3f}, {state_vector.max():.3f}]")
    print(f"  Mean: {state_vector.mean():.3f}")
    print(f"  Sample values: {state_vector[:10]}")
    
    # Describe
    description = encoder.describe_state(state_vector)
    print(f"\n✓ State description:")
    print(f"  Stations - mean occupancy: {description['stations']['mean_occupancy']:.2f}")
    print(f"  Vehicles - mean inventory: {description['vehicles']['mean_inventory']:.2f}")
    print(f"  Vehicles - locations: {description['vehicles']['locations']}")
    print(f"  Time: {description['time']['hour']}:00, weekday {description['time']['weekday']}")
    
    print("\n✅ State encoder test passed!")
    return encoder


if __name__ == "__main__":
    test_state_encoder()
