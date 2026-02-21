# Bike-Sharing Simulator

Discrete-event simulator for bike-sharing systems with rebalancing operations.

## 📋 Overview

This simulator implements the simulation model from **Base Paper Section 3.2**. It provides a realistic environment for testing and training rebalancing algorithms.

### Key Features

- ✅ **Discrete-event simulation** of bike rentals and returns
- ✅ **Station inventory tracking** with capacity constraints
- ✅ **Lost demand calculation** (failed rentals/returns)
- ✅ **Rebalancing operations** with vehicle fleet
- ✅ **Episode structure** (7am-11am planning horizon)
- ✅ **Comprehensive metrics** tracking and reporting

---

## 📁 Components

### Core Modules

1. **`environment.py`** - Main simulation environment
   - Orchestrates all components
   - Processes trips chronologically
   - Applies rebalancing policies
   - Tracks metrics

2. **`station.py`** - Individual station management
   - Tracks inventory (bikes) and capacity (docks)
   - Handles rentals and returns
   - Records lost demand
   - Supports rebalancing operations

3. **`trip_generator.py`** - Trip data loading and provision
   - Loads synthetic trip data
   - Filters trips for episode windows
   - Provides trips chronologically

4. **`rebalancing_fleet.py`** - Vehicle fleet management
   - Manages multiple vehicles
   - Executes rebalancing operations
   - Accounts for travel time

5. **`metrics.py`** - Performance metrics
   - Tracks lost rentals and returns
   - Calculates rates and percentages
   - Provides summary statistics

---

## 🚀 Quick Start

### Basic Usage

```python
from simulator import BikeShareEnvironment

# Initialize environment
env = BikeShareEnvironment(
    network_file='../data/synthetic/GT1/GT1_station_network.json',
    trips_file='../data/synthetic/GT1/GT1_trips_train.csv',
    num_vehicles=10,
    vehicle_capacity=20
)

# Run single episode (one day)
state, metrics, info = env.step(day=1)

print(f"Lost demand: {metrics['total_lost_demand_rate']:.2f}%")
```

### With Rebalancing Policy

```python
def my_policy(state, env):
    """Your rebalancing policy."""
    # state contains station inventories, time, etc.
    # Return list of (origin_id, dest_id, num_bikes) tuples
    actions = []
    # ... your logic here ...
    return actions

# Run with policy
state, metrics, info = env.step(day=1, rebalancing_policy=my_policy)
```

### Multiple Episodes

```python
# Run 10 days
days = range(1, 11)
results = env.run_multiple_episodes(days, rebalancing_policy=my_policy)

# Average performance
import numpy as np
avg_lost_demand = np.mean([r['total_lost_demand_rate'] for r in results])
print(f"Average lost demand: {avg_lost_demand:.2f}%")
```

---

## 📊 Test Results

Running `test_simulator.py` on GT1 data (no rebalancing):

```
Episodes: 10 days
Average Lost Demand: 31.94%
  - Lost rentals:    13.85%
  - Lost returns:    52.85%
```

**Observations:**
- High lost returns (52.85%) - stations fill up quickly
- Lower lost rentals (13.85%) - bikes available but no space to return
- **This is why rebalancing is needed!**

---

## 🎯 Episode Structure

Based on Base Paper Section 3.2:

```
Episode = One Day, 7am-11am (4 hours)

Timeline:
  06:59 - Episode starts, all stations at 50% capacity
  07:00 - First trips begin
  11:00 - Last trips begin
  ~11:30 - All trips complete, episode ends

Metrics calculated:
  - Lost rentals (customer couldn't get bike)
  - Lost returns (customer couldn't return bike)
  - Total lost demand
```

### Initial Conditions

- **All stations start at 50% capacity**
- **No bikes in transit**
- **All vehicles available**

This matches the base paper's experimental setup.

---

## 📈 State Space

The environment provides state information for decision-making:

```python
state = env.get_state()

# State contains:
{
    'station_inventories': {1: 20, 2: 15, ...},  # Current bikes at each station
    'station_occupancies': {1: 0.5, 2: 0.375, ...},  # Occupancy rates (0-1)
    'current_time': datetime(...),  # Current simulation time
    'current_day': 1,  # Day number
    'num_stations': 60
}
```

### Additional State Information

```python
# Get detailed station info
station_state = env.get_station_info(station_id=1)
# Returns: inventory, capacity, lost_rentals, lost_returns, etc.

# Get all stations
all_states = env.get_all_station_states()
```

---

## 🚚 Rebalancing Actions

Actions are specified as tuples:

```python
action = (origin_station_id, destination_station_id, num_bikes)

# Example: Move 10 bikes from station 5 to station 12
action = (5, 12, 10)

# Multiple actions
actions = [
    (5, 12, 10),
    (3, 15, 5),
    (20, 8, 8)
]
```

### Action Constraints

- ✅ **Origin must have bikes** available
- ✅ **Destination must have docks** available
- ✅ **Vehicle capacity** limits bikes per trip
- ✅ **Fleet size** limits concurrent operations

The simulator automatically handles these constraints.

---

## 📊 Metrics

### Per-Episode Metrics

```python
metrics = {
    'total_rentals_attempted': 398,
    'total_rentals_successful': 388,
    'total_rentals_lost': 10,
    'lost_rental_rate': 2.51,  # percentage
    
    'total_returns_attempted': 388,
    'total_returns_successful': 222,
    'total_returns_lost': 166,
    'lost_return_rate': 42.78,  # percentage
    
    'total_lost_demand': 176,
    'total_lost_demand_rate': 22.39,  # percentage
    
    'bikes_rebalanced': 0,
    'rebalancing_operations': 0,
    'num_stations': 60
}
```

### Aggregated Metrics

When running multiple episodes, the simulator calculates:
- **Average lost demand rates**
- **Total bikes rebalanced**
- **Total operations performed**

---

## 🔬 Validation

### Test Coverage

The test suite (`test_simulator.py`) validates:

1. ✅ Basic functionality (single episode)
2. ✅ Multiple episodes (10 days)
3. ✅ Rebalancing integration
4. ✅ Trip generator
5. ✅ Metrics calculation

### Performance Baseline

**Without rebalancing (10 episodes):**
- Lost demand: ~32%
- Lost rentals: ~14%
- Lost returns: ~53%

**This establishes the baseline for improvement.**

Your RL agent should achieve significantly better performance!

---

## 🎓 Next Steps

### For RL Training

1. **Implement DQN agent** that uses this environment
2. **Define state encoding** (station inventories → neural network input)
3. **Define action space** (which rebalancing operations to take)
4. **Train on 100 days** (GT1 training set)
5. **Evaluate on 50 days** (GT1 test set)

### Expected Performance (Base Paper)

With proper RL training:
- **GT1 DQN:** ~8-10% lost rentals
- **GT1 MIP-30:** ~11-13% lost rentals

**Target: Reduce lost demand from 32% to ~10%!**

---

## 📝 Implementation Notes

### Simplifications

For clarity and initial implementation:

1. **Instant rebalancing** - Bikes delivered immediately (no travel time simulation)
2. **Single rebalancing decision** - At start of episode (can be extended to continuous)
3. **Perfect information** - All trips known (for training; unknown in real deployment)

### Future Enhancements

- [ ] Continuous rebalancing decisions (every 30 min)
- [ ] Travel time simulation for vehicles
- [ ] Multi-vehicle routing optimization
- [ ] Stochastic trip arrival times
- [ ] Weather-dependent demand fluctuations

---

## 📚 References

- **Base Paper Section 3.2:** Simulation Model specification
- **Reference 21:** Instance generation and scenarios
- **Synthetic Data:** Generated using methodology from References 21 & 22

---

**Status:** ✅ Simulator complete and tested  
**Ready for:** RL algorithm implementation  
**Created:** Dec 1, 2025
