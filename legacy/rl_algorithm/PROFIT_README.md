# Profit-Based Bike-Sharing Rebalancing

## Overview

This extension goes beyond the base paper's lost-demand minimization by introducing a **profit-based reward function** that balances:

- **Revenue** from successful trips
- **Operational costs** (vehicle travel, driver time, bike handling)
- **Opportunity cost** of lost demand

This enables the RL agent to learn **economically optimal** rebalancing policies.

## Profit Model

### Revenue (Distance-Based)
```
Trip_Revenue = base_fare + (trip_distance_km × per_km_rate)
Total_Revenue = Σ Trip_Revenue for all successful trips
```
- Default: **$1.00 base + $0.75/km per trip**

### Operational Costs
```
Operational_Cost = Truck_Travel_Cost
```

| Cost Component | Formula | Default Value |
|----------------|---------|---------------|
| Truck Travel Cost | distance_km × cost_per_km | $1.00/km |

### Opportunity Cost
```
Lost_Demand_Cost = (lost_rentals + lost_returns) × penalty
```
- Default: **$5.00 per lost rental/return**

### Net Profit
```
Profit = Revenue - Operational_Cost - Lost_Demand_Cost
```

## Files Created

### `rl_algorithm/profit_reward.py`
Core profit calculation logic:
- `ProfitParameters`: Configurable economic parameters
- `ProfitRewardCalculator`: Step-by-step profit calculation
- Predefined parameter sets (conservative, aggressive, high_penalty)

### `rl_algorithm/profit_simulator.py`
Event-driven simulator with profit rewards:
- Extends base simulator with profit tracking
- Tracks distance traveled, time spent, bikes handled
- Provides detailed economic breakdown

### `rl_algorithm/train_profit.py`
Training script:
```bash
# Basic usage
python train_profit.py --gt GT0 --timesteps 50000

# Custom profit parameters (distance-based revenue)
python train_profit.py --gt GT0 --timesteps 50000 \
    --trip-base-fare 1.00 \
    --trip-per-km 0.75 \
    --cost-per-km 1.00 \
    --lost-penalty 5.00

# With specific activation function
python train_profit.py --gt GT0 --timesteps 50000 \
    --output-activation prelu \
    --output-dir ../results_profit_GT0/activation_prelu
```

### `rl_algorithm/evaluate_profit.py`
Evaluation script:
```bash
python evaluate_profit.py --gt GT0 \
    --model results_profit/GT0/GT0_profit_dqn_final.pth \
    --episodes 50
```

### `baselines/evaluate_profit_baseline.py`
Static baseline with profit metrics:
```bash
# Evaluate baseline
python evaluate_profit_baseline.py --gt GT0 --episodes 50

# Compare with DQN
python evaluate_profit_baseline.py --gt GT0 \
    --compare-dqn results_profit/GT0/evaluation/GT0_profit_evaluation_results.json
```

## Usage Example

### 1. Train Profit-Based Agent
```bash
cd code/rl_algorithm

# Train on GT0 with default parameters
python train_profit.py --gt GT0 --timesteps 100000 \
    --output-dir ../results_profit/GT0_default

# Train with high lost-demand penalty
python train_profit.py --gt GT0 --timesteps 100000 \
    --lost-demand-penalty 10.00 \
    --output-dir ../results_profit/GT0_high_penalty
```

### 2. Evaluate Static Baseline
```bash
cd code/baselines

python evaluate_profit_baseline.py --gt GT0 --episodes 50
```

### 3. Evaluate Trained Agent
```bash
cd code/rl_algorithm

python evaluate_profit.py --gt GT0 \
    --model ../results_profit/GT0_default/GT0_profit_dqn_final.pth \
    --episodes 50
```

### 4. Compare Results
```bash
cd code/baselines

python evaluate_profit_baseline.py --gt GT0 \
    --compare-dqn ../results_profit/GT0_default/evaluation/GT0_profit_evaluation_results.json
```

## Economic Insights

### Trade-offs to Consider

1. **Service Quality vs Cost**: Lower lost demand requires more vehicle movement
2. **Proactive vs Reactive**: Moving bikes preemptively costs more but prevents losses
3. **Route Efficiency**: Shorter routes reduce costs but may miss demand hotspots

### Expected Behavior

With default parameters:
- Agent should **reduce unnecessary movements** (high travel cost)
- Agent should **prioritize high-demand stations** (lost demand penalty > trip revenue)
- Agent should **batch rebalancing operations** (stop cost overhead)

### Parameter Sensitivity

| Parameter | High Value Effect | Low Value Effect |
|-----------|-------------------|------------------|
| `lost_demand_penalty` | More aggressive rebalancing | Fewer vehicle movements |
| `cost_per_km` | Shorter routes, local optimization | More distant stations visited |
| `trip_per_km` | Focus on longer trips | Even distribution |

## Comparison with Base Paper

| Aspect | Base Paper | Profit Extension |
|--------|------------|------------------|
| Reward | -Lost Demand | Revenue - Costs |
| Goal | Minimize service failures | Maximize profit |
| Costs | Not considered | Truck travel cost |
| Trade-offs | None | Service vs efficiency |
| Metrics | Lost demand rate | Net profit |

## GT0 Test Results

### Activation Function Comparison (50 test episodes)
| Activation | Avg Profit | Avg Lost Demand |
|------------|-----------|-----------------|
| None | $102.94 | 6.98% |
| ELU | $103.00 | 7.88% |
| Leaky ReLU | $99.23 | 6.98% |
| **PReLU** | **$109.12** | **5.96%** |

**Best Activation: PReLU** (highest profit, lowest lost demand)

## Future Extensions

1. **Dynamic Pricing**: Adjust trip revenue based on demand
2. **Driver Scheduling**: Include driver shift costs
3. **Multi-Objective**: Pareto frontier of profit vs service
4. **Demand Forecasting**: Proactive profit optimization
