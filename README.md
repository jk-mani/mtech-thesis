# Multi-Agent Deep Q-Network for Dynamic Bike-Sharing Rebalancing

This repository implements a Multi-Agent Deep Q-Network (DQN) approach for solving the Dynamic Bicycle Repositioning Problem (DBRP) in bike-sharing systems. It includes both **lost-demand minimization** and **profit-based** reward functions.

## Project Structure

```
code/
├── rl_algorithm/           # Core DQN implementation
│   ├── train.py            # Lost-demand DQN training script
│   ├── train_profit.py     # Profit-based DQN training script
│   ├── evaluate.py         # Lost-demand model evaluation
│   ├── evaluate_profit.py  # Profit-based model evaluation
│   ├── multi_agent_dqn.py  # Multi-agent DQN class
│   ├── dqn_network.py      # Neural network architecture
│   ├── continuous_time_simulator.py  # Discrete-event simulator
│   ├── profit_reward.py    # Profit calculation logic
│   ├── profit_simulator.py # Profit-aware simulator
│   ├── state_encoder.py    # State encoding for DQN
│   ├── action_space.py     # Action space definition
│   ├── events.py           # Event types for simulator
│   ├── vehicle.py          # Vehicle class
│   └── PROFIT_README.md    # Detailed profit-based documentation
├── baselines/              # Baseline methods
│   ├── static_rebalancing.py       # MIP-based static rebalancing
│   └── evaluate_static_baseline.py # Evaluate static baseline
├── simulator/              # Legacy simulator (deprecated)
├── synthetic_data_generation/  # Data generation scripts
│   ├── download_bixi_data.py       # Download BIXI trip data
│   ├── download_weather_data.py    # Download weather data
│   ├── generate_station_network.py # Generate GT1/GT2 networks
│   ├── generate_synthetic_trips.py # Generate synthetic trips
│   └── train_demand_model.py       # Train demand prediction model
├── results_GT0/            # GT0 lost-demand experiment results
├── results_profit_GT0/     # GT0 profit-based experiment results
├── run_gt0_experiments.sh  # GT0 lost-demand experiments
├── run_profit_gt0_experiments.sh  # GT0 profit experiments
├── evaluate_lostdemand_activations.py  # Test evaluation script
├── evaluate_profit_activations.py      # Profit test evaluation
├── plot_activation_comparison.py       # Plot generation
└── requirements.txt        # Python dependencies
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Synthetic Data (if not already present)

```bash
cd synthetic_data_generation
python generate_station_network.py
python generate_synthetic_trips.py
```

### 2. Compute Static Baseline

```bash
cd baselines
python static_rebalancing.py --gt GT1
python static_rebalancing.py --gt GT2
```

### 3. Train Lost-Demand DQN

```bash
# Basic training with timesteps
python rl_algorithm/train.py --gt GT0 --timesteps 20000 --output-dir results_GT0/experiment

# With specific output activation function
python rl_algorithm/train.py --gt GT0 --timesteps 20000 --output-activation elu --output-dir results_GT0/activation_elu

# With specific fill levels
python rl_algorithm/train.py --gt GT0 --timesteps 20000 --fill-levels 10 50 90 --output-dir results_GT0/fill_10_50_90
```

### 4. Train Profit-Based DQN

```bash
# Train with distance-based revenue
python rl_algorithm/train_profit.py --gt GT0 --timesteps 50000 \
    --trip-base-fare 1.00 --trip-per-km 0.75 \
    --cost-per-km 1.00 --lost-penalty 5.00 \
    --output-dir results_profit_GT0/experiment

# With specific activation function
python rl_algorithm/train_profit.py --gt GT0 --timesteps 50000 \
    --output-activation elu --output-dir results_profit_GT0/activation_elu
```

### 5. Evaluate Trained Models

```bash
# Evaluate lost-demand DQN on test set (50 episodes)
python rl_algorithm/evaluate.py --gt GT0 --model results_GT0/experiment/GT0_multi_agent_dqn_final.pth --episodes 50

# Evaluate profit DQN on test set
python rl_algorithm/evaluate_profit.py --gt GT0 --model results_profit_GT0/experiment/GT0_profit_dqn_best.pth --episodes 50

# Batch evaluation for activation study
python evaluate_lostdemand_activations.py   # Lost-demand models
python evaluate_profit_activations.py       # Profit models
```

## Running GT0 Experiments

```bash
# Run lost-demand activation experiments (20k steps each)
./run_gt0_experiments.sh

# Run profit-based activation experiments (50k steps each)
./run_profit_gt0_experiments.sh
```

## Key Parameters

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timesteps` | 100000 | Number of training timesteps |
| `--optimizer` | adam | Optimizer (sgd, adam) |
| `--lr` | 2.5e-4 | Learning rate |
| `--batch-size` | 256 | Batch size for training |
| `--gamma` | 0.99 | Discount factor |
| `--epsilon-start` | 1.0 | Initial exploration rate |
| `--epsilon-end` | 0.05 | Final exploration rate |
| `--hidden-activation` | relu | Hidden layer activation |
| `--output-activation` | none | Output activation (none, elu, leaky_relu, prelu) |
| `--fill-levels` | 10 50 90 | Target fill levels for action space |

### Profit Parameters (train_profit.py)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--trip-base-fare` | 1.00 | Base fare per trip ($) |
| `--trip-per-km` | 0.75 | Per-km trip rate ($) |
| `--cost-per-km` | 1.00 | Truck travel cost ($/km) |
| `--lost-penalty` | 5.00 | Lost demand penalty ($) |

## Datasets

- **GT0**: 10 stations, 2 vehicles (toy model for quick experiments)
- **GT1**: 60 stations, 4 vehicles, ~500 trips/day
- **GT2**: 60 stations, 4 vehicles, ~750 trips/day

All use:
- Training set: 100 days
- Test set: 50 days
- Planning horizon: 4 hours (7am-11am)

## Metrics

- **Lost Demand Rate**: Percentage of trips that couldn't be served
- **Profit**: Revenue - Truck Cost - Lost Demand Penalty

## GT0 Experiment Results

### Lost-Demand DQN (20k timesteps, test evaluation)
| Activation | Avg Lost Demand |
|------------|-----------------|
| None | 7.77% |
| **ELU** | **6.60%** |
| Leaky ReLU | 7.08% |
| PReLU | 7.72% |

### Profit-Based DQN (50k timesteps, test evaluation)
| Activation | Avg Profit | Avg Lost Demand |
|------------|-----------|-----------------|
| None | $102.94 | 6.98% |
| ELU | $103.00 | 7.88% |
| Leaky ReLU | $99.23 | 6.98% |
| **PReLU** | **$109.12** | **5.96%** |

**Key Findings:**
- **ELU** is best for lost-demand minimization
- **PReLU** is best for profit maximization

## References

Based on the methodology from:
> "Multi-Agent Deep Reinforcement Learning for Dynamic Bike Repositioning in Bike-Sharing Systems"

See `rl_algorithm/PROFIT_README.md` for detailed profit-based documentation.

## License

For academic use only.
