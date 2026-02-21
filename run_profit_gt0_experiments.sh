#!/bin/bash
################################################################################
# GT0 Profit-Based DQN Experiments
# 
# Tests different output activation functions with profit-based reward:
# - Profit = Trip Revenue - Truck Cost - Lost Demand Penalty
# - Distance-based revenue: $1.00 base + $0.75/km
# - Truck cost: $1.00/km
# - Lost demand penalty: $5.00/trip
################################################################################

set -e

# Configuration
TIMESTEPS=50000            # 50K timesteps
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$CODE_DIR/results_profit_GT0"

# GT0 specific parameters
NUM_STATIONS=10
NUM_VEHICLES=2
VEHICLE_CAPACITY=15

# Economic parameters
TRIP_BASE_FARE=1.00
TRIP_PER_KM=0.75
COST_PER_KM=1.00
LOST_PENALTY=5.00

echo "=============================================="
echo "GT0 Profit-Based DQN Experiments"
echo "=============================================="
echo "Timesteps: $TIMESTEPS"
echo "Trip Revenue: \$$TRIP_BASE_FARE + \$$TRIP_PER_KM/km"
echo "Truck Cost: \$$COST_PER_KM/km"
echo "Lost Penalty: \$$LOST_PENALTY"
echo "Results directory: $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# Function to run training
run_profit_training() {
    local name=$1
    local extra_args=$2
    local output_dir="$RESULTS_DIR/${name}"
    
    echo ""
    echo "[$(date +'%H:%M:%S')] Training Profit-DQN - $name ($TIMESTEPS timesteps)..."
    python "$CODE_DIR/rl_algorithm/train_profit_simple.py" \
        --gt GT0 \
        --timesteps $TIMESTEPS \
        --output-dir "$output_dir" \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        --trip-base-fare $TRIP_BASE_FARE \
        --trip-per-km $TRIP_PER_KM \
        --cost-per-km $COST_PER_KM \
        --lost-penalty $LOST_PENALTY \
        $extra_args
    
    echo "[$(date +'%H:%M:%S')] ✓ Completed: $name"
}

echo ""
echo "=============================================="
echo "Output Activation Function Experiments"
echo "=============================================="
echo "Testing: none, elu, leaky_relu, prelu"

# Output activation experiments (hidden activation = relu)
run_profit_training "activation_none" "--hidden-activation relu"
run_profit_training "activation_elu" "--hidden-activation relu --output-activation elu"
run_profit_training "activation_leaky_relu" "--hidden-activation relu --output-activation leaky_relu"
run_profit_training "activation_prelu" "--hidden-activation relu --output-activation prelu"

echo ""
echo "=============================================="
echo "ALL PROFIT-DQN EXPERIMENTS COMPLETE!"
echo "=============================================="
echo "Results saved in: $RESULTS_DIR"
echo ""
