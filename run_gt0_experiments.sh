#!/bin/bash
################################################################################
# GT0 Toy Model Experiment Runner
# 
# Runs all experiments from the paper on the GT0 toy model:
# - 4 output activation functions (none, elu, leaky_relu, prelu)
# - 3 fill level configurations ([10,50,90], [15,50,85], [0,50,100])
#
# GT0 Configuration:
# - 10 stations (vs 60 in GT1/GT2)
# - 2 vehicles with 15 bike capacity
# - ~40 trips per 4-hour horizon
################################################################################

set -e

# Configuration
TIMESTEPS=20000            # 20K timesteps (sufficient per ablation study)
EXPLORATION_FRAC=0.5       # Paper: 0.5 (epsilon decay over first 50% of timesteps)
EVAL_EPISODES=50           # Paper: 50 test episodes
OPTIMIZER="adam"           # Modern standard
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$CODE_DIR/results_GT0"

# GT0 specific parameters
NUM_STATIONS=10
NUM_VEHICLES=2
VEHICLE_CAPACITY=15

echo "=============================================="
echo "GT0 Toy Model Experiment Runner"
echo "=============================================="
echo "Timesteps: $TIMESTEPS"
echo "Exploration fraction: $EXPLORATION_FRAC"
echo "Optimizer: $OPTIMIZER"
echo "GT0 Config: $NUM_STATIONS stations, $NUM_VEHICLES vehicles, capacity $VEHICLE_CAPACITY"
echo "Results directory: $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# Function to run training
run_training() {
    local name=$1
    local extra_args=$2
    local output_dir="$RESULTS_DIR/${name}"
    
    echo ""
    echo "[$(date +'%H:%M:%S')] Training GT0 - $name ($TIMESTEPS timesteps)..."
    python "$CODE_DIR/rl_algorithm/train.py" \
        --gt "GT0" \
        --timesteps $TIMESTEPS \
        --exploration-fraction $EXPLORATION_FRAC \
        --optimizer $OPTIMIZER \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        --output-dir "$output_dir" \
        $extra_args
    
    echo "[$(date +'%H:%M:%S')] Evaluating GT0 - $name on TEST data..."
    python "$CODE_DIR/rl_algorithm/evaluate.py" \
        --gt "GT0" \
        --model "$output_dir/GT0_multi_agent_dqn_final.pth" \
        --episodes $EVAL_EPISODES \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        $extra_args
}

# Function to evaluate static baseline
eval_static_baseline() {
    echo "[$(date +'%H:%M:%S')] Evaluating static baseline for GT0 on TEST data..."
    python "$CODE_DIR/baselines/evaluate_static_baseline.py" \
        --gt "GT0" \
        --episodes $EVAL_EPISODES
}

################################################################################
# MAIN EXPERIMENTS
################################################################################

echo "=============================================="
echo "Phase 1: Static Baseline"
echo "=============================================="
eval_static_baseline

echo ""
echo "=============================================="
echo "Phase 2: Output Activation Function Experiments"
echo "=============================================="
echo "Testing: none, elu, leaky_relu, prelu (paper Section 5.2)"

# Output activation experiments (hidden activation = relu, as per paper)
run_training "activation_none" "--hidden-activation relu"
run_training "activation_elu" "--hidden-activation relu --output-activation elu"
run_training "activation_leaky_relu" "--hidden-activation relu --output-activation leaky_relu"
run_training "activation_prelu" "--hidden-activation relu --output-activation prelu"

echo ""
echo "=============================================="
echo "Phase 3: Fill Level Experiments"
echo "=============================================="
echo "Testing: [15,50,85], [0,50,100] (paper Section 5.2)"

# Fill level experiments (with default activations)
run_training "fill_15_50_85" "--fill-levels 15,50,85"
run_training "fill_0_50_100" "--fill-levels 0,50,100"

echo ""
echo "=============================================="
echo "ALL GT0 EXPERIMENTS COMPLETE!"
echo "=============================================="
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Experiments run:"
echo "  - Static baseline"
echo "  - 4 output activation configs (none, elu, leaky_relu, prelu)"
echo "  - 3 fill level configs ([10,50,90], [15,50,85], [0,50,100])"
echo ""
