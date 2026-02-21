#!/bin/bash
################################################################################
# Multi-Agent DQN Experiment Runner
# 
# Runs all experiments from the paper:
# - 4 activation functions (none, elu, leaky_relu, prelu)
# - 3 fill level configurations ([10,50,90], [15,50,85], [0,50,100])
# - 2 networks (GT1, GT2)
################################################################################

set -e

# Configuration (matching base paper)
TIMESTEPS=300000           # Paper: 3,000,000 (we use 300K for faster iteration)
EXPLORATION_FRAC=0.5       # Paper: 0.5 (epsilon decay over first 50% of timesteps)
EVAL_EPISODES=50           # Paper: 50 test episodes
OPTIMIZER="adam"           # Modern standard (paper used RMSprop)
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$CODE_DIR/results"

echo "=============================================="
echo "Multi-Agent DQN Experiment Runner"
echo "=============================================="
echo "Timesteps: $TIMESTEPS"
echo "Exploration fraction: $EXPLORATION_FRAC"
echo "Optimizer: $OPTIMIZER"
echo "Results directory: $RESULTS_DIR"
echo ""

# Function to run training (timestep-based)
run_training() {
    local gt=$1
    local name=$2
    local extra_args=$3
    local output_dir="$RESULTS_DIR/${gt}_${name}"
    
    echo "[$(date +'%H:%M:%S')] Training $gt - $name ($TIMESTEPS timesteps)..."
    python "$CODE_DIR/rl_algorithm/train.py" \
        --gt "$gt" \
        --timesteps $TIMESTEPS \
        --exploration-fraction $EXPLORATION_FRAC \
        --optimizer $OPTIMIZER \
        --output-dir "$output_dir" \
        $extra_args
    
    echo "[$(date +'%H:%M:%S')] Evaluating $gt - $name on TEST data..."
    python "$CODE_DIR/rl_algorithm/evaluate.py" \
        --gt "$gt" \
        --model "$output_dir/${gt}_multi_agent_dqn_final.pth" \
        --episodes $EVAL_EPISODES
}

# Function to evaluate static baseline on TEST data
eval_static_baseline() {
    local gt=$1
    echo "[$(date +'%H:%M:%S')] Evaluating static baseline for $gt on TEST data..."
    python "$CODE_DIR/baselines/evaluate_static_baseline.py" \
        --gt "$gt" \
        --episodes $EVAL_EPISODES
    # Note: Uses test data by default (use_test_data=True)
}

################################################################################
# MAIN EXPERIMENTS
################################################################################

echo "=============================================="
echo "Phase 1: Static Baselines"
echo "=============================================="
eval_static_baseline "GT1"
eval_static_baseline "GT2"

echo ""
echo "=============================================="
echo "Phase 2: Activation Function Experiments"
echo "=============================================="

# GT1 Activation Experiments
run_training "GT1" "activation_none" ""
run_training "GT1" "activation_elu" "--activation elu"
run_training "GT1" "activation_leaky_relu" "--activation leaky_relu"
run_training "GT1" "activation_prelu" "--activation prelu"

# GT2 Activation Experiments
run_training "GT2" "activation_none" ""
run_training "GT2" "activation_elu" "--activation elu"
run_training "GT2" "activation_leaky_relu" "--activation leaky_relu"
run_training "GT2" "activation_prelu" "--activation prelu"

echo ""
echo "=============================================="
echo "Phase 3: Fill Level Experiments"
echo "=============================================="

# GT1 Fill Level Experiments
run_training "GT1" "fill_10_50_90" "--fill-levels 10,50,90"
run_training "GT1" "fill_15_50_85" "--fill-levels 15,50,85"
run_training "GT1" "fill_0_50_100" "--fill-levels 0,50,100"

# GT2 Fill Level Experiments
run_training "GT2" "fill_10_50_90" "--fill-levels 10,50,90"
run_training "GT2" "fill_15_50_85" "--fill-levels 15,50,85"
run_training "GT2" "fill_0_50_100" "--fill-levels 0,50,100"

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=============================================="
echo "Results saved in: $RESULTS_DIR"
