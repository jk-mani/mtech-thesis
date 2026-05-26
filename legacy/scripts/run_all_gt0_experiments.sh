#!/bin/bash
################################################################################
# GT0 COMPLETE EXPERIMENT SUITE
# 
# This script runs ALL GT0 experiments for the thesis:
#
# PHASE 1: DATA GENERATION
#   - Build synthetic station network (10 stations)
#   - Generate synthetic trips (train + test)
#
# PHASE 2: STATIC BASELINE
#   - Evaluate static rebalancing baseline on test data
#
# PHASE 3: LOST-DEMAND DQN (Base Paper Reproduction)
#   - Output activations: none, elu, leaky_relu, prelu
#   - Fill levels: 10-50-90, 15-50-85, 0-50-100 (from base paper)
#
# PHASE 4: PROFIT-BASED DQN
#   - Output activations: none, elu, leaky_relu, prelu
#   - Fill levels: 10-50-90, 15-50-85, 0-50-100 (from base paper)
#   - Economic parameter variations
#
# PHASE 5: POLICY COMPARISON
#   - Compare best lost-demand vs best profit-based policies
#
# GT0 Configuration:
#   - 10 stations, 2 vehicles, 15 bike capacity
#   - ~40 trips per 4-hour horizon (7 AM - 11 AM)
################################################################################

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$CODE_DIR/../data/synthetic/GT0"

# Training parameters
TIMESTEPS_LOSTDEMAND=20000      # 20K for lost-demand (fast convergence)
TIMESTEPS_PROFIT=50000          # 50K for profit-based (more complex reward)
EXPLORATION_FRAC=0.5            # Epsilon decay over first 50%
EVAL_EPISODES=50                # Test episodes
OPTIMIZER="adam"

# GT0 network parameters
NUM_STATIONS=10
NUM_VEHICLES=2
VEHICLE_CAPACITY=15

# Economic parameters (profit-based)
TRIP_BASE_FARE=1.00
TRIP_PER_KM=0.75
COST_PER_KM=1.00
LOST_PENALTY=5.00

# Results directories
RESULTS_LOSTDEMAND="$CODE_DIR/results_GT0"
RESULTS_PROFIT="$CODE_DIR/results_profit_GT0"
RESULTS_ECONOMIC="$CODE_DIR/results_economic_sensitivity"
RESULTS_COMPARISON="$CODE_DIR/results_policy_comparison"

# Logging
LOG_FILE="$CODE_DIR/gt0_experiment_log_$(date +'%Y%m%d_%H%M%S').txt"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

section_header() {
    echo "" | tee -a "$LOG_FILE"
    echo "==============================================================================" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "==============================================================================" | tee -a "$LOG_FILE"
}

subsection_header() {
    echo "" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------------------------" | tee -a "$LOG_FILE"
}

# =============================================================================
# PHASE 1: DATA GENERATION
# =============================================================================

generate_synthetic_data() {
    section_header "PHASE 1: SYNTHETIC DATA GENERATION"
    
    if [ -f "$DATA_DIR/GT0_station_network.json" ] && [ -f "$DATA_DIR/GT0_trips_train.csv" ] && [ -f "$DATA_DIR/GT0_trips_test.csv" ]; then
        log "GT0 data already exists. Skipping generation."
        log "  Network: $DATA_DIR/GT0_station_network.json"
        log "  Train trips: $DATA_DIR/GT0_trips_train.csv"
        log "  Test trips: $DATA_DIR/GT0_trips_test.csv"
    else
        log "Generating GT0 synthetic data..."
        python "$CODE_DIR/synthetic_data_generation/generate_gt0_toy.py" 2>&1 | tee -a "$LOG_FILE"
        log "✓ Data generation complete"
    fi
}

# =============================================================================
# PHASE 2: STATIC BASELINE
# =============================================================================

run_static_baseline() {
    section_header "PHASE 2: STATIC BASELINE EVALUATION"
    
    log "Evaluating static rebalancing baseline on TEST data..."
    python "$CODE_DIR/baselines/evaluate_static_baseline.py" \
        --gt "GT0" \
        --episodes $EVAL_EPISODES 2>&1 | tee -a "$LOG_FILE"
    
    log "✓ Static baseline evaluation complete"
}

# =============================================================================
# PHASE 3: LOST-DEMAND DQN (Base Paper Reproduction)
# =============================================================================

run_lostdemand_training() {
    local name=$1
    local extra_args=$2
    local output_dir="$RESULTS_LOSTDEMAND/${name}"
    
    mkdir -p "$output_dir"
    
    log "Training Lost-Demand DQN: $name"
    python "$CODE_DIR/rl_algorithm/train.py" \
        --gt "GT0" \
        --timesteps $TIMESTEPS_LOSTDEMAND \
        --exploration-fraction $EXPLORATION_FRAC \
        --optimizer $OPTIMIZER \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        --output-dir "$output_dir" \
        $extra_args 2>&1 | tee -a "$LOG_FILE"
    
    log "Evaluating: $name on TEST data"
    python "$CODE_DIR/rl_algorithm/evaluate.py" \
        --gt "GT0" \
        --model "$output_dir/GT0_multi_agent_dqn_final.pth" \
        --episodes $EVAL_EPISODES \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        $extra_args 2>&1 | tee -a "$LOG_FILE"
    
    log "✓ Completed: $name"
}

run_lostdemand_experiments() {
    section_header "PHASE 3: LOST-DEMAND DQN EXPERIMENTS (Base Paper)"
    
    mkdir -p "$RESULTS_LOSTDEMAND"
    
    # ---------------------------------------------------------------------
    # 3.1: Output Activation Function Experiments (Fill Level: 10-50-90)
    # ---------------------------------------------------------------------
    subsection_header "3.1: Output Activation Functions (Fill: 10-50-90)"
    
    run_lostdemand_training "activation_none_fill_10_50_90" \
        "--hidden-activation relu --fill-levels 10,50,90"
    
    run_lostdemand_training "activation_elu_fill_10_50_90" \
        "--hidden-activation relu --output-activation elu --fill-levels 10,50,90"
    
    run_lostdemand_training "activation_leaky_relu_fill_10_50_90" \
        "--hidden-activation relu --output-activation leaky_relu --fill-levels 10,50,90"
    
    run_lostdemand_training "activation_prelu_fill_10_50_90" \
        "--hidden-activation relu --output-activation prelu --fill-levels 10,50,90"
    
    # ---------------------------------------------------------------------
    # 3.2: Fill Level Experiments with Best Activation (ELU)
    # ---------------------------------------------------------------------
    subsection_header "3.2: Fill Level Variations (Activation: ELU)"
    
    # 15-50-85 (base paper configuration)
    run_lostdemand_training "activation_elu_fill_15_50_85" \
        "--hidden-activation relu --output-activation elu --fill-levels 15,50,85"
    
    # 0-50-100 (extreme configuration from base paper)
    run_lostdemand_training "activation_elu_fill_0_50_100" \
        "--hidden-activation relu --output-activation elu --fill-levels 0,50,100"
    
    # 10-50-90 (already done in 3.1 with ELU activation)
    
    log "✓ All Lost-Demand DQN experiments complete"
}

# =============================================================================
# PHASE 4: PROFIT-BASED DQN
# =============================================================================

run_profit_training() {
    local name=$1
    local extra_args=$2
    local output_dir="$RESULTS_PROFIT/${name}"
    
    mkdir -p "$output_dir"
    
    log "Training Profit-Based DQN: $name"
    python "$CODE_DIR/rl_algorithm/train_profit.py" \
        --gt GT0 \
        --timesteps $TIMESTEPS_PROFIT \
        --output-dir "$output_dir" \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        --trip-base-fare $TRIP_BASE_FARE \
        --trip-per-km $TRIP_PER_KM \
        --cost-per-km $COST_PER_KM \
        --lost-penalty $LOST_PENALTY \
        $extra_args 2>&1 | tee -a "$LOG_FILE"
    
    log "✓ Completed: $name"
}

run_profit_experiments() {
    section_header "PHASE 4: PROFIT-BASED DQN EXPERIMENTS"
    
    mkdir -p "$RESULTS_PROFIT"
    
    # ---------------------------------------------------------------------
    # 4.1: Output Activation Function Experiments (Fill Level: 10-50-90)
    # ---------------------------------------------------------------------
    subsection_header "4.1: Output Activation Functions (Fill: 10-50-90)"
    
    run_profit_training "activation_none_fill_10_50_90" \
        "--hidden-activation relu --fill-levels 10,50,90"
    
    run_profit_training "activation_elu_fill_10_50_90" \
        "--hidden-activation relu --output-activation elu --fill-levels 10,50,90"
    
    run_profit_training "activation_leaky_relu_fill_10_50_90" \
        "--hidden-activation relu --output-activation leaky_relu --fill-levels 10,50,90"
    
    run_profit_training "activation_prelu_fill_10_50_90" \
        "--hidden-activation relu --output-activation prelu --fill-levels 10,50,90"
    
    # ---------------------------------------------------------------------
    # 4.2: Fill Level Experiments with Best Activation (PReLU)
    # ---------------------------------------------------------------------
    subsection_header "4.2: Fill Level Variations (Activation: PReLU)"
    
    # 15-50-85 (base paper configuration)
    run_profit_training "activation_prelu_fill_15_50_85" \
        "--hidden-activation relu --output-activation prelu --fill-levels 15,50,85"
    
    # 0-50-100 (extreme configuration from base paper)
    run_profit_training "activation_prelu_fill_0_50_100" \
        "--hidden-activation relu --output-activation prelu --fill-levels 0,50,100"
    
    # 10-50-90 (already done in 4.1 with PReLU activation)
    
    # ---------------------------------------------------------------------
    # 4.3: Economic Parameter Sensitivity (with PReLU, Fill: 10-50-90)
    # ---------------------------------------------------------------------
    subsection_header "4.3: Economic Parameter Sensitivity"
    
    mkdir -p "$RESULTS_ECONOMIC"
    
    # Baseline configuration
    log "Running economic sensitivity: baseline (cost=1.00, penalty=5.00)"
    python "$CODE_DIR/rl_algorithm/train_profit.py" \
        --gt GT0 \
        --timesteps $TIMESTEPS_PROFIT \
        --output-dir "$RESULTS_ECONOMIC/baseline" \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        --trip-base-fare $TRIP_BASE_FARE \
        --trip-per-km $TRIP_PER_KM \
        --cost-per-km 1.00 \
        --lost-penalty 5.00 \
        --output-activation prelu \
        --fill-levels 10,50,90 2>&1 | tee -a "$LOG_FILE"
    
    # Low truck cost
    log "Running economic sensitivity: low_cost (cost=0.50, penalty=5.00)"
    python "$CODE_DIR/rl_algorithm/train_profit.py" \
        --gt GT0 \
        --timesteps $TIMESTEPS_PROFIT \
        --output-dir "$RESULTS_ECONOMIC/low_cost" \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        --trip-base-fare $TRIP_BASE_FARE \
        --trip-per-km $TRIP_PER_KM \
        --cost-per-km 0.50 \
        --lost-penalty 5.00 \
        --output-activation prelu \
        --fill-levels 10,50,90 2>&1 | tee -a "$LOG_FILE"
    
    # High truck cost
    log "Running economic sensitivity: high_cost (cost=2.00, penalty=5.00)"
    python "$CODE_DIR/rl_algorithm/train_profit.py" \
        --gt GT0 \
        --timesteps $TIMESTEPS_PROFIT \
        --output-dir "$RESULTS_ECONOMIC/high_cost" \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        --trip-base-fare $TRIP_BASE_FARE \
        --trip-per-km $TRIP_PER_KM \
        --cost-per-km 2.00 \
        --lost-penalty 5.00 \
        --output-activation prelu \
        --fill-levels 10,50,90 2>&1 | tee -a "$LOG_FILE"
    
    # Low lost demand penalty
    log "Running economic sensitivity: low_penalty (cost=1.00, penalty=2.00)"
    python "$CODE_DIR/rl_algorithm/train_profit.py" \
        --gt GT0 \
        --timesteps $TIMESTEPS_PROFIT \
        --output-dir "$RESULTS_ECONOMIC/low_penalty" \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        --trip-base-fare $TRIP_BASE_FARE \
        --trip-per-km $TRIP_PER_KM \
        --cost-per-km 1.00 \
        --lost-penalty 2.00 \
        --output-activation prelu \
        --fill-levels 10,50,90 2>&1 | tee -a "$LOG_FILE"
    
    # High lost demand penalty
    log "Running economic sensitivity: high_penalty (cost=1.00, penalty=10.00)"
    python "$CODE_DIR/rl_algorithm/train_profit.py" \
        --gt GT0 \
        --timesteps $TIMESTEPS_PROFIT \
        --output-dir "$RESULTS_ECONOMIC/high_penalty" \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        --trip-base-fare $TRIP_BASE_FARE \
        --trip-per-km $TRIP_PER_KM \
        --cost-per-km 1.00 \
        --lost-penalty 10.00 \
        --output-activation prelu \
        --fill-levels 10,50,90 2>&1 | tee -a "$LOG_FILE"
    
    # Cheap truck + aggressive penalty (encourage more rebalancing)
    log "Running economic sensitivity: cheap_aggressive (cost=0.50, penalty=10.00)"
    python "$CODE_DIR/rl_algorithm/train_profit.py" \
        --gt GT0 \
        --timesteps $TIMESTEPS_PROFIT \
        --output-dir "$RESULTS_ECONOMIC/cheap_aggressive" \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        --trip-base-fare $TRIP_BASE_FARE \
        --trip-per-km $TRIP_PER_KM \
        --cost-per-km 0.50 \
        --lost-penalty 10.00 \
        --output-activation prelu \
        --fill-levels 10,50,90 2>&1 | tee -a "$LOG_FILE"
    
    # Expensive truck + conservative penalty (discourage rebalancing)
    log "Running economic sensitivity: expensive_conservative (cost=2.00, penalty=2.00)"
    python "$CODE_DIR/rl_algorithm/train_profit.py" \
        --gt GT0 \
        --timesteps $TIMESTEPS_PROFIT \
        --output-dir "$RESULTS_ECONOMIC/expensive_conservative" \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        --trip-base-fare $TRIP_BASE_FARE \
        --trip-per-km $TRIP_PER_KM \
        --cost-per-km 2.00 \
        --lost-penalty 2.00 \
        --output-activation prelu \
        --fill-levels 10,50,90 2>&1 | tee -a "$LOG_FILE"
    
    log "✓ All Profit-Based DQN experiments complete"
}

# =============================================================================
# PHASE 5: POLICY COMPARISON
# =============================================================================

run_policy_comparison() {
    section_header "PHASE 5: POLICY COMPARISON (Lost-Demand vs Profit)"
    
    mkdir -p "$RESULTS_COMPARISON"
    
    # Define best models from each approach
    LOST_DEMAND_MODEL="$RESULTS_LOSTDEMAND/activation_elu_fill_10_50_90/GT0_multi_agent_dqn_final.pth"
    PROFIT_MODEL="$RESULTS_PROFIT/activation_prelu_fill_10_50_90/GT0_profit_dqn_best.pth"
    
    # Check if models exist
    if [ ! -f "$LOST_DEMAND_MODEL" ]; then
        log "⚠ Lost-demand model not found: $LOST_DEMAND_MODEL"
        log "  Skipping policy comparison"
        return
    fi
    
    if [ ! -f "$PROFIT_MODEL" ]; then
        log "⚠ Profit model not found: $PROFIT_MODEL"
        log "  Skipping policy comparison"
        return
    fi
    
    log "Models found:"
    log "  Lost-Demand: $LOST_DEMAND_MODEL"
    log "  Profit:      $PROFIT_MODEL"
    
    # Run comparison script if it exists
    if [ -f "$RESULTS_COMPARISON/compare_policies.py" ]; then
        log "Running policy comparison..."
        python "$RESULTS_COMPARISON/compare_policies.py" \
            --lost-demand-model "$LOST_DEMAND_MODEL" \
            --profit-model "$PROFIT_MODEL" \
            --episodes $EVAL_EPISODES \
            --output "$RESULTS_COMPARISON/comparison_results.json" 2>&1 | tee -a "$LOG_FILE"
        
        log "✓ Policy comparison complete"
    else
        log "⚠ compare_policies.py not found. Run ./run_policy_comparison.sh to generate it."
    fi
}

# =============================================================================
# SUMMARY REPORT
# =============================================================================

generate_summary() {
    section_header "EXPERIMENT SUMMARY"
    
    log ""
    log "GT0 Configuration:"
    log "  Stations: $NUM_STATIONS"
    log "  Vehicles: $NUM_VEHICLES"
    log "  Vehicle Capacity: $VEHICLE_CAPACITY"
    log ""
    log "Lost-Demand DQN Experiments:"
    log "  Timesteps: $TIMESTEPS_LOSTDEMAND"
    log "  Activations tested: none, elu, leaky_relu, prelu"
    log "  Fill levels tested: 10-50-90, 15-50-85, 0-50-100 (base paper)"
    log "  Results: $RESULTS_LOSTDEMAND"
    log ""
    log "Profit-Based DQN Experiments:"
    log "  Timesteps: $TIMESTEPS_PROFIT"
    log "  Activations tested: none, elu, leaky_relu, prelu"
    log "  Fill levels tested: 10-50-90, 15-50-85, 0-50-100 (base paper)"
    log "  Economic parameters tested:"
    log "    - baseline: cost=\$1.00/km, penalty=\$5.00"
    log "    - low_cost: cost=\$0.50/km, penalty=\$5.00"
    log "    - high_cost: cost=\$2.00/km, penalty=\$5.00"
    log "    - low_penalty: cost=\$1.00/km, penalty=\$2.00"
    log "    - high_penalty: cost=\$1.00/km, penalty=\$10.00"
    log "    - cheap_aggressive: cost=\$0.50/km, penalty=\$10.00"
    log "    - expensive_conservative: cost=\$2.00/km, penalty=\$2.00"
    log "  Results: $RESULTS_PROFIT, $RESULTS_ECONOMIC"
    log ""
    log "Log file: $LOG_FILE"
    log ""
    
    # Count completed experiments
    local ld_count=$(find "$RESULTS_LOSTDEMAND" -name "*.pth" 2>/dev/null | wc -l | tr -d ' ')
    local pf_count=$(find "$RESULTS_PROFIT" -name "*.pth" 2>/dev/null | wc -l | tr -d ' ')
    local ec_count=$(find "$RESULTS_ECONOMIC" -name "*.pth" 2>/dev/null | wc -l | tr -d ' ')
    
    log "Completed Models:"
    log "  Lost-Demand DQN: $ld_count"
    log "  Profit-Based DQN: $pf_count"
    log "  Economic Sensitivity: $ec_count"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    section_header "GT0 COMPLETE EXPERIMENT SUITE"
    log "Started at: $(date)"
    log "Working directory: $CODE_DIR"
    log ""
    
    # Parse command line arguments
    SKIP_DATA=false
    SKIP_BASELINE=false
    SKIP_LOSTDEMAND=false
    SKIP_PROFIT=false
    SKIP_COMPARISON=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-data)
                SKIP_DATA=true
                shift
                ;;
            --skip-baseline)
                SKIP_BASELINE=true
                shift
                ;;
            --skip-lostdemand)
                SKIP_LOSTDEMAND=true
                shift
                ;;
            --skip-profit)
                SKIP_PROFIT=true
                shift
                ;;
            --skip-comparison)
                SKIP_COMPARISON=true
                shift
                ;;
            --only-data)
                SKIP_BASELINE=true
                SKIP_LOSTDEMAND=true
                SKIP_PROFIT=true
                SKIP_COMPARISON=true
                shift
                ;;
            --only-baseline)
                SKIP_DATA=true
                SKIP_LOSTDEMAND=true
                SKIP_PROFIT=true
                SKIP_COMPARISON=true
                shift
                ;;
            --only-lostdemand)
                SKIP_DATA=true
                SKIP_BASELINE=true
                SKIP_PROFIT=true
                SKIP_COMPARISON=true
                shift
                ;;
            --only-profit)
                SKIP_DATA=true
                SKIP_BASELINE=true
                SKIP_LOSTDEMAND=true
                SKIP_COMPARISON=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --skip-data         Skip synthetic data generation"
                echo "  --skip-baseline     Skip static baseline evaluation"
                echo "  --skip-lostdemand   Skip lost-demand DQN experiments"
                echo "  --skip-profit       Skip profit-based DQN experiments"
                echo "  --skip-comparison   Skip policy comparison"
                echo "  --only-data         Only run data generation"
                echo "  --only-baseline     Only run static baseline"
                echo "  --only-lostdemand   Only run lost-demand experiments"
                echo "  --only-profit       Only run profit experiments"
                echo "  --help              Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute phases
    if [ "$SKIP_DATA" = false ]; then
        generate_synthetic_data
    fi
    
    if [ "$SKIP_BASELINE" = false ]; then
        run_static_baseline
    fi
    
    if [ "$SKIP_LOSTDEMAND" = false ]; then
        run_lostdemand_experiments
    fi
    
    if [ "$SKIP_PROFIT" = false ]; then
        run_profit_experiments
    fi
    
    if [ "$SKIP_COMPARISON" = false ]; then
        run_policy_comparison
    fi
    
    # Generate summary
    generate_summary
    
    section_header "ALL GT0 EXPERIMENTS COMPLETE!"
    log "Finished at: $(date)"
}

# Run main function
main "$@"
