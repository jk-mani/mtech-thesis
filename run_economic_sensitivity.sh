#!/bin/bash
# Economic Parameter Sensitivity Experiment for GT0
# Tests different combinations of cost_per_km and lost_demand_penalty

set -e

# Configuration
GT_NAME="GT0"
TIMESTEPS=50000
BASE_DIR="results_economic_sensitivity"
ACTIVATION="prelu"  # Best from previous experiments

# Fixed parameters
TRIP_BASE_FARE=1.00
TRIP_PER_KM=0.75

echo "=============================================="
echo "Economic Parameter Sensitivity Experiment"
echo "GT: $GT_NAME | Timesteps: $TIMESTEPS"
echo "=============================================="

# Create results directory
mkdir -p "$BASE_DIR"

# Experiment configurations: (name, cost_per_km, lost_penalty)
declare -a EXPERIMENTS=(
    "baseline:1.00:5.00"
    "low_cost:0.50:5.00"
    "high_cost:2.00:5.00"
    "low_penalty:1.00:2.00"
    "high_penalty:1.00:10.00"
    "cheap_aggressive:0.50:10.00"
    "expensive_conservative:2.00:2.00"
)

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r name cost_km penalty <<< "$exp"
    
    OUTPUT_DIR="${BASE_DIR}/${name}"
    
    echo ""
    echo "=============================================="
    echo "Running: $name"
    echo "  cost_per_km: \$$cost_km"
    echo "  lost_penalty: \$$penalty"
    echo "  Output: $OUTPUT_DIR"
    echo "=============================================="
    
    python rl_algorithm/train_profit.py \
        --gt "$GT_NAME" \
        --timesteps "$TIMESTEPS" \
        --trip-base-fare "$TRIP_BASE_FARE" \
        --trip-per-km "$TRIP_PER_KM" \
        --cost-per-km "$cost_km" \
        --lost-penalty "$penalty" \
        --output-activation "$ACTIVATION" \
        --output-dir "$OUTPUT_DIR"
    
    echo "✅ Completed: $name"
done

echo ""
echo "=============================================="
echo "All economic sensitivity experiments complete!"
echo "Results saved in: $BASE_DIR/"
echo "=============================================="

# Summary of configurations
echo ""
echo "Experiment Summary:"
echo "-------------------"
printf "%-25s %12s %12s\n" "Name" "Cost/km" "Lost Penalty"
echo "---------------------------------------------------"
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r name cost_km penalty <<< "$exp"
    printf "%-25s %12s %12s\n" "$name" "\$$cost_km" "\$$penalty"
done
