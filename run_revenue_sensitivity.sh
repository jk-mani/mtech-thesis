#!/bin/bash
# Revenue Sensitivity Experiment for GT0
# Tests different trip revenue configurations
# NOTE: baseline_revenue already exists as results_economic_sensitivity/baseline

set -e

GT="GT0"
TIMESTEPS=50000
BASE_DIR="results_revenue_sensitivity"

# Only NEW revenue configurations (baseline already exists)
# Format: name:base_fare:per_km:cost_per_km:lost_penalty
EXPERIMENTS=(
    "low_revenue:0.50:0.50:1.00:5.00"
    "high_revenue:2.00:1.00:1.00:5.00"
)

echo "=============================================="
echo "Revenue Sensitivity Experiment"
echo "GT: $GT | Timesteps: $TIMESTEPS"
echo "=============================================="

mkdir -p $BASE_DIR

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r name base_fare per_km cost_km penalty <<< "$exp"
    
    OUTPUT_DIR="${BASE_DIR}/${name}"
    
    echo ""
    echo "=============================================="
    echo "Running: $name"
    echo "  Base fare: \$$base_fare"
    echo "  Per-km rate: \$$per_km"
    echo "  Cost/km: \$$cost_km"
    echo "  Lost penalty: \$$penalty"
    echo "  Output: $OUTPUT_DIR"
    echo "=============================================="
    
    python rl_algorithm/train_profit.py \
        --gt $GT \
        --timesteps $TIMESTEPS \
        --trip-base-fare $base_fare \
        --trip-per-km $per_km \
        --cost-per-km $cost_km \
        --lost-penalty $penalty \
        --output-dir $OUTPUT_DIR \
        --output-activation prelu
    
    echo "✅ Completed: $name"
done

echo ""
echo "=============================================="
echo "All revenue sensitivity experiments complete!"
echo "Results saved in: $BASE_DIR/"
echo "=============================================="

# Summary
echo ""
echo "Experiment Summary:"
echo "---------------------------------------------------"
printf "%-25s %10s %10s %10s %10s\n" "Name" "Base" "Per-km" "Cost/km" "Penalty"
echo "---------------------------------------------------"
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r name base_fare per_km cost_km penalty <<< "$exp"
    printf "%-25s %10s %10s %10s %10s\n" "$name" "\$$base_fare" "\$$per_km" "\$$cost_km" "\$$penalty"
done
