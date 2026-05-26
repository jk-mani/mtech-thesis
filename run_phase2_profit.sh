#!/bin/bash
###############################################################################
# PHASE 2 — Profit-aware DQN
#   Train 12 profit-DQN models (default regime $1/km, $5/trip) covering the
#   architecture matrix, plus 3 regime-specific profit-DQNs for the Tier-A
#   economic-regime study. Re-runs the unified evaluator at the end so the
#   final thesis_results.json has every model in one place.
###############################################################################
set -e

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"

TIMESTEPS_PR=50000
NUM_STATIONS=10
NUM_VEHICLES=2
VEHICLE_CAPACITY=15
TRIP_BASE_FARE=1.00
TRIP_PER_KM=0.75

RESULTS_PR="$CODE_DIR/results_profit_GT0"
RESULTS_REGIME="$CODE_DIR/results_regimes"
mkdir -p "$RESULTS_PR" "$RESULTS_REGIME"

LOG="$CODE_DIR/phase2_profit_$(date +'%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG") 2>&1

stamp() { date +'%H:%M:%S'; }

# ---------------------------------------------------------------------------
# Helper: train one profit-DQN
#   $1 = activation tag (none|elu|leaky_relu|prelu)
#   $2 = fill level (e.g. "10-50-90")
#   $3 = cost_per_km (default 1.00)
#   $4 = lost_penalty (default 5.00)
#   $5 = regime tag (default "default" → goes to results_profit_GT0,
#                    else → goes to results_regimes/<tag>_act_..._fill_...)
# ---------------------------------------------------------------------------
train_profit() {
    local act=$1 fill=$2 cost=${3:-1.00} pen=${4:-5.00} tag=${5:-default}
    local fill_csv=${fill//-/,}
    local fill_us=${fill//-/_}
    local act_arg=""
    [ "$act" != "none" ] && act_arg="--output-activation $act"

    if [ "$tag" = "default" ]; then
        local out="$RESULTS_PR/activation_${act}_fill_${fill_us}"
    else
        local out="$RESULTS_REGIME/${tag}_act_${act}_fill_${fill_us}"
    fi
    local final="$out/GT0_profit_dqn_final.pth"

    if [ -f "$final" ]; then
        echo "[$(stamp)] PR-DQN  ${act}/${fill}  cost=\$${cost} pen=\$${pen}  tag=${tag}  : already trained, skipping"
        return
    fi
    echo "[$(stamp)] PR-DQN  ${act}/${fill}  cost=\$${cost} pen=\$${pen}  tag=${tag}  : training ${TIMESTEPS_PR} steps"
    mkdir -p "$out"
    python "$CODE_DIR/rl_algorithm/train_profit.py" \
        --gt GT0 \
        --timesteps $TIMESTEPS_PR \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        --trip-base-fare $TRIP_BASE_FARE \
        --trip-per-km $TRIP_PER_KM \
        --cost-per-km $cost \
        --lost-penalty $pen \
        --hidden-activation relu \
        $act_arg \
        --fill-levels "$fill_csv" \
        --output-dir "$out"
}

# ==============================================================================
# 2A — Architecture matrix (12 profit-DQN, default $1/km $5/trip)
# ==============================================================================
echo ""
echo "================================================================"
echo "PHASE 2A — Profit-DQN architecture sweep (4 act × 3 fill)"
echo "================================================================"
echo "Started: $(date)"
echo ""

ACTS=(none elu leaky_relu prelu)
FILLS=(10-50-90 15-50-85 0-50-100)

for act in "${ACTS[@]}"; do
    for fill in "${FILLS[@]}"; do
        train_profit "$act" "$fill"
    done
done

# ==============================================================================
# 2B — Regime-specific profit-DQN (Tier-A economic study)
#   Architecture: leaky_relu + 10-50-90 (chapter best for profit reward).
# ==============================================================================
echo ""
echo "================================================================"
echo "PHASE 2B — Tier-A regime profit-DQN (3 regimes)"
echo "================================================================"

PR_ACT=leaky_relu
PR_FILL=10-50-90

train_profit  $PR_ACT $PR_FILL  0.50  10.00  service
train_profit  $PR_ACT $PR_FILL  2.00   3.00  moderate_cost
train_profit  $PR_ACT $PR_FILL  5.00   1.00  cost

# ==============================================================================
# UNIFIED EVALUATION — every model from Phase 1 + Phase 2 in one JSON/CSV
# ==============================================================================
echo ""
echo "================================================================"
echo "UNIFIED EVALUATION — static baseline + 12 LD-DQN + 12 PR-DQN + 3 regime PR-DQN"
echo "================================================================"
python "$CODE_DIR/evaluate_thesis_experiments.py"

echo ""
echo "================================================================"
echo "PHASE 2 COMPLETE."
echo "Finished: $(date)"
echo "Log:                 $LOG"
echo "Aggregated results:  $CODE_DIR/results_thesis/thesis_results.json"
echo "Flat summary:        $CODE_DIR/results_thesis/thesis_summary.csv"
echo "================================================================"
