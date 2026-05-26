#!/bin/bash
###############################################################################
# THESIS EXPERIMENT RUNNER (clean, like-for-like comparison)
#
# Goal: produce a single coherent set of results that supports two stories:
#   Tier B  — architecture ablation: 4 activations × 3 fill levels × 2 reward
#             types, all trained the same way, all evaluated on the SAME 50
#             test episodes, reporting the MEAN profit and MEAN lost-demand
#             rate (no cherry-picking the best episode).
#   Tier A  — economic-regime study: 3 deliberately chosen (cost_per_km,
#             lost_penalty) regimes designed to expose where the profit reward
#             beats the lost-demand reward. For each regime, we train a
#             dedicated profit-DQN and re-score the lost-demand DQN under that
#             regime's economics.
#
# Skips any training run whose output dir already contains a final model so
# this script is safe to re-run.
###############################################################################

set -e

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"

# ----- Common training params --------------------------------------------------
TIMESTEPS_LD=20000        # lost-demand reward converges fast
TIMESTEPS_PR=50000        # profit reward needs longer
EXPLORATION_FRAC=0.5
NUM_STATIONS=10
NUM_VEHICLES=2
VEHICLE_CAPACITY=15

# Default revenue model (kept fixed across all profit experiments)
TRIP_BASE_FARE=1.00
TRIP_PER_KM=0.75

# ----- Output dirs -------------------------------------------------------------
RESULTS_LD="$CODE_DIR/results_GT0"
RESULTS_PR="$CODE_DIR/results_profit_GT0"
RESULTS_REGIME="$CODE_DIR/results_regimes"

mkdir -p "$RESULTS_LD" "$RESULTS_PR" "$RESULTS_REGIME"

LOG="$CODE_DIR/thesis_experiments_$(date +'%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG") 2>&1

stamp() { date +'%H:%M:%S'; }

# ==============================================================================
# Tier B helpers — architecture ablation
# ==============================================================================
train_lostdemand() {
    local act=$1 fill=$2
    local fill_csv=${fill//-/,}
    local fill_us=${fill//-/_}
    local act_arg=""
    [ "$act" != "none" ] && act_arg="--output-activation $act"

    local out="$RESULTS_LD/activation_${act}_fill_${fill_us}"
    local final="$out/GT0_multi_agent_dqn_final.pth"

    if [ -f "$final" ]; then
        echo "[$(stamp)] LD-DQN  ${act}/${fill}  : already trained, skipping"
        return
    fi
    echo "[$(stamp)] LD-DQN  ${act}/${fill}  : training ${TIMESTEPS_LD} steps"
    mkdir -p "$out"
    python "$CODE_DIR/rl_algorithm/train.py" \
        --gt GT0 \
        --timesteps $TIMESTEPS_LD \
        --exploration-fraction $EXPLORATION_FRAC \
        --num-stations $NUM_STATIONS \
        --num-vehicles $NUM_VEHICLES \
        --vehicle-capacity $VEHICLE_CAPACITY \
        --hidden-activation relu \
        $act_arg \
        --fill-levels "$fill_csv" \
        --output-dir "$out"
}

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
# TIER B: Architecture ablation (4 activations × 3 fills × 2 reward types)
# ==============================================================================
echo ""
echo "================================================================"
echo "TIER B — architecture ablation (4 act × 3 fill × 2 reward)"
echo "================================================================"
ACTS=(none elu leaky_relu prelu)
FILLS=(10-50-90 15-50-85 0-50-100)

for act in "${ACTS[@]}"; do
    for fill in "${FILLS[@]}"; do
        train_lostdemand "$act" "$fill"
    done
done

for act in "${ACTS[@]}"; do
    for fill in "${FILLS[@]}"; do
        train_profit "$act" "$fill"   # default regime ($1/km, $5/trip)
    done
done

# ==============================================================================
# TIER A: Economic regime study
#   Three regimes designed for a monotonic profit-DQN advantage:
#     service       : $0.50/km truck, $10/trip penalty  (tie expected)
#     moderate_cost : $2.00/km truck, $3/trip penalty   (moderate gap)
#     cost          : $5.00/km truck, $1/trip penalty   (large gap)
#   Profit-DQN architecture: leaky_relu + fill 10-50-90 (chapter best).
# ==============================================================================
PR_ACT=leaky_relu
PR_FILL=10-50-90

echo ""
echo "================================================================"
echo "TIER A — economic regime study (profit-DQN re-trained per regime)"
echo "================================================================"

train_profit  $PR_ACT $PR_FILL  0.50  10.00  service
train_profit  $PR_ACT $PR_FILL  2.00   3.00  moderate_cost
train_profit  $PR_ACT $PR_FILL  5.00   1.00  cost

# ==============================================================================
# UNIFIED EVALUATION — same test set, mean metrics, all 3 regimes
# ==============================================================================
echo ""
echo "================================================================"
echo "UNIFIED EVALUATION — every model, same 50 test episodes"
echo "================================================================"
python "$CODE_DIR/evaluate_thesis_experiments.py"

echo ""
echo "================================================================"
echo "ALL DONE."
echo "Log: $LOG"
echo "Aggregated results: $CODE_DIR/results_thesis/thesis_results.json"
echo "================================================================"
