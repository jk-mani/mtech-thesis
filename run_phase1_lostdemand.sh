#!/bin/bash
###############################################################################
# PHASE 1 — Base-paper reproduction
#   Train all 12 lost-demand DQN models (4 activations × 3 fill levels) and
#   evaluate them against the static (no-rebalancing) baseline on the same 50
#   test episodes. Produces results_thesis/thesis_results.json with the static
#   baseline + 12 LD-DQN models.
###############################################################################
set -e

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"

TIMESTEPS_LD=20000
EXPLORATION_FRAC=0.5
NUM_STATIONS=10
NUM_VEHICLES=2
VEHICLE_CAPACITY=15

RESULTS_LD="$CODE_DIR/results_GT0"
mkdir -p "$RESULTS_LD"

LOG="$CODE_DIR/phase1_lostdemand_$(date +'%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG") 2>&1

stamp() { date +'%H:%M:%S'; }

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

echo ""
echo "================================================================"
echo "PHASE 1 — Lost-demand DQN architecture sweep (4 act × 3 fill)"
echo "================================================================"
echo "Started: $(date)"
echo ""

ACTS=(none elu leaky_relu prelu)
FILLS=(10-50-90 15-50-85 0-50-100)

for act in "${ACTS[@]}"; do
    for fill in "${FILLS[@]}"; do
        train_lostdemand "$act" "$fill"
    done
done

echo ""
echo "================================================================"
echo "UNIFIED EVALUATION — static baseline + 12 LD-DQN models"
echo "================================================================"
python "$CODE_DIR/evaluate_thesis_experiments.py"

echo ""
echo "================================================================"
echo "PHASE 1 COMPLETE."
echo "Finished: $(date)"
echo "Log:                 $LOG"
echo "Aggregated results:  $CODE_DIR/results_thesis/thesis_results.json"
echo "Flat summary:        $CODE_DIR/results_thesis/thesis_summary.csv"
echo "================================================================"
