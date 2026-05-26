"""
Unified, like-for-like evaluation of all trained DQN models for the thesis.

For every trained model under
  - results_GT0/activation_<act>_fill_<fill>/  (lost-demand DQN)
  - results_profit_GT0/activation_<act>_fill_<fill>/  (profit DQN, default regime)
  - results_regimes/<regime>_act_<act>_fill_<fill>/  (profit DQN, regime-specific)

we run ONE deterministic 50-episode rollout on the GT0 TEST set and re-score
the trajectory under each of THREE economic regimes:

    Tier-A regimes (monotonic gap by design)
    -----------------------------------------
    service        :  cost_per_km = $0.50,  lost_penalty = $10.00  (tie expected)
    moderate_cost  :  cost_per_km = $2.00,  lost_penalty =  $3.00  (moderate gap)
    cost           :  cost_per_km = $5.00,  lost_penalty =  $1.00  (large gap)

We report the MEAN profit and MEAN lost-demand rate over the 50 episodes (no
"best episode" cherry-picking) plus per-episode arrays for plotting.

Output: results_thesis/thesis_results.json
        results_thesis/thesis_summary.csv
"""

import sys
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate_with_profit import evaluate_policy_with_profit  # noqa: E402
from rl_algorithm.continuous_time_simulator import ContinuousTimeSimulator  # noqa: E402
from rl_algorithm.events import CustomerRental, CustomerReturn, VehicleArrival  # noqa: E402
import heapq  # noqa: E402

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
RESULTS_LD     = ROOT / "results_GT0"
RESULTS_PR     = ROOT / "results_profit_GT0"
RESULTS_REGIME = ROOT / "results_regimes"
OUT_DIR        = ROOT / "results_thesis"
OUT_DIR.mkdir(exist_ok=True)

NUM_EPISODES = 50

REGIMES = {
    "service":        {"cost_per_km": 0.50, "lost_penalty": 10.0},
    "moderate_cost":  {"cost_per_km": 2.00, "lost_penalty":  3.0},
    "cost":           {"cost_per_km": 5.00, "lost_penalty":  1.0},
}

TRIP_BASE_FARE = 1.0
TRIP_PER_KM    = 0.75


# --------------------------------------------------------------------------- #
# Model discovery
# --------------------------------------------------------------------------- #
DIR_RE = re.compile(
    r"^activation_(?P<act>none|elu|leaky_relu|prelu)_fill_(?P<fill>\d+_\d+_\d+)$"
)
REGIME_DIR_RE = re.compile(
    r"^(?P<regime>service|moderate_cost|cost)_act_(?P<act>none|elu|leaky_relu|prelu)_"
    r"fill_(?P<fill>\d+_\d+_\d+)$"
)


def parse_fill(s):
    return [int(x) / 100.0 for x in s.split("_")]


def parse_act(s):
    return None if s == "none" else s


def evaluate_static_baseline_with_profit(
    gt_name='GT0',
    num_episodes=50,
    cost_per_km=1.0,
    lost_penalty=5.0,
    trip_base_fare=1.0,
    trip_per_km=0.75,
    num_vehicles=2,
    vehicle_capacity=15,
):
    """Run the static (no-rebalancing) baseline on the GT0 TEST set and return
    per-episode dicts shaped identically to evaluate_policy_with_profit().

    Static = vehicles initialised but never moved. Truck distance is 0 by
    definition, so the only cost is the lost-demand penalty.
    """
    base_dir = Path(__file__).parent.parent / 'data' / 'synthetic' / gt_name
    network_file = str(base_dir / f'{gt_name}_station_network.json')
    trips_file = str(base_dir / f'{gt_name}_trips_test.csv')

    sim = ContinuousTimeSimulator(
        network_file=network_file,
        trips_file=trips_file,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
    )

    results = []
    for ep in range(num_episodes):
        day = (ep % 50) + 1
        sim.reset(day=day)

        # Drain events without taking any vehicle action
        while sim.current_time < sim.episode_end_time:
            if not sim.event_queue:
                break
            event = sim.event_queue[0]
            if event.time >= sim.episode_end_time:
                break
            heapq.heappop(sim.event_queue)
            sim.current_time = event.time
            if isinstance(event, CustomerRental):
                sim._process_rental(event)
            elif isinstance(event, CustomerReturn):
                sim._process_return(event)
            elif isinstance(event, VehicleArrival):
                pass  # static = no action

        metrics = sim.get_metrics()
        successful_trips = metrics['successful_rentals'] + metrics['successful_returns']
        lost_trips = metrics['total_lost_demand']

        avg_trip_km = 2.5  # matches evaluate_policy_with_profit convention
        revenue    = successful_trips * (trip_base_fare + trip_per_km * avg_trip_km)
        truck_cost = 0.0
        lost_cost  = lost_trips * lost_penalty
        profit     = revenue - truck_cost - lost_cost

        results.append({
            'day': day,
            'lost_demand_rate': metrics['total_lost_demand_rate'],
            'lost_trips': lost_trips,
            'successful_trips': successful_trips,
            'action_count': 0,
            'revenue': revenue,
            'truck_distance_km': 0.0,
            'truck_cost': truck_cost,
            'lost_cost': lost_cost,
            'profit': profit,
        })
    return results


def discover_models():
    """Return list of (kind, label, model_path, output_activation, fill_levels, regime_tag)."""
    models = []

    # Static baseline (always included; no checkpoint file)
    models.append({
        "kind":               "static_baseline",
        "label":              "Static baseline (no rebalancing)",
        "act_str":            "n/a",
        "fill_str":           "n/a",
        "model_path":         "(no model — pure simulator)",
        "output_activation":  None,
        "fill_levels":        None,
        "regime_tag":         None,
    })

    # Lost-demand DQN
    for d in sorted(RESULTS_LD.glob("activation_*_fill_*")):
        m = DIR_RE.match(d.name)
        if not m:
            continue
        ckpt = d / "GT0_multi_agent_dqn_final.pth"
        if not ckpt.exists():
            continue
        models.append({
            "kind":               "lost_demand",
            "label":              f"LD  {m['act']}  {m['fill']}",
            "act_str":            m["act"],
            "fill_str":           m["fill"],
            "model_path":         str(ckpt),
            "output_activation":  parse_act(m["act"]),
            "fill_levels":        parse_fill(m["fill"]),
            "regime_tag":         None,
        })

    # Profit DQN — default regime
    for d in sorted(RESULTS_PR.glob("activation_*_fill_*")):
        m = DIR_RE.match(d.name)
        if not m:
            continue
        ckpt = d / "GT0_profit_dqn_final.pth"
        if not ckpt.exists():
            ckpt = d / "GT0_profit_dqn_best.pth"
            if not ckpt.exists():
                continue
        models.append({
            "kind":               "profit",
            "label":              f"PR  {m['act']}  {m['fill']}",
            "act_str":            m["act"],
            "fill_str":           m["fill"],
            "model_path":         str(ckpt),
            "output_activation":  parse_act(m["act"]),
            "fill_levels":        parse_fill(m["fill"]),
            "regime_tag":         "default_train",
        })

    # Profit DQN — regime-specific (in results_regimes/)
    if RESULTS_REGIME.exists():
        for d in sorted(RESULTS_REGIME.iterdir()):
            if not d.is_dir() and not d.is_symlink():
                continue
            m = REGIME_DIR_RE.match(d.name)
            if not m:
                continue
            target = d.resolve()
            ckpt = target / "GT0_profit_dqn_final.pth"
            if not ckpt.exists():
                ckpt = target / "GT0_profit_dqn_best.pth"
                if not ckpt.exists():
                    continue
            models.append({
                "kind":               "profit_regime",
                "label":              f"PR-{m['regime']}  {m['act']}  {m['fill']}",
                "act_str":            m["act"],
                "fill_str":           m["fill"],
                "model_path":         str(ckpt),
                "output_activation":  parse_act(m["act"]),
                "fill_levels":        parse_fill(m["fill"]),
                "regime_tag":         m["regime"],
            })

    return models


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_trajectory(episodes, cost_per_km, lost_penalty):
    """Re-compute profit per episode for the given economic regime."""
    out = []
    for r in episodes:
        revenue   = r["revenue"]
        truck_km  = r["truck_distance_km"]
        lost_n    = r["lost_trips"]
        truck_cost = truck_km * cost_per_km
        lost_cost  = lost_n * lost_penalty
        profit     = revenue - truck_cost - lost_cost
        out.append({
            "profit":     profit,
            "revenue":    revenue,
            "truck_cost": truck_cost,
            "lost_cost":  lost_cost,
        })
    return out


def aggregate(episodes_raw, regimes):
    """episodes_raw = list of dicts from evaluate_policy_with_profit."""
    base = {
        "n_episodes":       len(episodes_raw),
        "mean_lost_rate":   float(np.mean([r["lost_demand_rate"] for r in episodes_raw])),
        "std_lost_rate":    float(np.std([r["lost_demand_rate"]  for r in episodes_raw])),
        "mean_lost_trips":  float(np.mean([r["lost_trips"]       for r in episodes_raw])),
        "mean_truck_km":    float(np.mean([r["truck_distance_km"] for r in episodes_raw])),
        "mean_actions":     float(np.mean([r["action_count"]     for r in episodes_raw])),
        "mean_revenue":     float(np.mean([r["revenue"]          for r in episodes_raw])),
    }
    by_regime = {}
    for name, p in regimes.items():
        rescored = score_trajectory(episodes_raw, **p)
        profits  = [x["profit"] for x in rescored]
        by_regime[name] = {
            "cost_per_km":     p["cost_per_km"],
            "lost_penalty":    p["lost_penalty"],
            "mean_profit":     float(np.mean(profits)),
            "std_profit":      float(np.std(profits)),
            "mean_truck_cost": float(np.mean([x["truck_cost"] for x in rescored])),
            "mean_lost_cost":  float(np.mean([x["lost_cost"]  for x in rescored])),
            "per_episode_profit": profits,
        }
    base["by_regime"] = by_regime
    return base


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    models = discover_models()
    print(f"Discovered {len(models)} trained models")
    for m in models:
        print(f"  - {m['kind']:<14}  {m['label']}")

    if not models:
        print("No trained models found. Did you run training first?")
        sys.exit(1)

    # We run each model ONCE (the trajectory — truck distance, lost trips,
    # revenue — depends only on the policy, NOT on the cost/penalty parameters
    # used in the profit formula). Then we re-score the trajectory under all
    # three regimes analytically.
    SCORE_AT = REGIMES["moderate_cost"]   # arbitrary; only affects the unused profit field in episodes_raw

    out = {"models": [], "regimes": REGIMES, "config": {
        "num_episodes":   NUM_EPISODES,
        "trip_base_fare": TRIP_BASE_FARE,
        "trip_per_km":    TRIP_PER_KM,
    }}

    for i, m in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {m['label']}  ({m['model_path']})")
        if m["kind"] == "static_baseline":
            episodes = evaluate_static_baseline_with_profit(
                num_episodes   = NUM_EPISODES,
                cost_per_km    = SCORE_AT["cost_per_km"],
                lost_penalty   = SCORE_AT["lost_penalty"],
                trip_base_fare = TRIP_BASE_FARE,
                trip_per_km    = TRIP_PER_KM,
            )
        else:
            episodes = evaluate_policy_with_profit(
                model_path        = m["model_path"],
                output_activation = m["output_activation"],
                fill_levels       = m["fill_levels"],
                num_episodes      = NUM_EPISODES,
                cost_per_km       = SCORE_AT["cost_per_km"],
                lost_penalty      = SCORE_AT["lost_penalty"],
                trip_base_fare    = TRIP_BASE_FARE,
                trip_per_km       = TRIP_PER_KM,
            )
        agg = aggregate(episodes, REGIMES)
        out["models"].append({
            **m,
            "aggregated":  agg,
            "episodes_raw": episodes,
        })

        s = agg
        print(
            f"    lost-rate={s['mean_lost_rate']:5.2f}%   "
            f"truck-km={s['mean_truck_km']:5.1f}   "
            f"profit[svc]=${s['by_regime']['service']['mean_profit']:7.2f}   "
            f"profit[mod]=${s['by_regime']['moderate_cost']['mean_profit']:7.2f}   "
            f"profit[cst]=${s['by_regime']['cost']['mean_profit']:7.2f}"
        )

    # --- save full JSON ----------------------------------------------------
    out_path = OUT_DIR / "thesis_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull JSON written to {out_path}")

    # --- save flat summary CSV --------------------------------------------
    csv_path = OUT_DIR / "thesis_summary.csv"
    with open(csv_path, "w") as f:
        f.write(
            "kind,regime_tag,activation,fill,mean_lost_rate,mean_truck_km,mean_actions,"
            "profit_service_mean,profit_service_std,"
            "profit_moderate_cost_mean,profit_moderate_cost_std,"
            "profit_cost_mean,profit_cost_std\n"
        )
        for m in out["models"]:
            agg = m["aggregated"]
            f.write(
                f"{m['kind']},{m['regime_tag'] or ''},{m['act_str']},{m['fill_str']},"
                f"{agg['mean_lost_rate']:.4f},{agg['mean_truck_km']:.4f},{agg['mean_actions']:.2f},"
                f"{agg['by_regime']['service']['mean_profit']:.4f},{agg['by_regime']['service']['std_profit']:.4f},"
                f"{agg['by_regime']['moderate_cost']['mean_profit']:.4f},{agg['by_regime']['moderate_cost']['std_profit']:.4f},"
                f"{agg['by_regime']['cost']['mean_profit']:.4f},{agg['by_regime']['cost']['std_profit']:.4f}\n"
            )
    print(f"Flat summary CSV written to {csv_path}")


if __name__ == "__main__":
    main()
