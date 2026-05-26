"""
Tabular summary of results_thesis/thesis_results.json.

Reads the unified evaluator's output and prints three tables:
  TABLE 1 — Mean profit per regime for every trained policy
  TABLE 2 — Lost-demand reproduction (base-paper) summary
  TABLE 3 — Tier-A regime-specific profit-DQN advantage
"""

import json
from pathlib import Path


RESULTS = Path(__file__).parent / "results_thesis" / "thesis_results.json"


def profit(m, regime):
    return m["aggregated"]["by_regime"][regime]["mean_profit"]


def fmt_profit(m, regime):
    return f"${profit(m, regime):7.2f}"


def main():
    with open(RESULTS) as f:
        d = json.load(f)

    static = next(x for x in d["models"] if x["kind"] == "static_baseline")
    lds    = sorted(
        [x for x in d["models"] if x["kind"] == "lost_demand"],
        key=lambda x: x["aggregated"]["mean_lost_rate"],
    )
    prs    = sorted(
        [x for x in d["models"] if x["kind"] == "profit"],
        key=lambda x: -profit(x, "cost"),
    )
    prrs   = [x for x in d["models"] if x["kind"] == "profit_regime"]

    # ----------------- TABLE 1 -----------------
    print("=" * 116)
    print("TABLE 1 — Mean metrics over 50 test episodes")
    print("=" * 116)
    header = (
        f"{'policy':<22}{'act':<13}{'fill':<13}"
        f"{'lost%':>8}{'truck km':>10}"
        f"{'svc ($0.5/$10)':>17}{'mod ($2/$3)':>14}{'cost ($5/$1)':>15}"
    )
    print(header)
    print("-" * 116)

    a = static["aggregated"]
    print(
        f"{'Static baseline':<22}{'n/a':<13}{'n/a':<13}"
        f"{a['mean_lost_rate']:>8.2f}{a['mean_truck_km']:>10.1f}"
        f"{fmt_profit(static, 'service'):>17}"
        f"{fmt_profit(static, 'moderate_cost'):>14}"
        f"{fmt_profit(static, 'cost'):>15}"
    )

    for m in lds:
        a = m["aggregated"]
        print(
            f"{'LD-DQN':<22}{m['act_str']:<13}{m['fill_str']:<13}"
            f"{a['mean_lost_rate']:>8.2f}{a['mean_truck_km']:>10.1f}"
            f"{fmt_profit(m, 'service'):>17}"
            f"{fmt_profit(m, 'moderate_cost'):>14}"
            f"{fmt_profit(m, 'cost'):>15}"
        )

    for m in prs:
        a = m["aggregated"]
        print(
            f"{'PR-DQN (default)':<22}{m['act_str']:<13}{m['fill_str']:<13}"
            f"{a['mean_lost_rate']:>8.2f}{a['mean_truck_km']:>10.1f}"
            f"{fmt_profit(m, 'service'):>17}"
            f"{fmt_profit(m, 'moderate_cost'):>14}"
            f"{fmt_profit(m, 'cost'):>15}"
        )

    for m in prrs:
        a = m["aggregated"]
        tag = m["regime_tag"]
        print(
            f"{f'PR-DQN [{tag}]':<22}{m['act_str']:<13}{m['fill_str']:<13}"
            f"{a['mean_lost_rate']:>8.2f}{a['mean_truck_km']:>10.1f}"
            f"{fmt_profit(m, 'service'):>17}"
            f"{fmt_profit(m, 'moderate_cost'):>14}"
            f"{fmt_profit(m, 'cost'):>15}"
        )

    # ----------------- TABLE 2 — base-paper reproduction -----------------
    print()
    print("=" * 80)
    print("TABLE 2 — Base-paper reproduction (LD-DQN vs static baseline)")
    print("=" * 80)
    s_rate = static["aggregated"]["mean_lost_rate"]
    print(f"Static baseline lost-demand rate: {s_rate:.2f}%")
    print()
    d_abs_label = "Δ abs"
    d_rel_label = "Δ rel"
    print(f"{'activation':<13}{'fill':<13}{'lost%':>8}{d_abs_label:>10}{d_rel_label:>10}")
    print("-" * 54)
    for m in lds:
        r = m["aggregated"]["mean_lost_rate"]
        d_abs = r - s_rate
        d_rel = (r - s_rate) / s_rate * 100
        sign = "✓" if d_abs < 0 else "✗"
        print(
            f"{m['act_str']:<13}{m['fill_str']:<13}{r:>8.2f}"
            f"{d_abs:>+9.2f}{d_rel:>+9.1f}%  {sign}"
        )

    # ----------------- TABLE 3 — regime-specific PR vs everything else -----------------
    print()
    print("=" * 90)
    print("TABLE 3 — Best policy under each economic regime")
    print("=" * 90)
    for regime in ["service", "moderate_cost", "cost"]:
        params = d["regimes"][regime]
        candidates = [static] + lds + prs + prrs
        best = max(candidates, key=lambda x: profit(x, regime))
        print()
        print(f"Regime: {regime}  (cost=${params['cost_per_km']}/km, penalty=${params['lost_penalty']}/trip)")
        print(f"  Static baseline             : {fmt_profit(static, regime)}")
        ld_best = max(lds, key=lambda x: profit(x, regime))
        print(f"  Best LD-DQN ({ld_best['act_str']}/{ld_best['fill_str']})   : {fmt_profit(ld_best, regime)}")
        pr_best = max(prs, key=lambda x: profit(x, regime))
        print(f"  Best PR-DQN ({pr_best['act_str']}/{pr_best['fill_str']})   : {fmt_profit(pr_best, regime)}")
        if prrs:
            matched = next((x for x in prrs if x["regime_tag"] == regime), None)
            if matched:
                print(f"  PR-DQN trained for this regime: {fmt_profit(matched, regime)}")
        print(f"  >>> WINNER overall: {best['kind']:<16} {best.get('act_str','')}/{best.get('fill_str','')}"
              f"  regime_tag={best.get('regime_tag') or '-'}  => {fmt_profit(best, regime)}")


if __name__ == "__main__":
    main()
