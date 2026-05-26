"""
Generate publication-quality figures for the thesis (Chapter 6).

Inputs : results_thesis/thesis_results.json
Outputs: final_report/Chapters/Chapter_6/figures/
            fig_lost_demand_reproduction.png
            fig_lost_demand_heatmap.png
            fig_profit_by_regime.png
            fig_truck_vs_lost_scatter.png

The 'static do-nothing baseline' appears only in the lost-demand chart
(base-paper reproduction). It is omitted from all profit comparisons per
explicit user instruction.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


ROOT       = Path(__file__).resolve().parent
RESULTS    = ROOT / "results_thesis" / "thesis_results.json"
OUT_DIR    = ROOT.parent / "final_report" / "Chapters" / "Chapter_6" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- visual style --
plt.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "figure.dpi":       120,
    "savefig.dpi":      240,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
})

FILL_COLORS = {
    "15_50_85":  "#1f77b4",   # blue — chapter best
    "10_50_90":  "#ff7f0e",   # orange — base-paper default
    "0_50_100":  "#d62728",   # red — pathological
}
FILL_LABELS = {
    "15_50_85":  "fill 15-50-85",
    "10_50_90":  "fill 10-50-90",
    "0_50_100":  "fill 0-50-100",
}
ACT_ORDER  = ["none", "elu", "leaky_relu", "prelu"]
FILL_ORDER = ["15_50_85", "10_50_90", "0_50_100"]


def load():
    with open(RESULTS) as f:
        return json.load(f)


def get(d, kind):
    return [m for m in d["models"] if m["kind"] == kind]


# ---------------------------------------------------------------------------
# FIGURE 1 — Base-paper reproduction: lost-demand rate, static + LD-DQN matrix
# ---------------------------------------------------------------------------
def fig_lost_demand_reproduction(d):
    static = next(x for x in d["models"] if x["kind"] == "static_baseline")
    lds    = get(d, "lost_demand")
    s_rate = static["aggregated"]["mean_lost_rate"]
    s_std  = static["aggregated"]["std_lost_rate"]

    # 12 LD-DQN, sorted by lost rate (worst to best for visual flow → best on right)
    lds_sorted = sorted(lds, key=lambda x: -x["aggregated"]["mean_lost_rate"])

    fig, ax = plt.subplots(figsize=(12, 5.3))

    labels = []
    rates  = []
    stds   = []
    colors = []
    for m in lds_sorted:
        labels.append(f"{m['act_str']}\n{m['fill_str'].replace('_','-')}")
        rates.append(m["aggregated"]["mean_lost_rate"])
        stds.append(m["aggregated"]["std_lost_rate"])
        colors.append(FILL_COLORS[m["fill_str"]])

    x = np.arange(len(labels))
    bars = ax.bar(x, rates, yerr=stds, capsize=3, color=colors,
                  edgecolor="black", linewidth=0.6, alpha=0.92)

    # Static baseline as horizontal band (mean ± std)
    ax.axhline(s_rate, color="black", linewidth=1.8, linestyle="--",
               label=f"Static baseline = {s_rate:.2f}%")
    ax.axhspan(s_rate - s_std, s_rate + s_std, color="black", alpha=0.08)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8.5)
    ax.set_ylabel("Mean lost-demand rate over 50 test episodes (%)")
    ax.set_title("Lost-demand DQN vs static baseline — GT0 (10 stations, 2 vehicles)")
    ax.set_ylim(0, max(max(rates), s_rate) * 1.18)

    # Legend: fill levels + static
    handles = [mpatches.Patch(color=FILL_COLORS[f], label=FILL_LABELS[f])
               for f in FILL_ORDER]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                              linewidth=1.8, label=f"Static = {s_rate:.2f}%"))
    ax.legend(handles=handles, loc="upper right", framealpha=0.95)

    plt.savefig(OUT_DIR / "fig_lost_demand_reproduction.png")
    plt.close()
    print(f"  saved {OUT_DIR / 'fig_lost_demand_reproduction.png'}")


# ---------------------------------------------------------------------------
# FIGURE 2 — Lost-demand heatmap: activation × fill
# ---------------------------------------------------------------------------
def fig_lost_demand_heatmap(d):
    lds = get(d, "lost_demand")
    grid = np.full((len(ACT_ORDER), len(FILL_ORDER)), np.nan)
    for m in lds:
        i = ACT_ORDER.index(m["act_str"])
        j = FILL_ORDER.index(m["fill_str"])
        grid[i, j] = m["aggregated"]["mean_lost_rate"]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    vmax = max(grid.max(), 14.0)
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=vmax, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i,j]:.2f}%",
                    ha="center", va="center",
                    color="white" if grid[i, j] > vmax * 0.55 else "black",
                    fontweight="bold", fontsize=11)
    ax.set_xticks(range(len(FILL_ORDER)))
    ax.set_xticklabels([f.replace("_", "-") for f in FILL_ORDER])
    ax.set_yticks(range(len(ACT_ORDER)))
    ax.set_yticklabels(ACT_ORDER)
    ax.set_xlabel("Fill-level configuration")
    ax.set_ylabel("Output activation")
    ax.set_title("LD-DQN lost-demand rate (%): activation × fill")
    cbar = plt.colorbar(im, ax=ax, label="Lost-demand rate (%)")
    plt.savefig(OUT_DIR / "fig_lost_demand_heatmap.png")
    plt.close()
    print(f"  saved {OUT_DIR / 'fig_lost_demand_heatmap.png'}")


# ---------------------------------------------------------------------------
# FIGURE 3 — Profit by regime (no static baseline per user request)
#    LD-DQN(best)  vs  PR-DQN-default(best)  vs  PR-DQN-regime-specific
# ---------------------------------------------------------------------------
def fig_profit_by_regime(d):
    lds = get(d, "lost_demand")
    prs = get(d, "profit")
    prrs = get(d, "profit_regime")

    regimes = ["service", "moderate_cost", "cost"]
    regime_labels = [
        f"Service\n($0.50/km, $10/trip)",
        f"Moderate\n($2/km, $3/trip)",
        f"Cost-priority\n($5/km, $1/trip)",
    ]

    # For each regime, find best LD-DQN, best PR-DQN(default), and matched PR-DQN-regime
    rows = {"LD-DQN (best)": [], "PR-DQN (default-trained, best)": [],
            "PR-DQN (regime-trained)": []}
    rows_std = {k: [] for k in rows}

    for r in regimes:
        best_ld = max(lds, key=lambda m: m["aggregated"]["by_regime"][r]["mean_profit"])
        best_pr = max(prs, key=lambda m: m["aggregated"]["by_regime"][r]["mean_profit"])
        matched_prr = next((m for m in prrs if m["regime_tag"] == r), None)

        rows["LD-DQN (best)"].append(best_ld["aggregated"]["by_regime"][r]["mean_profit"])
        rows_std["LD-DQN (best)"].append(best_ld["aggregated"]["by_regime"][r]["std_profit"])

        rows["PR-DQN (default-trained, best)"].append(
            best_pr["aggregated"]["by_regime"][r]["mean_profit"])
        rows_std["PR-DQN (default-trained, best)"].append(
            best_pr["aggregated"]["by_regime"][r]["std_profit"])

        if matched_prr is not None:
            rows["PR-DQN (regime-trained)"].append(
                matched_prr["aggregated"]["by_regime"][r]["mean_profit"])
            rows_std["PR-DQN (regime-trained)"].append(
                matched_prr["aggregated"]["by_regime"][r]["std_profit"])
        else:
            rows["PR-DQN (regime-trained)"].append(np.nan)
            rows_std["PR-DQN (regime-trained)"].append(0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(regimes))
    width = 0.26
    colors = {"LD-DQN (best)": "#ff7f0e",
              "PR-DQN (default-trained, best)": "#2ca02c",
              "PR-DQN (regime-trained)": "#1f77b4"}

    for i, key in enumerate(rows):
        offset = (i - 1) * width
        vals = rows[key]
        errs = rows_std[key]
        bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=4,
                      label=key, color=colors[key],
                      edgecolor="black", linewidth=0.6, alpha=0.92)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            yoff = 5 if v >= 0 else -14
            ax.text(b.get_x() + b.get_width() / 2, v + yoff,
                    f"${v:.0f}", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels)
    ax.set_ylabel("Mean profit per episode (USD)")
    ax.set_title("Mean profit across economic regimes — LD-DQN vs Profit-DQN")
    ax.legend(loc="lower left", framealpha=0.95)
    plt.savefig(OUT_DIR / "fig_profit_by_regime.png")
    plt.close()
    print(f"  saved {OUT_DIR / 'fig_profit_by_regime.png'}")


# ---------------------------------------------------------------------------
# FIGURE 4 — Truck-km vs lost-demand scatter (policy diversity)
# ---------------------------------------------------------------------------
def fig_truck_vs_lost_scatter(d):
    lds = get(d, "lost_demand")
    prs = get(d, "profit")
    prrs = get(d, "profit_regime")
    static = next(x for x in d["models"] if x["kind"] == "static_baseline")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Static baseline
    ax.scatter(0, static["aggregated"]["mean_lost_rate"],
               s=180, marker="X", color="black", label="Static baseline",
               zorder=5)

    # LD-DQN
    for m in lds:
        a = m["aggregated"]
        ax.scatter(a["mean_truck_km"], a["mean_lost_rate"],
                   color=FILL_COLORS[m["fill_str"]], marker="o",
                   s=70, edgecolor="black", linewidth=0.5, alpha=0.85)

    # PR-DQN default
    for m in prs:
        a = m["aggregated"]
        ax.scatter(a["mean_truck_km"], a["mean_lost_rate"],
                   color=FILL_COLORS[m["fill_str"]], marker="s",
                   s=70, edgecolor="black", linewidth=0.5, alpha=0.85)

    # Regime-specific PR-DQN — annotated
    for m in prrs:
        a = m["aggregated"]
        ax.scatter(a["mean_truck_km"], a["mean_lost_rate"],
                   color="#9467bd", marker="*", s=300,
                   edgecolor="black", linewidth=0.8, zorder=6)
        ax.annotate(f"PR-DQN[{m['regime_tag']}]",
                    xy=(a["mean_truck_km"], a["mean_lost_rate"]),
                    xytext=(8, 7), textcoords="offset points",
                    fontsize=9, fontweight="bold")

    ax.set_xlabel("Mean truck distance per episode (km)")
    ax.set_ylabel("Mean lost-demand rate (%)")
    ax.set_title("Policy diversity: truck use vs lost demand")

    handles = [
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="black",
                   markersize=11, label="Static (no rebalancing)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=8, label="LD-DQN"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=8, label="PR-DQN (default)"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#9467bd",
                   markeredgecolor="black", markersize=14, label="PR-DQN (regime-trained)"),
    ]
    handles.append(mpatches.Patch(color="none", label=""))
    for f in FILL_ORDER:
        handles.append(mpatches.Patch(color=FILL_COLORS[f], label=FILL_LABELS[f]))
    ax.legend(handles=handles, loc="upper right", framealpha=0.95, ncol=1)

    plt.savefig(OUT_DIR / "fig_truck_vs_lost_scatter.png")
    plt.close()
    print(f"  saved {OUT_DIR / 'fig_truck_vs_lost_scatter.png'}")


def main():
    d = load()
    print(f"Loaded {len(d['models'])} models from {RESULTS}")
    print(f"Writing figures to {OUT_DIR}")
    fig_lost_demand_reproduction(d)
    fig_lost_demand_heatmap(d)
    fig_profit_by_regime(d)
    fig_truck_vs_lost_scatter(d)
    print("Done.")


if __name__ == "__main__":
    main()
