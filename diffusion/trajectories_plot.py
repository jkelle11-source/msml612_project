"""M7-3's trajectory figure, from ``results/trajectories/trajectories.npz`` (``diffusion.trajectories``).

    python -m diffusion.trajectories_plot          # writes results/trajectories/trajectories.png and .pdf

Needs only numpy and matplotlib, which this repo's environment does not install, so run it from the
repo root with any Python that has them (Python 3.10 or later). Five patients (rows) by HR, serum
sodium and GCS (columns), in clinical units, over the 48 hours:

- visible values (what the model was shown): filled black dots;
- held-out values (what the holdout scores): open black rings;
- the diffusion model: the median of its draws (solid blue), its 5-95% band (light blue) and
  three single draws (thin blue);
- forward fill (orange steps) and mean imputation (dashed aqua), the two baselines.

Each method has its own line style as well as its own colour, so none is told apart by colour alone.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parents[1] / "results" / "trajectories"

# Categorical slots 1-3 of the dataviz reference palette, checked for colour-blind separation.
DIFFUSION, FORWARD_FILL, MEAN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#5f5e59", "#e6e5e0"
TITLES = {"HR": "Heart rate (bpm)", "Na": "Serum sodium (mmol/L)", "GCS": "Glasgow Coma Score"}
SHOWN_DRAWS = 3

def figure(data, record=None):
    """The 5 x 3 figure for ``data`` (the arrays ``trajectories.npz`` holds)."""
    hours, variables = data["hours"], [str(v) for v in data["variables"]]
    n_patients = len(data["record_ids"])
    low, high = np.quantile(data["draws"], [0.05, 0.95], axis=0)
    fig, axes = plt.subplots(n_patients, len(variables), figsize=(12, 2.3 * n_patients), sharex=True,
                             constrained_layout=True)
    for i in range(n_patients):
        for j, variable in enumerate(variables):
            ax = axes[i, j]
            ax.fill_between(hours, low[i, :, j], high[i, :, j], color=DIFFUSION, alpha=0.16, linewidth=0)
            for k in range(SHOWN_DRAWS):
                ax.plot(hours, data["draws"][k, i, :, j], color=DIFFUSION, linewidth=0.7, alpha=0.55)
            ax.plot(hours, data["median"][i, :, j], color=DIFFUSION, linewidth=2)
            ax.plot(hours, data["forward_fill"][i, :, j], color=FORWARD_FILL, linewidth=1.6, drawstyle="steps-post")
            ax.plot(hours, data["mean_imputation"][i, :, j], color=MEAN, linewidth=1.6, linestyle=(0, (5, 3)))
            visible, held_out = data["visible"][i, :, j], data["held_out"][i, :, j]
            ax.plot(hours[visible], data["truth"][i, visible, j], "o", color=INK, markersize=3.5, zorder=5)
            ax.plot(hours[held_out], data["truth"][i, held_out, j], "o", markerfacecolor="white",
                    markeredgecolor=INK, markeredgewidth=1.6, markersize=8, zorder=6)
            if variable == "GCS":
                ax.set_ylim(2.3, 15.7)
                ax.set_yticks([3, 6, 9, 12, 15])
            ax.set_xlim(hours[0], hours[-1])
            ax.set_xticks(range(0, int(hours[-1]) + 1, 12))
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
            if i == 0:
                ax.set_title(TITLES.get(variable, variable), color=INK, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"Patient {data['record_ids'][i]}", color=INK, fontsize=9)
            if i == n_patients - 1:
                ax.set_xlabel("Hour since admission", color=MUTED, fontsize=9)
    legend = [
        Line2D([], [], marker="o", linestyle="", color=INK, markersize=4, label="Visible to the model"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=INK,
               markeredgewidth=1.6, markersize=8, label="Held out (scored)"),
        Line2D([], [], color=DIFFUSION, linewidth=2, label="Diffusion: median of draws"),
        Patch(facecolor=DIFFUSION, alpha=0.16, label="Diffusion: 5-95% of draws"),
        Line2D([], [], color=DIFFUSION, linewidth=0.7, alpha=0.55, label=f"Diffusion: {SHOWN_DRAWS} single draws"),
        Line2D([], [], color=FORWARD_FILL, linewidth=1.6, drawstyle="steps-post", label="Forward fill"),
        Line2D([], [], color=MEAN, linewidth=1.6, linestyle=(0, (5, 3)), label="Mean imputation"),
    ]
    fig.legend(handles=legend, loc="outside upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    if record is not None:
        fig.suptitle(f"{record['variant']} (seed {record['seed']}, {record['draws']} draws), "
                     f"test split, 10% random holdout", color=MUTED, fontsize=10, y=-0.005, va="top")
    return fig

def main(argv=None):
    parser = argparse.ArgumentParser(description="M7-3's trajectory figure.")
    parser.add_argument("--data", type=Path, default=OUT / "trajectories.npz")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    data = dict(np.load(args.data))
    record_path = args.data.with_suffix(".json")
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    fig = figure(data, record)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out / f"trajectories.{suffix}", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out / 'trajectories.png'} and .pdf")

if __name__ == "__main__":
    main()
