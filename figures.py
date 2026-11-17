"""Hero chart for the repo README. Reads committed results only."""
# --- shared chart style (identical in every repo's figures.py) ---------------------------
# Palette: first three categorical slots of the validated reference palette (pass all-pairs
# CVD checks in light mode). Ink and grid are text tokens, never series colours.
import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "regular", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "axes.labelcolor": INK2, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
})
FIGSIZE = (7.2, 4.0)   # one hero size for all six projects


def finish(fig, ax, title, subtitle, path):
    """Title (what the chart says) and subtitle (what is plotted, from which file), then save."""
    ax.set_title(title, color=INK, pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
# -------------------------------------------------------------------------------------------


import numpy as np
import pandas as pd


def hero(path="figures/hero.png"):
    d = pd.read_csv("studies/varx_permutation_study/results/spec_relrmse_gdp_h1.csv")
    r = d.rel_rmse_core_var
    better = d.dm_p_one_sided < 0.05          # significantly better than the core VAR
    below = (r < 1).mean()
    bins = np.arange(0.80, 1.92, 0.04)
    n, edges = np.histogram(r[~better], bins=bins)
    ns, _ = np.histogram(r[better], bins=bins)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    w = np.diff(edges)
    ax.bar(edges[:-1], n, width=w, align="edge", color=BLUE, edgecolor=SURFACE, linewidth=1.0)
    ax.bar(edges[:-1], ns, bottom=n, width=w, align="edge", color=ORANGE, edgecolor=SURFACE, linewidth=1.0)
    ax.axvline(1.0, color=INK, lw=1.4)
    ax.set_ylim(0, n.max() * 1.15)
    ax.text(1.02, n.max() * 1.12, "core VAR (1.0)", color=INK, fontsize=9, va="top")
    ax.text(1.25, n.max() * 0.62,
            f"{below:.0%} of {len(r)} specs are below 1.0\n{better.sum()} of {len(r)} are significantly\n"
            f"better (Diebold-Mariano, 5%)", color=INK, fontsize=10, va="top")
    ax.set_xlabel("RMSE relative to the core VAR (below 1.0 is better)")
    ax.set_ylabel("Number of specifications")
    finish(fig, ax, f"{below:.0%} of specifications beat the core VAR for GDP one quarter ahead, none significantly",
           "Relative RMSE, valid specs, BIC lags, GDP h=1, one-sided DM test (spec_relrmse_gdp_h1.csv)", path)


if __name__ == "__main__":
    hero()
