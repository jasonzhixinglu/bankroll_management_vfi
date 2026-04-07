#!/usr/bin/env python3
"""
scripts/generate_notebooks.py

Programmatically generate the two analysis notebooks using nbformat.
Run once; the resulting .ipynb files are committed alongside the source.

  python scripts/generate_notebooks.py
"""

from pathlib import Path
import nbformat as nbf

NB_DIR = Path(__file__).parent.parent / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def new_nb(cells) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11.0"}
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# Shared preamble code (imports + style)
# ─────────────────────────────────────────────────────────────────────────────

PREAMBLE = """\
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

FIG_DIR = Path("../quarto/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── clean matplotlib style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 130,
    "figure.figsize": (9, 4),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
})
"""

# ─────────────────────────────────────────────────────────────────────────────
# notebook 01 — Finite-horizon DP
# ─────────────────────────────────────────────────────────────────────────────

NB01_PARAMS = """\
from brm import OptimalBRM, calc_eta

# ── model parameters ────────────────────────────────────────────────────────
B     = [0,   200, 500]   # buyin sizes ($)
mu    = [7,    12,  20]   # mean hourly gain ($/hr)
sigma = [0,    80, 250]   # hourly std dev  ($/hr)
T     = 2000              # number of periods
xmax  = 1e6               # maximum bankroll on the grid ($)
xmin  = 1                 # minimum bankroll on the grid ($)
gridsize = 100            # number of grid points

GAME_LABELS = ["work ($7/hr)", "1/2 ($12/hr)", "2/5 ($20/hr)"]
T_PLOT = [0, 1950, 1999]  # periods to highlight in plots
"""

NB01_SOLVE = """\
model = OptimalBRM(B, mu, sigma, T=T)
model.solve(xmax=xmax, gridsize=gridsize, xmin=xmin)

xgrid = model.xgrid
print(f"Grid: {gridsize} points from ${xgrid[0]:,.0f} to ${xgrid[-1]:,.0f}")
print(f"Value function shape: {model.Vt.shape}  (T x gridsize)")
print(f"Policy shape:         {model.ft.shape}")
"""

NB01_POLICY = """\
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

fig, ax = plt.subplots()
for t, c in zip(T_PLOT, COLORS):
    ax.step(xgrid, model.ft[t, :], where="post", color=c, lw=1.8, label=f"t = {t}")

ax.set_xscale("log")
ax.set_xlabel("Bankroll ($)")
ax.set_ylabel("Optimal game")
ax.set_title("Optimal policy $f_t(x)$")
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(GAME_LABELS)
ax.legend(title="Period")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
fig.tight_layout()
fig.savefig(FIG_DIR / "finite_dp_policy.png", bbox_inches="tight")
plt.show()
"""

NB01_VF = """\
fig, ax = plt.subplots()
for t, c in zip(T_PLOT, COLORS):
    ax.plot(xgrid, model.Vt[t, :], color=c, lw=1.8, label=f"t = {t}")
# risk-neutral benchmark: V(x) = x
ax.plot(xgrid, xgrid, "--", color="gray", lw=1.2, label="V = x  (risk neutral)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Bankroll ($)")
ax.set_ylabel("$V_t(x)$")
ax.set_title("Value function $V_t(x)$")
ax.legend(title="Period")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
fig.tight_layout()
fig.savefig(FIG_DIR / "finite_dp_vf.png", bbox_inches="tight")
plt.show()
"""

NB01_ETA = """\
eta = calc_eta(xgrid, model.Vt[0, :])

# Trim boundary artefacts (fsolve is noisy near the grid edges)
trim = slice(8, -4)
x_plot   = xgrid[trim]
eta_plot = eta[trim]

fig, ax = plt.subplots()
ax.plot(x_plot, eta_plot, color="#1f77b4", lw=1.8, label="Estimated $\\\\eta(x)$")
ax.axhline(1, ls="--", color="#ff7f0e", lw=1.2, label="$\\\\eta = 1$  (log utility)")
ax.axhline(0, ls="--", color="gray",    lw=1.2, label="$\\\\eta = 0$  (risk neutral)")

ax.set_xscale("log")
ax.set_xlabel("Bankroll ($)")
ax.set_ylabel("$\\\\eta$  (CRRA coefficient)")
ax.set_title("Implied coefficient of relative risk aversion at $t = 0$")
ax.legend()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
fig.tight_layout()
fig.savefig(FIG_DIR / "finite_dp_eta.png", bbox_inches="tight")
plt.show()

# ── spot-check table ────────────────────────────────────────────────────────
print(f"{'Bankroll':>14}    eta")
print("-" * 24)
for x_tgt in [1_000, 5_000, 20_000, 100_000, 1_000_000]:
    idx = int(np.argmin(np.abs(xgrid - x_tgt)))
    print(f"  ${x_tgt:>11,.0f}    {eta[idx]:.3f}")
"""

cells_01 = [
    md("# Finite-Horizon Dynamic Programming\n\n"
       "Backward induction over $T=2000$ periods. Each period the player "
       "chooses among work, the 1/2 game, and the 2/5 game. "
       "The terminal value is $V_T(x) = x$ (risk-neutral), and the model is "
       "solved backwards to $V_0$.\n\n"
       "**Key question:** At what bankroll should we move up from 1/2 to 2/5?"),

    md("## Imports and style"),
    code(PREAMBLE),

    md("## Parameters"),
    code(NB01_PARAMS),

    md("## Solve the model"),
    code(NB01_SOLVE),

    md("## Optimal policy $f_t(x)$\n\n"
       "The optimal game as a function of bankroll for three representative "
       "periods. As $t \\to T$ the player becomes increasingly risk-neutral "
       "because the horizon is short and optionality has little time to compound."),
    code(NB01_POLICY),

    md("## Value function $V_t(x)$\n\n"
       "On a log–log scale the value function curves upward relative to "
       "the risk-neutral benchmark $V=x$, reflecting the option value "
       "of moving up in stakes."),
    code(NB01_VF),

    md("## Implied CRRA coefficient $\\eta(x)$\n\n"
       "Recovered by inverting the local ratio of CRRA utilities at "
       "consecutive grid points. Values above the $\\eta=1$ line indicate "
       "greater risk aversion than log utility; the coefficient "
       "falls toward zero as the bankroll grows and optionality saturates."),
    code(NB01_ETA),
]

nb01 = new_nb(cells_01)
path_01 = NB_DIR / "01_finite_dp.ipynb"
with open(path_01, "w") as f:
    nbf.write(nb01, f)
print(f"Written: {path_01}")


# ─────────────────────────────────────────────────────────────────────────────
# notebook 02 — OBM vs Kelly
# ─────────────────────────────────────────────────────────────────────────────

NB02_PARAMS = """\
from brm import (
    gen_xgrid,
    build_sampling_df,
    kelly_cew,
    solve_obm,
    extract_threshold,
)

# ── game parameters  (buyin, winrate, std) ──────────────────────────────────
game_choices = {
    0: (0,    5,   0),   # outside option: work ($5/hr, no variance)
    1: (100,  10, 100),  # 1/2 NL
    2: (200,  15, 200),  # 2/5 NL
    3: (500,  20, 500),  # 5/10 NL
}

GAME_LABELS = {
    0: "work  ($5/hr)",
    1: "1/2   ($10/hr)",
    2: "2/5   ($15/hr)",
    3: "5/10  ($20/hr)",
}
GAME_COLORS = {0: "#888888", 1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}

# ── grid ────────────────────────────────────────────────────────────────────
gridsize = 100
xmax     = 500_000
xmin     = game_choices[1][0]   # smallest buyin = 100
N        = len(game_choices)

xgrid = gen_xgrid(xmax, xmin=xmin, gridsize=gridsize)
print(f"Grid: {gridsize} points from ${xgrid[0]:,.0f} to ${xgrid[-1]:,.0f}")
"""

NB02_SOLVE = """\
# ── Kelly benchmark (closed-form, one-shot) ─────────────────────────────────
sampling_df         = build_sampling_df(game_choices, xgrid)
CEW_choices_kelly, CEW_kelly = kelly_cew(sampling_df, xgrid)

# ── OBM value iteration ─────────────────────────────────────────────────────
CEW_array, CEW_choices_obm, vf = solve_obm(game_choices, xgrid, tol=1e-2)

n_iter = CEW_array.shape[1] - 1   # column 0 = Kelly init; columns 1..k = OBM
print(f"OBM converged in {n_iter} iterations")

# ── Convergence distance from successive CEW columns ────────────────────────
# CEW_array[:,k] = max-over-games CEW at iteration k
# Convergence distance ≈ max|CEW_k - CEW_{k-1}| / max(CEW_{k-1})
conv_dists = np.array([
    np.max(np.abs(CEW_array[:, k] - CEW_array[:, k - 1])) / np.max(CEW_array[:, k - 1])
    for k in range(1, CEW_array.shape[1])
])

# ── Thresholds ───────────────────────────────────────────────────────────────
thresh_kelly = {n: extract_threshold(n, CEW_choices_kelly, xgrid, game_choices)
                for n in range(1, N)}
thresh_obm   = {n: extract_threshold(n, CEW_choices_obm,   xgrid, game_choices)
                for n in range(1, N)}

print()
print(f"{'Transition':<16}  {'Kelly':>10}  {'OBM':>10}  {'Δ (OBM−Kelly)':>14}")
print("-" * 56)
for n in range(1, N):
    delta = thresh_obm[n] - thresh_kelly[n]
    print(f"  game {n-1} → game {n}     ${thresh_kelly[n]:>8,.0f}  ${thresh_obm[n]:>8,.0f}  ${delta:>+12,.0f}")
"""

NB02_CONVERGENCE = """\
# Skip the very first distance (Kelly → OBM iter 1) which is always large
iters_plot = np.arange(2, len(conv_dists) + 1)
dists_plot = conv_dists[1:]

fig, ax = plt.subplots()
ax.plot(iters_plot, dists_plot, color="#1f77b4", lw=1.8)
ax.axhline(0.01, ls="--", color="gray", lw=1.2, label="tol = 0.01")
ax.set_yscale("log")
ax.set_xlabel("Iteration")
ax.set_ylabel("CEW distance  (log scale)")
ax.set_title("OBM value iteration — convergence")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "obm_convergence.png", bbox_inches="tight")
plt.show()
"""

NB02_OBM_CEW = """\
def _cew_plot(CEW_choices, thresholds, game_choices, title, fname):
    \"\"\"Plot CEW curves by game with threshold verticals and true-EV horizontals.\"\"\"
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for n in range(N):
        sub = CEW_choices[CEW_choices["n"] == n].sort_values("x")
        ax.plot(sub["x"], sub["CEW"],
                color=GAME_COLORS[n], lw=1.8, label=GAME_LABELS[n])

    # true EV horizontals (dotted)
    for n in range(1, N):
        ev = game_choices[n][1]
        ax.axhline(ev, ls=":", color=GAME_COLORS[n], lw=1.0, alpha=0.7)

    # threshold verticals (dashed gray)
    for n, thresh in thresholds.items():
        ax.axvline(thresh, ls="--", color="gray", lw=0.9)
        ax.text(thresh * 1.04, ax.get_ylim()[1] * 0.97,
                f"${thresh:,.0f}", fontsize=8, color="gray", va="top")

    ax.set_xscale("log")
    _, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Bankroll ($)")
    ax.set_ylabel("CEW  ($/hr)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.show()


_cew_plot(
    CEW_choices_obm, thresh_obm, game_choices,
    title="OBM Certainty-Equivalent Winrate by game",
    fname="obm_cew.png",
)
"""

NB02_KELLY_CEW = """\
_cew_plot(
    CEW_choices_kelly, thresh_kelly, game_choices,
    title="Kelly Certainty-Equivalent Winrate by game",
    fname="kelly_cew.png",
)
"""

NB02_THRESHOLDS = """\
import matplotlib.patches as mpatches

transitions = [f"game {n-1}→{n}" for n in range(1, N)]
x_pos = np.arange(len(transitions))
bar_w = 0.35

kelly_vals = [thresh_kelly[n] for n in range(1, N)]
obm_vals   = [thresh_obm[n]   for n in range(1, N)]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x_pos - bar_w / 2, kelly_vals, bar_w, color="#1f77b4", alpha=0.85, label="Kelly")
ax.bar(x_pos + bar_w / 2, obm_vals,   bar_w, color="#ff7f0e", alpha=0.85, label="OBM")

# annotate bar tops
for i, (k, o) in enumerate(zip(kelly_vals, obm_vals)):
    ax.text(i - bar_w / 2, k * 1.02, f"${k:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + bar_w / 2, o * 1.02, f"${o:,.0f}", ha="center", va="bottom", fontsize=8)

ax.set_yscale("log")
ax.set_xticks(x_pos)
ax.set_xticklabels(transitions)
ax.set_ylabel("Bankroll threshold ($)  [log scale]")
ax.set_title("Move-up thresholds: OBM vs Kelly")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
fig.tight_layout()
fig.savefig(FIG_DIR / "obm_vs_kelly_thresholds.png", bbox_inches="tight")
plt.show()

# ── summary table ────────────────────────────────────────────────────────────
print(f"\\n{'Transition':<16}  {'Kelly':>10}  {'OBM':>10}  {'OBM/Kelly':>10}")
print("-" * 52)
for n in range(1, N):
    ratio = thresh_obm[n] / thresh_kelly[n]
    print(f"  game {n-1} → game {n}     ${thresh_kelly[n]:>8,.0f}  ${thresh_obm[n]:>8,.0f}  {ratio:>9.2f}x")
"""

cells_02 = [
    md("# OBM vs Kelly Bankroll Management\n\n"
       "Value iteration converges to a stationary Certainty-Equivalent Winrate "
       "(CEW) for each bankroll level. The CEW is the per-period dollar amount "
       "that makes the player indifferent between accepting it with certainty "
       "and playing optimally.\n\n"
       "The **Kelly benchmark** (log utility, no iteration) is compared against "
       "the fully optimal **OBM** solution."),

    md("## Imports and style"),
    code(PREAMBLE),

    md("## Parameters"),
    code(NB02_PARAMS),

    md("## Solve: OBM and Kelly"),
    code(NB02_SOLVE),

    md("## Convergence\n\n"
       "Convergence is measured as the max absolute change in CEW "
       "across grid points, normalised by the current max CEW. "
       "The plot starts from iteration 2 (iteration 1 is the large "
       "jump from the Kelly initialisation to the first OBM update)."),
    code(NB02_CONVERGENCE),

    md("## OBM Certainty-Equivalent Winrate\n\n"
       "Curves show the per-hour CEW for each game as a function of bankroll. "
       "Dotted horizontals mark the true EV for each game. "
       "Dashed verticals mark the move-up thresholds. "
       "The optimal game at any bankroll is the one with the highest CEW."),
    code(NB02_OBM_CEW),

    md("## Kelly Certainty-Equivalent Winrate\n\n"
       "Same layout as above, but using one-shot Kelly (log utility). "
       "Kelly thresholds are systematically higher than OBM thresholds "
       "because Kelly does not account for the compounding optionality "
       "of moving up in stakes."),
    code(NB02_KELLY_CEW),

    md("## Threshold comparison: OBM vs Kelly\n\n"
       "OBM thresholds are strictly lower than Kelly thresholds — the model "
       "recommends moving up sooner, because the option value of the higher "
       "game lowers the effective risk of under-rolling."),
    code(NB02_THRESHOLDS),
]

nb02 = new_nb(cells_02)
path_02 = NB_DIR / "02_obm_vs_kelly.ipynb"
with open(path_02, "w") as f:
    nbf.write(nb02, f)
print(f"Written: {path_02}")
