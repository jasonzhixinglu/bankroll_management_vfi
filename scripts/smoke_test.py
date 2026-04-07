#!/usr/bin/env python3
"""
scripts/smoke_test.py

End-to-end smoke test for both BRM models using the baseline parameters
from the original notebooks. Prints a brief summary of results.
No plots are produced.

Exit code: 0 if both models pass, 1 if either fails.
"""

import sys
import time
import traceback

import numpy as np

from brm import (
    OptimalBRM,
    build_sampling_df,
    calc_eta,
    extract_threshold,
    gen_xgrid,
    kelly_cew,
    solve_obm,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

WIDTH = 62


def header(title: str) -> None:
    print(f"\n{'─' * WIDTH}")
    print(f"  {title}")
    print(f"{'─' * WIDTH}")


def ok(msg: str) -> None:
    print(f"  [PASS]  {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL]  {msg}")


def row(label: str, value: str, indent: int = 4) -> None:
    print(f"{' ' * indent}{label:<36}{value}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Finite-horizon DP
# ─────────────────────────────────────────────────────────────────────────────

header("Model 1 — Finite-horizon DP")

B     = np.array([0,  200, 500])
mu    = np.array([7,   12,  20])
sigma = np.array([0,   80, 250])
T     = 2000
xmax  = 1e6

print(f"  B = {B},  mu = {mu},  sigma = {sigma}")
print(f"  T = {T},  xmax = {xmax:.0e},  gridsize = 100")
print(f"  (Note: T = 2000 may take several minutes)")
print()

dp_ok = False
try:
    t0 = time.time()
    model = OptimalBRM(B, mu, sigma, T=T)
    model.solve(xmax=xmax, gridsize=100)
    elapsed = time.time() - t0
    ok(f"Solved in {elapsed:.1f}s")

    # ── 1a. Game-switch threshold at t=0 ────────────────────────────────────
    xgrid_dp = model.xgrid
    ft0      = model.ft[0, :]

    # Find where the optimal game first switches from ≤2 to >2 (move to 2/5)
    switch_idx = np.argmax(ft0 > 2)
    if switch_idx > 0:
        lo = xgrid_dp[switch_idx - 1]
        hi = xgrid_dp[switch_idx]
        row("Switch to 2/5 (t=0):", f"${lo:,.0f} – ${hi:,.0f}")
    else:
        row("Switch to 2/5 (t=0):", "not found in grid")

    # ── 1b. Implied CRRA at representative bankroll levels ──────────────────
    V0  = model.Vt[0, :]
    eta = calc_eta(xgrid_dp, V0)

    print()
    print("  Implied CRRA coefficient η(x) at t=0")
    print(f"    {'Bankroll':>12}    {'η':>8}    interpretation")
    print(f"    {'':->12}    {'':->8}    {'':->20}")

    checkpoints = [1_000, 10_000, 100_000, 1_000_000]
    for x_tgt in checkpoints:
        idx = int(np.argmin(np.abs(xgrid_dp - x_tgt)))
        e   = eta[idx]
        note = (
            "≈ log utility"     if abs(e - 1) < 0.05 else
            "≈ risk neutral"    if abs(e)     < 0.05 else
            "below log utility" if e < 1 else
            "above log utility"
        )
        print(f"    ${x_tgt:>11,.0f}    {e:>8.3f}    {note}")

    dp_ok = True

except Exception:
    fail("Exception during finite-horizon DP")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Infinite-horizon OBM
# ─────────────────────────────────────────────────────────────────────────────

header("Model 2 — Infinite-horizon OBM")

game_choices = {
    0: (0,   5,   0),    # outside option: no buyin, $5/hr, no variance
    1: (100, 10, 100),   # 1/2 game
    2: (200, 15, 200),   # 2/5 game
    3: (500, 20, 500),   # 5/10 game
}
gridsize_obm = 100
xmax_obm     = 500_000
xmin_obm     = game_choices[1][0]   # smallest non-trivial buyin = 100
N            = len(game_choices)

print(f"  game_choices = {game_choices}")
print(f"  gridsize = {gridsize_obm},  xmin = {xmin_obm},  xmax = {xmax_obm:,},  tol = 1e-2")
print()

xgrid_obm = gen_xgrid(xmax_obm, xmin=xmin_obm, gridsize=gridsize_obm)

obm_ok = False
try:
    # ── 2a. Kelly benchmark (one-shot, no iteration) ─────────────────────────
    t0          = time.time()
    sampling_df = build_sampling_df(game_choices, xgrid_obm)
    CEW_choices_kelly, _ = kelly_cew(sampling_df, xgrid_obm)
    elapsed_kelly = time.time() - t0
    ok(f"Kelly benchmark solved in {elapsed_kelly:.2f}s")

    # ── 2b. OBM value iteration ──────────────────────────────────────────────
    t0 = time.time()
    CEW_array, CEW_choices_obm, vf = solve_obm(game_choices, xgrid_obm, tol=1e-2)
    elapsed_obm = time.time() - t0

    # CEW_array: column 0 = Kelly initialisation, columns 1..k = OBM iterations
    n_iter = CEW_array.shape[1] - 1

    # Approximate final convergence distance from the last two stored CEW columns
    final_dist = (
        np.max(np.abs(CEW_array[:, -1] - CEW_array[:, -2]))
        / np.max(CEW_array[:, -2])
    )

    ok(f"OBM solved in {elapsed_obm:.2f}s  |  {n_iter} iterations")
    row("Final convergence distance:", f"{final_dist:.6f}  (tol = 1e-2)")

    # ── 2c. Game-switch thresholds ───────────────────────────────────────────
    print()
    print("  Game-switch thresholds  (bankroll at which to move up)")
    print()
    print(f"    {'Transition':<20}  {'Kelly':>12}  {'OBM':>12}  {'Δ (OBM−Kelly)':>14}")
    print(f"    {'':─<20}  {'':─>12}  {'':─>12}  {'':─>14}")

    for n in range(1, N):
        label = f"game {n-1} → game {n}"
        try:
            t_kelly = extract_threshold(n, CEW_choices_kelly, xgrid_obm, game_choices)
            s_kelly = f"${t_kelly:>10,.0f}"
        except Exception as e:
            t_kelly = float("nan")
            s_kelly = f"  {'error':>10}"

        try:
            t_obm = extract_threshold(n, CEW_choices_obm, xgrid_obm, game_choices)
            s_obm = f"${t_obm:>10,.0f}"
        except Exception as e:
            t_obm = float("nan")
            s_obm = f"  {'error':>10}"

        if np.isfinite(t_kelly) and np.isfinite(t_obm):
            delta = t_obm - t_kelly
            s_delta = f"${delta:>+12,.0f}"
        else:
            s_delta = f"{'n/a':>14}"

        print(f"    {label:<20}  {s_kelly:>12}  {s_obm:>12}  {s_delta:>14}")

    obm_ok = True

except Exception:
    fail("Exception during infinite-horizon OBM")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

header("Summary")
print(f"  {'Finite-horizon DP':<30}  {'PASS' if dp_ok  else 'FAIL'}")
print(f"  {'Infinite-horizon OBM':<30}  {'PASS' if obm_ok else 'FAIL'}")
print()

sys.exit(0 if (dp_ok and obm_ok) else 1)
