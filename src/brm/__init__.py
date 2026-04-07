"""
brm — Optimal Bankroll Management
==================================

Two models for selecting and switching between poker games optimally:

  1. Finite-horizon DP  (finite_dp)
     Backward induction over T periods with Bernoulli stack-off transitions.

  2. Infinite-horizon OBM  (infinite_obm, kelly, cew)
     Value iteration converging on CEW distance, with log-linear interpolation
     and 7-point normal quadrature. Benchmarked against one-shot Kelly.

Public API
----------
Grid utilities
    gen_xgrid           geometric bankroll grid
    sample_loglin_vf    log-linear value-function interpolation
    build_sampling_df   quadrature sampling table for OBM / Kelly

Finite-horizon DP
    OptimalBRM          backward-induction solver
    calc_eta            implied CRRA coefficient from the value function

Infinite-horizon OBM
    solve_obm           value iteration to convergence
    kelly_cew           one-shot Kelly benchmark
    compute_cew         Newton-solve CEW per (bankroll, game) row

Thresholds
    extract_threshold   bankroll level at which to move up from game n-1 to n

Quick example
-------------
>>> from brm import gen_xgrid, solve_obm, extract_threshold
>>> game_choices = {0: (0, 5, 0), 1: (100, 10, 100), 2: (500, 20, 500)}
>>> xgrid = gen_xgrid(xmax=500_000, xmin=100, gridsize=80)
>>> CEW_array, CEW_choices, vf = solve_obm(game_choices, xgrid)
>>> thresh = extract_threshold(2, CEW_choices, xgrid, game_choices)
"""

from .grid import gen_xgrid, sample_loglin_vf, build_sampling_df
from .finite_dp import OptimalBRM, calc_eta
from .kelly import kelly_cew
from .cew import compute_cew
from .infinite_obm import solve_obm
from .thresholds import extract_threshold

__all__ = [
    "gen_xgrid",
    "sample_loglin_vf",
    "build_sampling_df",
    "OptimalBRM",
    "calc_eta",
    "kelly_cew",
    "compute_cew",
    "solve_obm",
    "extract_threshold",
]
