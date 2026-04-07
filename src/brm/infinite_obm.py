"""
Infinite-horizon Optimal Bankroll Management via value iteration.

Algorithm
---------
1. Initialise V_0(x) = log(x)  [Kelly / log-utility starting point]
2. Compute E[V_k(x')] for each (bankroll, game) pair using 7-point
   normal quadrature, with log-linear interpolation of V.
3. Update V_{k+1}(x) = max_n E[V_k(x')].
4. Compute the CEW for each (x, game) via Newton solve on V_k^{-1}.
5. Repeat until the CEW distance  max|CEW_k - CEW_{k-1}| / max(CEW_{k-1})
   falls below tol (minimum 2 iterations are always run).

Non-viable games: game n is excluded when x <= max(2*buyin, 3*std). In
that period, the player sits out and V is evaluated at the unchanged x
using the *previous* iteration's value function.

Provides
--------
- solve_obm : run value iteration and return (CEW_array, CEW_choices, vf)
"""

import numpy as np
import pandas as pd

from .grid import build_sampling_df, sample_loglin_vf
from .kelly import kelly_cew
from .cew import compute_cew


def solve_obm(
    game_choices: dict,
    xgrid: np.ndarray,
    tol: float = 1e-2,
) -> tuple:
    """
    Solve the infinite-horizon OBM model by CEW-distance value iteration.

    Parameters
    ----------
    game_choices : dict
        Mapping {n: (buyin, winrate, std)} for each game n.
    xgrid : np.ndarray
        Bankroll grid from grid.gen_xgrid.
    tol : float
        Convergence tolerance on max relative CEW change (default 1e-2).

    Returns
    -------
    CEW_array : np.ndarray, shape (gridsize, n_iter + 1)
        CEW at each grid point per iteration. Column 0 is the Kelly baseline;
        subsequent columns are OBM CEW values from each iteration.
    CEW_choices : pd.DataFrame
        Final per-(bankroll, game) DataFrame with columns
        ['x', 'n', 'exp_val', 'CEW'].
    vf : np.ndarray, shape (gridsize,)
        Converged value function.
    """
    # --- Precompute quadrature sampling table (static across iterations) ---
    sampling_df = build_sampling_df(game_choices, xgrid)

    # --- Initialise: V_0(x) = log(x), CEW_0 = Kelly CEW ---
    vf = np.log(xgrid)
    CEW_choices, CEW_kelly = kelly_cew(sampling_df, xgrid)
    CEW_array = CEW_kelly.copy()   # shape (gridsize,); will grow column-wise

    dist = np.inf
    n_iter = 0

    while dist > tol or n_iter < 2:
        n_iter += 1
        vf_old = vf.copy()
        CEW_old = CEW_choices["CEW"].values.copy()

        # --- E-step: sample V(x') at next-period bankrolls ---
        value_sample = sampling_df.copy()
        value_sample["exp_val"] = sample_loglin_vf(
            sampling_df["xp"].values, xgrid, vf
        )

        # Non-viable rows: player sits out; evaluate V_old at current bankroll x
        non_viable = value_sample["viable"] == 0
        value_sample.loc[non_viable, "exp_val"] = sample_loglin_vf(
            value_sample.loc[non_viable, "x"].values, xgrid, vf_old
        )

        # Apply quadrature weights to get E[V(x')] per quadrature point
        value_sample["exp_val"] *= value_sample["wt"]

        # --- Aggregate: sum over quadrature points for each (x, game) ---
        vf_choices = (
            value_sample[["x", "n", "exp_val"]]
            .groupby(["x", "n"])
            .sum()
            .reset_index()
        )

        # --- M-step: update value function as max over games ---
        vf = vf_choices.groupby("x")["exp_val"].max().values

        # --- CEW solve: invert V_old to get certainty-equivalent bankrolls ---
        CEW_choices = compute_cew(vf_choices, xgrid, vf_old)
        CEW = CEW_choices.groupby("x")["CEW"].max().values
        CEW_array = np.c_[CEW_array, CEW]

        # --- Convergence: max absolute CEW change, normalised by current max ---
        dist = np.max(np.abs(CEW_choices["CEW"].values - CEW_old)) / np.max(CEW_old)

    return CEW_array, CEW_choices, vf
