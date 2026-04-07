"""
Bankroll threshold extraction for optimal bankroll management.

The threshold for moving up from game n-1 to game n is defined as the
bankroll level where the player is indifferent between the two games, i.e.,
the zero-crossing of:

    CEW_n(x)  -  CEW_{n-1}(x)  =  0

The solve uses Newton's method on a log-linear interpolant of the CEW
difference, restricted to the viable region for game n.

Provides
--------
- extract_threshold : bankroll threshold for switching from game n-1 to n
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import newton


def extract_threshold(
    n: int,
    CEW_choices: pd.DataFrame,
    xgrid: np.ndarray,
    game_choices: dict,
) -> float:
    """
    Find the bankroll at which game n first becomes preferable to game n-1.

    The threshold is the zero-crossing of CEW_n(x) - CEW_{n-1}(x), found
    via Newton's method on a log-linear interpolant. The search is restricted
    to the viable region for game n: x > max(2 * buyin_n, 3 * std_n).

    Parameters
    ----------
    n : int
        Game index to move *up* to (must be >= 1).
    CEW_choices : pd.DataFrame
        Per-(bankroll, game) DataFrame with columns ['x', 'n', 'CEW'],
        as returned by cew.compute_cew or infinite_obm.solve_obm.
    xgrid : np.ndarray
        Bankroll grid points.
    game_choices : dict
        Mapping {n: (buyin, winrate, std)} for each game n.

    Returns
    -------
    float : bankroll threshold for moving up to game n.
    """
    buyin, _, std = game_choices[n]

    cew_n    = CEW_choices.loc[CEW_choices["n"] == n,     "CEW"].values
    cew_prev = CEW_choices.loc[CEW_choices["n"] == n - 1, "CEW"].values
    cew_diff = cew_n - cew_prev

    # Restrict to viable region: game n requires x > max(2*buyin, 3*std)
    min_bankroll = max(buyin * 2, std * 3)
    cew_diff = cew_diff.copy()
    cew_diff[xgrid < min_bankroll] = np.nan

    valid = ~np.isnan(cew_diff)
    log_x_valid    = np.log(xgrid[valid])
    cew_diff_valid = cew_diff[valid]

    # Log-linear interpolant of CEW difference (with extrapolation at tails)
    interpolant = interp1d(log_x_valid, cew_diff_valid, fill_value="extrapolate")

    # Initial guess: geometric midpoint of the viable bankroll range
    x0 = np.mean(log_x_valid)
    log_threshold = newton(interpolant, x0)

    return float(np.exp(log_threshold))
