"""
Grid utilities for the bankroll management model.

Provides:
- gen_xgrid:        geometric (log-spaced) bankroll grid
- sample_loglin_vf: log-linear interpolation of the value function
- build_sampling_df: precomputed quadrature sampling table used by OBM and Kelly
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm


def gen_xgrid(xmax: float, xmin: float = 1.0, gridsize: int = 20) -> np.ndarray:
    """
    Generate a geometric (log-spaced) grid over bankroll space.

    Parameters
    ----------
    xmax : float
        Maximum bankroll on the grid.
    xmin : float
        Minimum bankroll on the grid (default 1.0).
    gridsize : int
        Number of grid points (default 20).

    Returns
    -------
    np.ndarray of shape (gridsize,)
    """
    return np.geomspace(xmin, xmax, num=gridsize)


def sample_loglin_vf(
    points: np.ndarray,
    xgrid: np.ndarray,
    vf: np.ndarray,
) -> np.ndarray:
    """
    Evaluate a value function at arbitrary bankroll points using log-linear
    interpolation (linear in log-bankroll), with linear extrapolation at the
    tails.

    Parameters
    ----------
    points : array-like
        Bankroll values at which to evaluate V.
    xgrid : np.ndarray
        Grid points on which vf is defined.
    vf : np.ndarray
        Value function values at each point in xgrid.

    Returns
    -------
    np.ndarray of sampled values, same length as points.
    """
    f = interp1d(np.log(xgrid), vf, fill_value="extrapolate")
    return f(np.log(np.asarray(points, dtype=float)))


def build_sampling_df(
    game_choices: dict,
    xgrid: np.ndarray,
    num_pts: int = 7,
) -> pd.DataFrame:
    """
    Precompute the quadrature sampling table used by the OBM and Kelly models.

    For each (bankroll x, game n) pair, draws num_pts quadrature points from
    a normal approximation to the next-period bankroll distribution:

        x' = x + winrate + z * std,   z ~ N(0,1) discretised to num_pts points

    Viability: game n is available only when x > max(2 * buyin, 3 * std).
    When not viable, x' is set to x (the player sits out and keeps their roll).

    Parameters
    ----------
    game_choices : dict
        Mapping {n: (buyin, winrate, std)} for each game n.
    xgrid : np.ndarray
        Bankroll grid points from gen_xgrid.
    num_pts : int
        Number of quadrature points per (x, game) pair (default 7).

    Returns
    -------
    pd.DataFrame with columns ['wt', 'n', 'x', 'xp', 'viable']
        wt     : quadrature weight
        n      : game index
        x      : current bankroll
        xp     : next-period bankroll (or x when not viable)
        viable : 1.0 if game is available at this bankroll, else 0.0
    """
    quad_pts = np.linspace(-3, 3, num_pts)
    quad_wts = norm.pdf(quad_pts) / norm.pdf(quad_pts).sum()

    records = []
    for x in xgrid:
        for n, (buyin, winrate, std) in game_choices.items():
            viable = float(x > max(std * 3, buyin * 2))
            for wt, z in zip(quad_wts, quad_pts):
                xp = (x + winrate + z * std) if viable else x
                records.append({"wt": wt, "n": n, "x": x, "xp": xp, "viable": viable})

    return pd.DataFrame(records)
