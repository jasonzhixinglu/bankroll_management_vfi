"""
CEW (Certainty-Equivalent Winrate) computation via Newton solve.

The CEW for bankroll x and game n is the per-period dollar amount that
makes the player indifferent between earning it with certainty and playing
game n under the current value function. Formally, it solves:

    V( x + CEW )  =  E[ V(x') ]

where V is the current value function and the expectation is taken over
next-period bankrolls x'. We invert V via log-linear interpolation and
solve using Newton's method applied simultaneously to all (x, game) rows.

Provides
--------
- compute_cew : Newton-solve CEW for each (bankroll, game) row
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import newton


def compute_cew(
    vf_choices: pd.DataFrame,
    xgrid: np.ndarray,
    vf: np.ndarray,
) -> pd.DataFrame:
    """
    Compute the Certainty-Equivalent Winrate for each (bankroll, game) pair.

    Solves V(CE_x) = E[V(x')] for CE_x using Newton's method, where V is
    interpolated log-linearly on xgrid. Returns CEW = CE_x - x.

    CEW is set to zero for rows where exp_val <= 0 (game not played or
    extrapolation artifacts).

    Parameters
    ----------
    vf_choices : pd.DataFrame
        Per-(bankroll, game) rows with columns ['x', 'n', 'exp_val'], where
        exp_val = E[V(x')] computed using quadrature weights.
    xgrid : np.ndarray
        Bankroll grid points.
    vf : np.ndarray
        Value function on xgrid, used to invert V (i.e., solve V(CE_x) = target).
        Typically this is vf from the *previous* iteration so that the
        inverse is well-defined (monotone) during value iteration.

    Returns
    -------
    pd.DataFrame — vf_choices with an added 'CEW' column.
    """
    # Log-linear interpolant of V: maps log(x) -> V(x)
    v_interp = interp1d(np.log(xgrid), vf, fill_value="extrapolate")

    def _residual(log_xce: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Residual: V(exp(log_xce)) - target = 0."""
        return v_interp(log_xce) - target

    # Solve for log(CE_x) for all rows simultaneously
    log_xce = newton(
        _residual,
        x0=np.log(vf_choices["x"].values),
        args=(vf_choices["exp_val"].values,),
        maxiter=200,
    )

    result = vf_choices.copy()
    result["CEW"] = np.exp(log_xce) - vf_choices["x"].values
    result["CEW"] *= (result["exp_val"] > 0).astype(float)  # zero out non-positive rows

    return result
