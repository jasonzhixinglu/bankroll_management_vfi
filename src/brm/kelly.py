"""
One-shot Kelly benchmark for bankroll management.

The Kelly criterion maximises E[log x'], which is equivalent to log utility
(CRRA with eta = 1). It is optimal for long-run growth when stakes can be
continuously rescaled. Here we compute it in closed form from the precomputed
quadrature sampling table — no value-function iteration is needed.

Kelly CEW (Certainty-Equivalent Winrate) is defined as:

    CEW_kelly(x, n)  =  exp( E[log x'] )  -  x

where the expectation uses the quadrature weights in sampling_df.
This equals the geometric mean of next-period bankrolls, minus x.

Provides
--------
- kelly_cew : compute Kelly CEW for each (bankroll, game) pair
"""

import numpy as np
import pandas as pd


def kelly_cew(
    sampling_df: pd.DataFrame,
    xgrid: np.ndarray,
) -> tuple:
    """
    Compute the Kelly Certainty-Equivalent Winrate for every (bankroll, game)
    pair using log utility — no value-function iteration required.

    Parameters
    ----------
    sampling_df : pd.DataFrame
        Precomputed sampling table from grid.build_sampling_df.
        Required columns: ['wt', 'n', 'x', 'xp', 'viable'].
    xgrid : np.ndarray
        Bankroll grid points (determines ordering of the returned array).

    Returns
    -------
    CEW_choices : pd.DataFrame
        Per-(bankroll, game) rows with columns ['x', 'n', 'exp_val', 'CEW'].
        exp_val = E[log x'] under quadrature weights.
        CEW     = exp(E[log x']) - x  (Kelly certainty-equivalent winrate).
    CEW : np.ndarray of shape (gridsize,)
        Optimal (max over games) Kelly CEW at each bankroll grid point.
    """
    df = sampling_df.copy()
    # E[log x'] = weighted sum of log(x') over quadrature points
    df["exp_val"] = np.log(df["xp"]) * df["wt"]

    CEW_choices = (
        df[["x", "n", "exp_val"]]
        .groupby(["x", "n"])
        .sum()
        .reset_index()
    )
    # Kelly CE bankroll = exp(E[log x']); CEW = CE bankroll - x
    CEW_choices["CEW"] = np.exp(CEW_choices["exp_val"]) - CEW_choices["x"]

    CEW = CEW_choices.groupby("x")["CEW"].max().values

    return CEW_choices, CEW
