"""
Finite-horizon dynamic programming model for optimal bankroll management.

Solves the player's problem by backward induction over T periods. The state
is the current bankroll x on a geometric grid. Each period the player
selects from N discrete games. The terminal value is V_T(x) = x (risk
neutral), and the model is solved backwards to V_0.

Transition (Bernoulli stack-off rule)
--------------------------------------
Game i has buyin B_i, mean gain mu_i, and std sigma_i. Each period:
  - With probability p_i = (sigma_i / B_i)^2 a full buyin is won or lost.
  - With probability 1 - p_i the bankroll shifts by mu_i (no stack-off).

Viability: game i requires x >= B_i + x_min.

Provides
--------
- OptimalBRM  : backward-induction solver
- calc_eta    : implied CRRA coefficient from the solved value function
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import fsolve


class OptimalBRM:
    """
    Finite-horizon optimal bankroll management via backward induction.

    Parameters
    ----------
    B : array-like, shape (N,)
        Buyin sizes for each game (dollars).
    mu : array-like, shape (N,)
        Mean hourly gain for each game (dollars/hour).
    sigma : array-like, shape (N,)
        Hourly standard deviation for each game (dollars/hour).
    T : int
        Number of periods (default 2000).
    """

    def __init__(self, B, mu, sigma, T: int = 2000):
        self.B = np.asarray(B, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.T = T
        self.N = len(mu)

    def init_VF(self, xmax: float, gridsize: int, xmin: float = 1.0):
        """
        Set up the bankroll grid and initialise the terminal value V_T(x) = x.

        Parameters
        ----------
        xmax : float
            Maximum bankroll on the grid.
        gridsize : int
            Number of grid points.
        xmin : float
            Minimum bankroll (default 1.0).
        """
        self.xmin = xmin
        self.xmax = xmax
        self.gridsize = gridsize
        self.xgrid = np.geomspace(xmin, xmax, num=gridsize)
        # Shape (T+1, gridsize): row t holds V_t(x); V_T(x) = x initialised here.
        self.Vt_init = np.tile(self.xgrid, (self.T + 1, 1))

    def solve(self, xmax: float, gridsize: int = 100, xmin: float = 1.0):
        """
        Solve for V_t(x) and the optimal policy f_t(x) by backward induction.

        Sets the following attributes after solving:
          Vt  : ndarray (T, gridsize)  — value function at each period
          Vti : ndarray (T, gridsize, N) — per-game values before maximisation
          ft  : ndarray (T, gridsize)  — optimal game choice (1-indexed)

        Parameters
        ----------
        xmax : float
            Maximum bankroll on the grid.
        gridsize : int
            Number of grid points (default 100).
        xmin : float
            Minimum bankroll (default 1.0).
        """
        self.init_VF(xmax, gridsize, xmin=xmin)
        B, mu, sigma = self.B, self.mu, self.sigma
        T, N, xmin = self.T, self.N, self.xmin
        xgrid = self.xgrid

        Vt = self.Vt_init.copy()          # (T+1, gridsize)
        Vti = np.zeros((T, gridsize, N))  # per-game values
        ft = np.zeros((T, gridsize))      # optimal policy

        for tback in range(T):
            t = T - (tback + 1)
            Vnext = Vt[t + 1, :]

            for i in range(N):
                Bi = B[i]
                mui = mu[i]
                pi = (sigma[i] / Bi) ** 2 if Bi > 0 else 0.0

                # --- interpolate V at next-period bankrolls ---
                # No stack-off: bankroll shifts by mu_i
                Viu = griddata(xgrid, Vnext, xgrid + mui, method="linear")
                Viu[np.isnan(Viu)] = Vnext[np.isnan(Viu)] + mui

                # Stack-off win: bankroll shifts by mu_i + B_i
                Viw = griddata(xgrid, Vnext, xgrid + mui + Bi, method="linear")
                Viw[np.isnan(Viw)] = Vnext[np.isnan(Viw)] + (mui + Bi)

                # Stack-off loss: bankroll shifts by mu_i - B_i
                Vil = griddata(xgrid, Vnext, xgrid + mui - Bi, method="linear")
                Vil[np.isnan(Vil)] = Vnext[np.isnan(Vil)] + (mui - Bi)

                Vi = (1 - pi) * Viu + pi * (Viw + Vil) / 2
                Vi[xgrid < Bi + xmin] = xmin   # game not viable below buyin threshold

                Vti[t, :, i] = Vi

            Vt[t, :] = np.amax(Vti[t, :, :], axis=1)
            ft[t, :] = np.argmax(Vti[t, :, :], axis=1) + 1  # 1-indexed game labels

        self.Vt = Vt[:-1, :]   # drop terminal row, shape (T, gridsize)
        self.Vti = Vti
        self.ft = ft


def calc_eta(xgrid: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Estimate the implied coefficient of relative risk aversion (CRRA) at each
    grid point by locally inverting the CRRA utility ratio.

    For CRRA utility u(x; eta) = (x^{1-eta} - 1) / (1 - eta), the ratio
    u(x_n) / u(x_{n+1}) pins down eta at each consecutive pair of grid
    points. eta = 0 corresponds to risk neutrality; eta = 1 to log utility.

    Parameters
    ----------
    xgrid : np.ndarray
        Bankroll grid points.
    V : np.ndarray
        Value function evaluated on xgrid (e.g., Vt[0, :]).

    Returns
    -------
    np.ndarray of implied eta at each grid point.
    """
    gridsize = len(xgrid)
    eta = np.zeros(gridsize)

    for n in range(1, gridsize - 1):
        def _residual(e, n=n):
            num = xgrid[n] ** (1 - e) - 1
            den = xgrid[n + 1] ** (1 - e) - 1
            return V[n] / V[n + 1] - num / den

        eta[n] = fsolve(_residual, 0.0)[0]

    # Multiplicative extrapolation at the two boundary points
    eta[0] = eta[1] ** 2 / eta[2]
    eta[-1] = eta[-2] ** 2 / eta[-3]

    return eta
