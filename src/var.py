"""Portfolio VaR forecasting for copula-MSM models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm, t


def portfolio_var_mc(
    state_probs_1: np.ndarray,
    state_probs_2: np.ndarray,
    sigma_1: float,
    sigma_2: float,
    h_1: np.ndarray,
    h_2: np.ndarray,
    copula_params: dict,
    mean_1: float = 0.0,
    mean_2: float = 0.0,
    copula: str = "student",
    pi: float = 0.5,
    alpha: float = 0.05,
    n_samples: int = 10_000,
    seed: int | None = None,
) -> float:
    """
    Portfolio VaR at level alpha via Monte Carlo — Eq. (12)-(13).
 
    Steps:
        1. Sample (u1, u2) from the fitted copula
        2. Map to returns via MSM inverse CDF
        3. Add marginal means.
        4. r_p = pi*r1 + (1-pi)*r2
        5. VaR = -quantile(r_p, alpha)
 
    Parameters
    ----------
    state_probs_1/2 : np.ndarray
        Filtered state probabilities P(M_{t-1}=m_j | I_{t-1}) for each asset.
    sigma_1/2 : float
        MSM scale parameter for each asset.
    h_1/2 : np.ndarray
        h(m_j) = sqrt(prod(m_j^i)) for each state.
    copula_params : dict
        {"rho": ..., "nu": ...} for Student, {"rho": ...} for Gaussian.
    copula : str
        "student" or "gaussian".
    pi : float
        Weight on asset 1. Paper uses pi=0.5.
    alpha : float
        VaR level. Paper uses 0.01 and 0.05.
    n_samples : int
        Monte Carlo draws. 10_000 gives good precision.
    seed : int or None
        Random seed.
 
    Returns
    -------
    float
        VaR_t(alpha) as a positive loss number.
    """
    rng = np.random.default_rng(seed)
    if copula == "student":
        u1, u2 = _sample_student_copula(
            rho = copula_params["rho"],
            nu=copula_params["nu"],
            n_samples=n_samples,
            rng=rng
        )
    elif copula=="gaussian":
        u1, u2 = _sample_gaussian_copula(
            rho = copula_params["rho"],
            n_samples=n_samples,
            rng=rng
        )
    else:
        raise ValueError("Unsupported copula. Use student or gaussian")
    
    r1 = mean_1 + _msm_quantile(u1, state_probs_1, sigma_1, h_1)
    r2 = mean_2 + _msm_quantile(u2, state_probs_2, sigma_2, h_2)
    r_p = pi * r1 + (1.0 - pi)*r2
    return float(-np.quantile(r_p, alpha))

def portfolio_var_brent(
    state_probs_1: np.ndarray,
    state_probs_2: np.ndarray,
    sigma_1: float,
    sigma_2: float,
    h_1: np.ndarray,
    h_2: np.ndarray,
    copula_params: dict,
    copula: str = "student",
    pi: float = 0.5,
    alpha: float = 0.05,
    n_samples: int = 50_000,
    bracket: tuple[float, float] = (-20.0, 5.0),
    seed: int | None = 0,
) -> float:
    """
    Portfolio VaR via Brent root-finding on the portfolio CDF — Eq. (15).
 
    Solves: Pr(r_p <= VaR) - alpha = 0
 
    Parameters
    ----------
    n_samples : int
        MC draws for CDF evaluation at each Brent iteration.
    bracket : tuple
        (a, b) search interval. Must satisfy F_p(a) < alpha < F_p(b).
    """
    rng = np.random.default_rng(seed)
 
    def objective(v: float) -> float:
        return _portfolio_cdf_mc(
            v,
            state_probs_1, state_probs_2,
            sigma_1, sigma_2,
            h_1, h_2,
            copula_params, copula,
            pi, n_samples, rng,
        ) - alpha
 
    return float(brentq(objective, bracket[0], bracket[1], xtol=1e-3, maxiter=20))


def forecast_msm_copula_var_oos_fixed_params(
    msm_1,
    msm_2,
    returns_1: pd.Series,
    returns_2: pd.Series,
    copula_params: dict,
    copula: str = "student",
    pi: float = 0.5,
    alpha: float = 0.05,
    n_oos: int = 500,
    method: str = "mc",
    n_samples: int = 10_000,
    seed: int | None = 42,
    verbose: bool = True,
) -> pd.Series:
    """
    One-step-ahead VaR forecasts over the out-of-sample window.
 
    Paper setup:
        - T = 1635 total observations
        - n_oos = 500 out-of-sample periods
        - In-sample = first 1135, out-of-sample = last 500
    bracket : tuple
        (a, b) search interval. Must satisfy F_p(a) < alpha < F_p(b) (ignored if method="brent").
    
    """
    try:
        from src.msm import msm_filter_from_result, make_msm_states
    except ImportError:
        from msm import msm_filter_from_result, make_msm_states
 
    if n_oos <= 0:
        raise ValueError("n_oos must be positive.")
    if len(returns_1.dropna()) != len(returns_2.dropna()):
        raise ValueError("returns_1 and returns_2 should have the same length.")
    
    # Run the Hamilton filter on the full series for each asset
    filter_1 = msm_filter_from_result(returns_1, msm_1)
    filter_2 = msm_filter_from_result(returns_2, msm_2)
 
    # predicted_probs shape: (T, n_states) — P(M_t = m_j | I_{t-1})
    predicted_probs_1 = filter_1["predicted_probs"].to_numpy()
    predicted_probs_2 = filter_2["predicted_probs"].to_numpy()
 
    # Precompute h(m_j) = sqrt(prod(m_j)) for each state
    states_1 = make_msm_states(k=msm_1.k, m0=msm_1.params.m0)
    states_2 = make_msm_states(k=msm_2.k, m0=msm_2.params.m0)
    h_1 = np.sqrt(np.prod(states_1, axis=1))
    h_2 = np.sqrt(np.prod(states_2, axis=1))
    sigma_1 = msm_1.params.sigma
    sigma_2 = msm_2.params.sigma
    mean_1 = msm_1.mean_return
    mean_2 = msm_2.mean_return
 
    n_obs = predicted_probs_1.shape[0]
    if n_oos >= n_obs:
        raise ValueError("n_oos must be smaller than the sample size.")
    
    t_start = n_obs - n_oos
    var_values = np.zeros(n_oos)
    var_index = returns_1.dropna().index[t_start:]
 
    for i in range(n_oos):
        t   = t_start + i
        sp1 = predicted_probs_1[t]
        sp2 = predicted_probs_2[t]
 
        if method == "mc":
            var_values[i] = portfolio_var_mc(
                state_probs_1=sp1,
                state_probs_2=sp2,
                sigma_1=sigma_1,
                sigma_2=sigma_2,
                mean_1=mean_1,
                mean_2=mean_2,
                h_1=h_1,
                h_2=h_2,
                copula_params=copula_params,
                copula=copula,
                pi=pi,
                alpha=alpha,
                n_samples=n_samples,
                seed=seed + i if seed is not None else None,
            )
        elif method == "brent":
            var_values[i] = portfolio_var_brent(
                state_probs_1=sp1,
                state_probs_2=sp2,
                sigma_1=sigma_1,
                sigma_2=sigma_2,
                h_1=h_1,
                h_2=h_2,
                copula_params=copula_params,
                copula=copula,
                pi=pi,
                alpha=alpha,
                seed=seed + i if seed is not None else None,
            )
        else:
            raise ValueError("method must be 'mc' or 'brent'")
 
        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_oos}] VaR_{int(alpha*100)}% = {var_values[i]:.4f}")
 
    return pd.Series(
        var_values,
        index=var_index,
        name=f"VaR_{int(alpha * 100)}",
    )


def _portfolio_cdf_mc(
    var_candidate: float,
    state_probs_1: np.ndarray,
    state_probs_2: np.ndarray,
    sigma_1: float,
    sigma_2: float,
    h_1: np.ndarray,
    h_2: np.ndarray,
    copula_params: dict,
    copula: str,
    pi: float,
    n_samples: int,
    rng: np.random.Generator,
    ) -> float:
    """Estimate Pr(r_p <= var_candidate) via MC for use in Brent solver."""
    if copula == "student":
        u1, u2 = _sample_student_copula(
            rho=copula_params["rho"],
            nu=copula_params["nu"],
            n_samples=n_samples,
            rng=rng,
        )
    else:
        u1, u2 = _sample_gaussian_copula(
            rho=copula_params["rho"],
            n_samples=n_samples,
            rng=rng,
        )
 
    r1  = _msm_quantile(u1, state_probs_1, sigma_1, h_1)
    r2  = _msm_quantile(u2, state_probs_2, sigma_2, h_2)
    r_p = pi * r1 + (1.0 - pi) * r2
 
    return float(np.mean(r_p <= var_candidate))

def _sample_student_copula(
    rho: float,
    nu: float,
    n_samples: int,
    rng: np.random.Generator,
    )-> tuple[np.ndarray, np.ndarray]:
    """Sample (u1, u2) from a bivariate Student-t copula"""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal(mean=[0.0, 0.0], cov = cov, size = n_samples)
    chi2 = rng.chisquare(df=nu, size=n_samples)
    x = z * np.sqrt(nu / chi2[:, None])
    return t.cdf(x[:,0], df=nu), t.cdf(x[:,1], df=nu)

def _sample_gaussian_copula(
    rho: float,
    n_samples: int,
    rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Sample (u1, u2) from a bivariate Gaussian copula."""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n_samples)
    return norm.cdf(z[:, 0]), norm.cdf(z[:, 1])

def _msm_quantile(
    u: np.ndarray,
    state_probs: np.ndarray,
    sigma: float,
    h: np.ndarray,
    ) -> np.ndarray:
    """
    Invert the MSM conditional CDF
    F(y | I_{t-1}) = sum_j P(M_{t-1}=m_j | I_{t-1}) * Phi(y / (sigma*h_j))
    Inverted on a fine grid via linear interpolation.
    """
    y_grid = np.linspace(-15.0, 15.0, 5000)
    cdf_grid = np.sum(
        state_probs[:, None] * norm.cdf(y_grid[None, :] / (sigma * h[:, None])),
        axis=0,
    )
    return np.interp(u, cdf_grid, y_grid)

