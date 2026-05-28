"""Paper-like VaR forecasting helpers for Segnon & Trede copula-MSM replication.

Conventions
-----------
- Inputs are percentage log returns: 100 * log(P_t/P_{t-1}).
- VaR is a signed return quantile, usually negative:
      P(r_p,t <= VaR_t(alpha) | Omega_{t-1}) = alpha
- Rolling evaluation uses a fixed estimation window, e.g. 1135 observations,
  and forecasts the next 500 one-day-ahead VaR values.

Main paper-like functions
-------------------------
- forecast_historical_var_rolling
- forecast_variance_covariance_var_rolling
- forecast_riskmetrics_var_rolling
- forecast_ccc_garch_var_rolling
- forecast_garch_copula_var_rolling
- forecast_msm_copula_var_rolling
- forecast_all_var_models

Notes
-----
The copula portfolio CDF is evaluated through the identity

    F_p(q) = integral_0^1 P(U1 <= F1((q - (1-pi)F2^{-1}(u2))/pi) | U2=u2) du2.

For Gaussian and Student copulas, the conditional CDF is analytic. For Plackett,
Clayton, rotated Clayton, Frank, Gumbel, rotated Gumbel, and SJC, it is computed
as a stable finite-difference derivative of the copula CDF with respect to u2.
This is slower but directly matches the paper's numerical-integration approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, multivariate_t, norm, t

EPS = 1e-10

SUPPORTED_COPULAS = (
    "gaussian",
    "student",
    "plackett",
    "clayton",
    "rotated_clayton",
    "sjc",
    "frank",
    "gumbel",
    "rotated_gumbel",
)


@dataclass(frozen=True)
class RollingSpec:
    """Rolling-window VaR evaluation specification."""

    n_insample: int = 1135
    n_oos: int = 500
    alpha: float = 0.05
    weights: tuple[float, float] = (0.5, 0.5)


# -----------------------------------------------------------------------------
# Generic validation and rolling-window helpers
# -----------------------------------------------------------------------------


def prepare_bivariate_returns(
    returns: pd.DataFrame,
    n_insample: int = 1135,
    n_oos: int = 500,
) -> pd.DataFrame:
    """Clean and truncate bivariate returns to the paper's rolling sample length."""
    frame = returns.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if frame.shape[1] != 2:
        raise ValueError("The paper replication expects exactly two return columns.")

    required = n_insample + n_oos
    if len(frame) < required:
        raise ValueError(f"Need at least {required} observations, got {len(frame)}.")

    # Use the first required observations to reproduce the 1135/500 split exactly.
    return frame.iloc[:required].copy()


def validate_alpha_weights(alpha: float, weights: Iterable[float]) -> np.ndarray:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    w = np.asarray(tuple(weights), dtype=float)
    if w.shape != (2,):
        raise ValueError("weights must contain exactly two entries.")
    if np.any(w < 0.0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must be non-negative and sum to 1.")
    return w


def rolling_windows(frame: pd.DataFrame, n_insample: int, n_oos: int):
    """Yield (i, forecast_date, estimation_window) for a fixed-size rolling scheme."""
    frame = prepare_bivariate_returns(frame, n_insample=n_insample, n_oos=n_oos)
    for i in range(n_oos):
        start = i
        end = i + n_insample
        yield i, frame.index[end], frame.iloc[start:end]


# -----------------------------------------------------------------------------
# Benchmark VaR methods from Section 3.3 / Section 4.3
# -----------------------------------------------------------------------------


def forecast_historical_var_rolling(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    weights: Iterable[float] = (0.5, 0.5),
    n_insample: int = 1135,
    n_oos: int = 500,
) -> pd.Series:
    """Rolling historical simulation VaR of portfolio returns."""
    w = validate_alpha_weights(alpha, weights)
    frame = prepare_bivariate_returns(returns, n_insample, n_oos)
    portfolio = frame @ w
    values, dates = [], []

    for i in range(n_oos):
        end = i + n_insample
        window = portfolio.iloc[i:end]
        values.append(float(np.quantile(window, alpha)))
        dates.append(portfolio.index[end])

    return pd.Series(values, index=dates, name=f"HS_VaR_{alpha:g}")


def forecast_variance_covariance_var_rolling(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    weights: Iterable[float] = (0.5, 0.5),
    n_insample: int = 1135,
    n_oos: int = 500,
    include_mean: bool = True,
) -> pd.Series:
    """Rolling variance-covariance VaR under conditional normality."""
    w = validate_alpha_weights(alpha, weights)
    frame = prepare_bivariate_returns(returns, n_insample, n_oos)
    z_alpha = norm.ppf(alpha)
    values, dates = [], []

    for _, date, window in rolling_windows(frame, n_insample, n_oos):
        mu = window.mean().to_numpy(dtype=float) if include_mean else np.zeros(2)
        cov = window.cov().to_numpy(dtype=float)
        port_mu = float(w @ mu)
        port_var = float(w @ cov @ w)
        values.append(port_mu + np.sqrt(max(port_var, 0.0)) * z_alpha)
        dates.append(date)

    return pd.Series(values, index=dates, name=f"VarCov_VaR_{alpha:g}")


def forecast_riskmetrics_var_rolling(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    weights: Iterable[float] = (0.5, 0.5),
    lambda_: float = 0.94,
    n_insample: int = 1135,
    n_oos: int = 500,
    include_mean: bool = False,
) -> pd.Series:
    """Paper-like RiskMetrics VaR using scalar EWMA variance of portfolio returns.

    sigma^2_{p,t|t-1} = (1-lambda) r^2_{p,t-1} + lambda sigma^2_{p,t-1|t-2}.
    """
    w = validate_alpha_weights(alpha, weights)
    if not 0.0 < lambda_ < 1.0:
        raise ValueError("lambda_ must be in (0, 1).")

    frame = prepare_bivariate_returns(returns, n_insample, n_oos)
    portfolio = frame @ w
    z_alpha = norm.ppf(alpha)

    initial_window = portfolio.iloc[:n_insample]
    sigma2 = float(initial_window.var(ddof=1))
    values, dates = [], []

    for i in range(n_oos):
        forecast_pos = i + n_insample
        # The forecast for date t uses r_{t-1}.
        r_lag = float(portfolio.iloc[forecast_pos - 1])
        sigma2 = (1.0 - lambda_) * r_lag**2 + lambda_ * sigma2
        mu = float(portfolio.iloc[i:forecast_pos].mean()) if include_mean else 0.0
        values.append(mu + np.sqrt(max(sigma2, 0.0)) * z_alpha)
        dates.append(portfolio.index[forecast_pos])

    return pd.Series(values, index=dates, name=f"RiskMetrics_VaR_{alpha:g}")


# -----------------------------------------------------------------------------
# Copula CDFs and conditional CDFs: P(U1 <= u1 | U2 = u2) = dC/du2
# -----------------------------------------------------------------------------


def _clip_unit(x):
    return np.clip(np.asarray(x, dtype=float), EPS, 1.0 - EPS)


def copula_cdf(u: np.ndarray, v: np.ndarray, copula_params: dict[str, float], copula: str) -> np.ndarray:
    """Bivariate copula CDF C(u, v) for all paper copulas."""
    copula = copula.lower()
    u = _clip_unit(u)
    v = _clip_unit(v)

    if copula == "gaussian":
        rho = float(copula_params["rho"])
        z = np.column_stack([norm.ppf(u.ravel()), norm.ppf(v.ravel())])
        corr = np.array([[1.0, rho], [rho, 1.0]])
        out = multivariate_normal.cdf(z, mean=np.zeros(2), cov=corr)
        return np.asarray(out).reshape(np.broadcast(u, v).shape)

    if copula == "student":
        rho = float(copula_params["rho"])
        nu = float(copula_params["nu"])
        x = np.column_stack([t.ppf(u.ravel(), df=nu), t.ppf(v.ravel(), df=nu)])
        shape = np.array([[1.0, rho], [rho, 1.0]])
        out = multivariate_t.cdf(x, loc=np.zeros(2), shape=shape, df=nu)
        return np.asarray(out).reshape(np.broadcast(u, v).shape)

    if copula == "clayton":
        theta = float(copula_params["theta"])
        return np.maximum((u ** (-theta) + v ** (-theta) - 1.0), EPS) ** (-1.0 / theta)

    if copula == "rotated_clayton":
        theta = float(copula_params["theta"])
        return u + v - 1.0 + copula_cdf(1.0 - u, 1.0 - v, {"theta": theta}, "clayton")

    if copula == "gumbel":
        theta = float(copula_params["theta"])
        x = (-np.log(u)) ** theta + (-np.log(v)) ** theta
        return np.exp(-(x ** (1.0 / theta)))

    if copula == "rotated_gumbel":
        theta = float(copula_params["theta"])
        return u + v - 1.0 + copula_cdf(1.0 - u, 1.0 - v, {"theta": theta}, "gumbel")

    if copula == "frank":
        theta = float(copula_params["theta"])
        if abs(theta) < 1e-8:
            return u * v
        a = np.expm1(-theta * u)
        b = np.expm1(-theta * v)
        d = np.expm1(-theta)
        inside = 1.0 + (a * b) / d
        return -np.log(np.maximum(inside, EPS)) / theta

    if copula == "plackett":
        theta = float(copula_params["theta"])
        if abs(theta - 1.0) < 1e-8:
            return u * v
        a = 1.0 + (theta - 1.0) * (u + v)
        disc = np.maximum(a * a - 4.0 * theta * (theta - 1.0) * u * v, EPS)
        return (a - np.sqrt(disc)) / (2.0 * (theta - 1.0))

    if copula == "sjc":
        tau_u = float(copula_params["tau_u"])
        tau_l = float(copula_params["tau_l"])
        jc = _joe_clayton_cdf(u, v, tau_u=tau_u, tau_l=tau_l)
        rotated = u + v - 1.0 + _joe_clayton_cdf(1.0 - u, 1.0 - v, tau_u=tau_l, tau_l=tau_u)
        return 0.5 * (jc + rotated)

    raise ValueError(f"Unsupported copula: {copula}")


def _joe_clayton_cdf(
    u: np.ndarray,
    v: np.ndarray,
    tau_u: float,
    tau_l: float,
) -> np.ndarray:
    """Stable Joe-Clayton / BB7 CDF parameterized by tail dependence."""
    u = np.clip(np.asarray(u, dtype=float), EPS, 1.0 - EPS)
    v = np.clip(np.asarray(v, dtype=float), EPS, 1.0 - EPS)

    if not (0.0 < tau_u < 1.0) or not (0.0 < tau_l < 1.0):
        raise ValueError("SJC tau_u and tau_l must be in (0, 1).")

    kappa = 1.0 / np.log2(2.0 - tau_u)
    gamma = -1.0 / np.log2(tau_l)

    log_1_minus_u = np.log1p(-u)
    log_1_minus_v = np.log1p(-v)

    a = -np.expm1(kappa * log_1_minus_u)
    b = -np.expm1(kappa * log_1_minus_v)

    a = np.clip(a, EPS, 1.0 - EPS)
    b = np.clip(b, EPS, 1.0 - EPS)

    log_a = np.log(a)
    log_b = np.log(b)

    x = -gamma * log_a
    y = -gamma * log_b

    m = np.maximum(x, y)

    log_s = m + np.log(
        np.exp(x - m) + np.exp(y - m) - np.exp(-m)
    )

    log_s = np.maximum(log_s, np.log(1.0 + EPS))

    s_neg_q = np.exp(-log_s / gamma)
    s_neg_q = np.clip(s_neg_q, EPS, 1.0 - EPS)

    inner = 1.0 - s_neg_q
    inner = np.clip(inner, EPS, 1.0 - EPS)

    # C = 1 - inner^(1/kappa), computed stably
    log_inner = np.log(inner)
    cdf = -np.expm1((1.0 / kappa) * log_inner)

    return np.clip(cdf, EPS, 1.0 - EPS)


def copula_conditional_cdf_u1_given_u2(
    u1: np.ndarray,
    u2: np.ndarray,
    copula_params: dict[str, float],
    copula: str = "student",
    finite_diff_step: float = 1e-5,
) -> np.ndarray:
    """Compute P(U1 <= u1 | U2 = u2) = partial C(u1,u2)/partial u2."""
    copula = copula.lower()
    u1 = _clip_unit(u1)
    u2 = _clip_unit(u2)

    if copula == "gaussian":
        rho = float(copula_params["rho"])
        z1 = norm.ppf(u1)
        z2 = norm.ppf(u2)
        return norm.cdf((z1 - rho * z2) / np.sqrt(1.0 - rho**2))

    if copula == "student":
        rho = float(copula_params["rho"])
        nu = float(copula_params["nu"])
        x1 = t.ppf(u1, df=nu)
        x2 = t.ppf(u2, df=nu)
        cond_df = nu + 1.0
        cond_mean = rho * x2
        cond_scale = np.sqrt(((nu + x2**2) * (1.0 - rho**2)) / (nu + 1.0))
        return t.cdf((x1 - cond_mean) / cond_scale, df=cond_df)

    # Numerical derivative for remaining copulas. Use central differences away
    # from boundaries and one-sided differences near boundaries.
    h = finite_diff_step
    u2_low = np.maximum(EPS, u2 - h)
    u2_high = np.minimum(1.0 - EPS, u2 + h)
    c_high = copula_cdf(u1, u2_high, copula_params, copula)
    c_low = copula_cdf(u1, u2_low, copula_params, copula)
    deriv = (c_high - c_low) / (u2_high - u2_low)
    return np.clip(deriv, EPS, 1.0 - EPS)


# -----------------------------------------------------------------------------
# Marginal CDF / quantile helpers
# -----------------------------------------------------------------------------


def msm_conditional_cdf(y: np.ndarray | float, state_probs: np.ndarray, sigma: float, h: np.ndarray) -> np.ndarray:
    """MSM conditional CDF for centered returns."""
    y_values = np.asarray(y, dtype=float).reshape(-1)
    p = _normalize_probabilities(state_probs)
    h = _validate_positive_vector(h, "h")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    cdf = np.sum(p[:, None] * norm.cdf(y_values[None, :] / (sigma * h[:, None])), axis=0)
    return np.clip(cdf, EPS, 1.0 - EPS)


def msm_conditional_quantile(
    u: np.ndarray | float,
    state_probs: np.ndarray,
    sigma: float,
    h: np.ndarray,
    grid_size: int = 20001,
    tail_std_multiplier: float = 10.0,
) -> np.ndarray:
    """Invert MSM mixture-normal CDF by monotone interpolation."""
    uniforms = _clip_unit(u).reshape(-1)
    p = _normalize_probabilities(state_probs)
    h = _validate_positive_vector(h, "h")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    max_scale = sigma * float(np.max(h))
    grid = np.linspace(-tail_std_multiplier * max_scale, tail_std_multiplier * max_scale, grid_size)
    cdf_grid = np.sum(p[:, None] * norm.cdf(grid[None, :] / (sigma * h[:, None])), axis=0)
    cdf_grid = np.maximum.accumulate(np.clip(cdf_grid, EPS, 1.0 - EPS))
    return np.interp(uniforms, cdf_grid, grid)


def _normalize_probabilities(state_probs: np.ndarray) -> np.ndarray:
    p = np.asarray(state_probs, dtype=float).reshape(-1)
    if np.any(p < 0) or not np.isfinite(p).all():
        raise ValueError("state probabilities must be finite and non-negative.")
    total = p.sum()
    if total <= 0:
        raise ValueError("state probabilities must sum to a positive value.")
    return p / total


def _validate_positive_vector(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if np.any(arr <= 0) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite positive values.")
    return arr


def _unit_interval_gauss_legendre(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    if n_nodes < 51:
        raise ValueError("n_nodes should be at least 51.")
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    nodes = np.clip(0.5 * (x + 1.0), EPS, 1.0 - EPS)
    weights = 0.5 * w
    return nodes, weights


# -----------------------------------------------------------------------------
# Copula-based portfolio CDF and VaR solvers
# -----------------------------------------------------------------------------


def portfolio_cdf_from_margins_and_copula(
    q: float,
    inverse_cdf_2: Callable[[np.ndarray], np.ndarray],
    cdf_1: Callable[[np.ndarray], np.ndarray],
    copula_params: dict[str, float],
    copula: str,
    pi: float = 0.5,
    integration_nodes: int = 501,
) -> float:
    """Generic copula portfolio CDF for two continuous conditional margins."""
    if not 0.0 < pi < 1.0:
        raise ValueError("pi must be in (0, 1).")
    nodes, weights = _unit_interval_gauss_legendre(integration_nodes)
    r2 = inverse_cdf_2(nodes)
    r1_threshold = (q - (1.0 - pi) * r2) / pi
    u1_threshold = cdf_1(r1_threshold)
    cond = copula_conditional_cdf_u1_given_u2(u1_threshold, nodes, copula_params, copula)
    return float(np.sum(weights * cond))


def solve_portfolio_var(
    alpha: float,
    inverse_cdf_1: Callable[[np.ndarray], np.ndarray],
    inverse_cdf_2: Callable[[np.ndarray], np.ndarray],
    cdf_1: Callable[[np.ndarray], np.ndarray],
    copula_params: dict[str, float],
    copula: str,
    pi: float = 0.5,
    integration_nodes: int = 501,
    root_tol: float = 1e-4,
) -> float:
    """Solve F_p(q) = alpha for the signed portfolio return VaR."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    if pi == 1.0:
        return float(inverse_cdf_1(np.array([alpha]))[0])
    if pi == 0.0:
        return float(inverse_cdf_2(np.array([alpha]))[0])

    # Conservative bracket based on near-endpoint marginal quantiles.
    low_u, high_u = EPS, 1.0 - EPS
    lower = pi * inverse_cdf_1(np.array([low_u]))[0] + (1.0 - pi) * inverse_cdf_2(np.array([low_u]))[0]
    upper = pi * inverse_cdf_1(np.array([high_u]))[0] + (1.0 - pi) * inverse_cdf_2(np.array([high_u]))[0]

    def objective(q: float) -> float:
        return portfolio_cdf_from_margins_and_copula(
            q=q,
            inverse_cdf_2=inverse_cdf_2,
            cdf_1=cdf_1,
            copula_params=copula_params,
            copula=copula,
            pi=pi,
            integration_nodes=integration_nodes,
        ) - alpha

    f_low = objective(lower)
    f_high = objective(upper)
    expand = 0
    while not (f_low <= 0.0 <= f_high):
        width = upper - lower
        lower -= width
        upper += width
        f_low = objective(lower)
        f_high = objective(upper)
        expand += 1
        if expand > 12:
            raise RuntimeError(f"Unable to bracket VaR root: [{lower}, {upper}], f=[{f_low}, {f_high}]")

    return float(brentq(objective, lower, upper, xtol=root_tol, rtol=1e-6, maxiter=100))


# -----------------------------------------------------------------------------
# Rolling Copula-MSM VaR
# -----------------------------------------------------------------------------


def forecast_msm_copula_var_rolling(
    returns: pd.DataFrame,
    copula: str = "student",
    alpha: float = 0.05,
    weights: Iterable[float] = (0.5, 0.5),
    k: int = 5,
    n_insample: int = 1135,
    n_oos: int = 500,
    n_starts: int = 10,
    seed: int = 123,
    integration_nodes: int = 501,
    root_tol: float = 1e-4,
    verbose: bool = True,
) -> pd.Series:
    """Rolling one-step-ahead Copula-MSM VaR with MSM warm-start.

    Paper-like structure:
        - fixed rolling window;
        - MSM re-estimated at each date;
        - copula re-estimated at each date;
        - one-step-ahead portfolio VaR solved numerically.

    Warm-start only changes the numerical initialization of the MSM optimizer.
    It does not change the likelihood, model, rolling window, copula estimation,
    or VaR equation.
    """
    try:
        from src.msm import (
            fit_msm,
            make_msm_states,
            msm_filter_from_result,
            renewal_probabilities_from_gamma_k,
            transition_matrix_from_gammas,
        )
        from src.copulas import fit_copula
    except ImportError:
        from msm import (
            fit_msm,
            make_msm_states,
            msm_filter_from_result,
            renewal_probabilities_from_gamma_k,
            transition_matrix_from_gammas,
        )
        from copulas import fit_copula

    w = validate_alpha_weights(alpha, weights)
    pi = float(w[0])

    frame = prepare_bivariate_returns(
        returns,
        n_insample=n_insample,
        n_oos=n_oos,
    )

    assets = list(frame.columns)
    values: list[float] = []
    dates: list[pd.Timestamp] = []

    # Warm-start containers.
    # prev_xj = [m0, sigma, b, gamma_k] from previous rolling window.
    prev_x1 = None
    prev_x2 = None

    for i, date, window in rolling_windows(frame, n_insample, n_oos):
        r1 = window[assets[0]]
        r2 = window[assets[1]]

        initial_extra_1 = [prev_x1] if prev_x1 is not None else None
        initial_extra_2 = [prev_x2] if prev_x2 is not None else None

        msm_1 = fit_msm(
            returns=r1,
            k=k,
            n_starts=n_starts,
            seed=seed + 10000 * i + 1,
            verbose=False,
            initial_points_extra=initial_extra_1,
        )

        msm_2 = fit_msm(
            returns=r2,
            k=k,
            n_starts=n_starts,
            seed=seed + 10000 * i + 2,
            verbose=False,
            initial_points_extra=initial_extra_2,
        )

        # Update warm-starts for next rolling window.
        prev_x1 = np.array(
            [
                msm_1.params.m0,
                msm_1.params.sigma,
                msm_1.params.b,
                msm_1.params.gamma_k,
            ],
            dtype=float,
        )

        prev_x2 = np.array(
            [
                msm_2.params.m0,
                msm_2.params.sigma,
                msm_2.params.b,
                msm_2.params.gamma_k,
            ],
            dtype=float,
        )

        filt_1 = msm_filter_from_result(r1, msm_1)
        filt_2 = msm_filter_from_result(r2, msm_2)

        pit = pd.concat(
            [
                filt_1["pit"].rename(assets[0]),
                filt_2["pit"].rename(assets[1]),
            ],
            axis=1,
        ).dropna()

        cop_fit = fit_copula(
            pit,
            copula=copula,
            margin_model="MSM",
        )

        gammas_1 = renewal_probabilities_from_gamma_k(
            k=k,
            b=msm_1.params.b,
            gamma_k=msm_1.params.gamma_k,
        )

        gammas_2 = renewal_probabilities_from_gamma_k(
            k=k,
            b=msm_2.params.b,
            gamma_k=msm_2.params.gamma_k,
        )

        A_1 = transition_matrix_from_gammas(gammas_1)
        A_2 = transition_matrix_from_gammas(gammas_2)

        # One-step-ahead predictive state probabilities:
        # P(S_t | Omega_{t-1}) = P(S_{t-1} | Omega_{t-1}) A
        p1 = filt_1["filtered_probs"].iloc[-1].to_numpy(dtype=float) @ A_1
        p2 = filt_2["filtered_probs"].iloc[-1].to_numpy(dtype=float) @ A_2

        states_1 = make_msm_states(k=k, m0=msm_1.params.m0)
        states_2 = make_msm_states(k=k, m0=msm_2.params.m0)

        h1 = np.sqrt(np.prod(states_1, axis=1))
        h2 = np.sqrt(np.prod(states_2, axis=1))

        def inv1(u):
            return (
                msm_1.mean_return
                + msm_conditional_quantile(
                    u,
                    p1,
                    msm_1.params.sigma,
                    h1,
                )
            )

        def inv2(u):
            return (
                msm_2.mean_return
                + msm_conditional_quantile(
                    u,
                    p2,
                    msm_2.params.sigma,
                    h2,
                )
            )

        def cdf1(x):
            return msm_conditional_cdf(
                np.asarray(x) - msm_1.mean_return,
                p1,
                msm_1.params.sigma,
                h1,
            )

        var_t = solve_portfolio_var(
            alpha=alpha,
            inverse_cdf_1=inv1,
            inverse_cdf_2=inv2,
            cdf_1=cdf1,
            copula_params=cop_fit.params,
            copula=copula,
            pi=pi,
            integration_nodes=integration_nodes,
            root_tol=root_tol,
        )

        values.append(var_t)
        dates.append(date)

        if verbose and (i + 1) % 10 == 0:
            print(
                f"MSM-{copula} alpha={alpha:g}: "
                f"{i + 1}/{n_oos}, "
                f"VaR={var_t:.4f}, "
                f"x1={prev_x1.round(4)}, "
                f"x2={prev_x2.round(4)}"
            )

    return pd.Series(
        values,
        index=dates,
        name=f"CopulaMSM_{copula}_VaR_{alpha:g}",
    )

# -----------------------------------------------------------------------------
# Rolling GARCH, Copula-GARCH, CCC-GARCH
# -----------------------------------------------------------------------------


def _fit_arch_garch_11(series: pd.Series, dist: str = "normal"):
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("Install the `arch` package: pip install arch") from exc
    model = arch_model(series, mean="Constant", vol="GARCH", p=1, q=1, dist=dist, rescale=False)
    return model.fit(disp="off")


def _garch_one_step_forecast(result) -> tuple[float, float]:
    """Return one-step-ahead mean and volatility from an arch result."""
    params = result.params
    mu = float(params.get("mu", 0.0))
    forecast = result.forecast(horizon=1, reindex=False)
    variance = float(forecast.variance.iloc[-1, 0])
    return mu, np.sqrt(max(variance, 0.0))


def forecast_ccc_garch_var_rolling(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    weights: Iterable[float] = (0.5, 0.5),
    n_insample: int = 1135,
    n_oos: int = 500,
    include_mean: bool = True,
    verbose: bool = True,
) -> pd.Series:
    """Rolling CCC-GARCH VaR with univariate GARCH(1,1)-normal margins."""
    w = validate_alpha_weights(alpha, weights)
    frame = prepare_bivariate_returns(returns, n_insample, n_oos)
    assets = list(frame.columns)
    z_alpha = norm.ppf(alpha)
    values, dates = [], []

    for i, date, window in rolling_windows(frame, n_insample, n_oos):
        fits = [_fit_arch_garch_11(window[a]) for a in assets]
        forecasts = [_garch_one_step_forecast(fit) for fit in fits]
        mus = np.array([m for m, _ in forecasts], dtype=float)
        sigmas = np.array([s for _, s in forecasts], dtype=float)

        std_resids = pd.concat(
            [fit.std_resid.dropna().rename(asset) for fit, asset in zip(fits, assets)], axis=1
        ).dropna()
        rho = float(std_resids.corr().iloc[0, 1])
        corr = np.array([[1.0, rho], [rho, 1.0]])
        D = np.diag(sigmas)
        cov = D @ corr @ D
        port_mu = float(w @ mus) if include_mean else 0.0
        port_var = float(w @ cov @ w)
        var_t = port_mu + np.sqrt(max(port_var, 0.0)) * z_alpha
        values.append(var_t)
        dates.append(date)

        if verbose and (i + 1) % 25 == 0:
            print(f"CCC-GARCH alpha={alpha:g}: {i + 1}/{n_oos}, VaR={var_t:.4f}")

    return pd.Series(values, index=dates, name=f"CCC_GARCH_VaR_{alpha:g}")


def forecast_garch_copula_var_rolling(
    returns: pd.DataFrame,
    copula: str = "student",
    alpha: float = 0.05,
    weights: Iterable[float] = (0.5, 0.5),
    n_insample: int = 1135,
    n_oos: int = 500,
    integration_nodes: int = 501,
    root_tol: float = 1e-4,
    verbose: bool = True,
) -> pd.Series:
    """Rolling one-step-ahead Copula-GARCH VaR with Gaussian GARCH margins."""
    try:
        from src.copulas import fit_copula
    except ImportError:
        from copulas import fit_copula

    w = validate_alpha_weights(alpha, weights)
    pi = float(w[0])
    frame = prepare_bivariate_returns(returns, n_insample, n_oos)
    assets = list(frame.columns)
    values, dates = [], []

    for i, date, window in rolling_windows(frame, n_insample, n_oos):
        fits = [_fit_arch_garch_11(window[a]) for a in assets]
        forecasts = [_garch_one_step_forecast(fit) for fit in fits]
        mu1, sigma1 = forecasts[0]
        mu2, sigma2 = forecasts[1]

        pits = []
        for fit, asset in zip(fits, assets):
            std = fit.std_resid.dropna()
            pits.append(pd.Series(norm.cdf(std), index=std.index, name=asset).clip(EPS, 1.0 - EPS))
        pit = pd.concat(pits, axis=1).dropna()
        cop_fit = fit_copula(pit, copula=copula, margin_model="GARCH")

        def inv1(u):
            return mu1 + sigma1 * norm.ppf(_clip_unit(u))

        def inv2(u):
            return mu2 + sigma2 * norm.ppf(_clip_unit(u))

        def cdf1(x):
            return norm.cdf((np.asarray(x, dtype=float) - mu1) / sigma1)

        var_t = solve_portfolio_var(
            alpha=alpha,
            inverse_cdf_1=inv1,
            inverse_cdf_2=inv2,
            cdf_1=cdf1,
            copula_params=cop_fit.params,
            copula=copula,
            pi=pi,
            integration_nodes=integration_nodes,
            root_tol=root_tol,
        )
        values.append(var_t)
        dates.append(date)

        if verbose and (i + 1) % 25 == 0:
            print(f"GARCH-{copula} alpha={alpha:g}: {i + 1}/{n_oos}, VaR={var_t:.4f}")

    return pd.Series(values, index=dates, name=f"CopulaGARCH_{copula}_VaR_{alpha:g}")


# -----------------------------------------------------------------------------
# Convenience wrappers and backtest inputs
# -----------------------------------------------------------------------------


def portfolio_returns(returns: pd.DataFrame, weights: Iterable[float] = (0.5, 0.5)) -> pd.Series:
    w = validate_alpha_weights(0.05, weights)
    frame = returns.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    out = frame @ w
    out.name = "portfolio_return"
    return out


def var_exceedances(realized_portfolio_returns: pd.Series, var_forecasts: pd.Series) -> pd.Series:
    """VaR hit sequence I_t = 1{r_p,t < VaR_t}."""
    aligned = pd.concat([realized_portfolio_returns.rename("r"), var_forecasts.rename("VaR")], axis=1).dropna()
    return aligned["r"].lt(aligned["VaR"]).astype(int).rename("hit")


def violation_frequency(realized_portfolio_returns: pd.Series, var_forecasts: pd.Series) -> float:
    return float(var_exceedances(realized_portfolio_returns, var_forecasts).mean())


def forecast_all_var_models(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    weights: Iterable[float] = (0.5, 0.5),
    n_insample: int = 1135,
    n_oos: int = 500,
    copulas: Iterable[str] = SUPPORTED_COPULAS,
    include_msm: bool = True,
    include_garch_copula: bool = True,
    msm_k: int = 5,
    msm_n_starts: int = 10,
    integration_nodes: int = 501,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute the full paper-like VaR panel for one alpha.

    This is computationally expensive, especially Copula-MSM rolling.
    """
    results: dict[str, pd.Series] = {}

    results["Historical"] = forecast_historical_var_rolling(returns, alpha, weights, n_insample, n_oos)
    results["RiskMetrics"] = forecast_riskmetrics_var_rolling(returns, alpha, weights, 0.94, n_insample, n_oos)
    results["Var-Covar"] = forecast_variance_covariance_var_rolling(returns, alpha, weights, n_insample, n_oos)
    results["CCC-GARCH"] = forecast_ccc_garch_var_rolling(returns, alpha, weights, n_insample, n_oos, verbose=verbose)

    if include_garch_copula:
        for copula in copulas:
            results[f"Copula-GARCH {copula}"] = forecast_garch_copula_var_rolling(
                returns=returns,
                copula=copula,
                alpha=alpha,
                weights=weights,
                n_insample=n_insample,
                n_oos=n_oos,
                integration_nodes=integration_nodes,
                verbose=verbose,
            )

    if include_msm:
        for copula in copulas:
            results[f"Copula-MSM {copula}"] = forecast_msm_copula_var_rolling(
                returns=returns,
                copula=copula,
                alpha=alpha,
                weights=weights,
                k=msm_k,
                n_insample=n_insample,
                n_oos=n_oos,
                n_starts=msm_n_starts,
                integration_nodes=integration_nodes,
                verbose=verbose,
            )

    return pd.concat(results, axis=1)


# ============================================================
# VaR Loss
# ============================================================

def var_loss_series(r, var, alpha):
    """
    VaRl_t(alpha) = (alpha - I_t) * (r_t - VaR_t)
    avec I_t = 1(r_t < VaR_t).
    Plus la perte moyenne est faible, meilleur est le modèle.
    """
    aligned = pd.concat([r, var], axis=1).dropna()
    rt = aligned.iloc[:, 0]
    vt = aligned.iloc[:, 1]
    hit = (rt < vt).astype(float)
    loss = (alpha - hit) * (rt - vt)
    return loss


def smooth_var_loss_series(r, var, alpha, nu=25.0):
    """
    SVaRl_t(alpha) = [alpha - h_nu(r_t, VaR_t)] * (r_t - VaR_t)
    h_nu(a,b) = [1 + exp(nu(a-b))]^{-1}
    """
    aligned = pd.concat([r, var], axis=1).dropna()
    rt = aligned.iloc[:, 0]
    vt = aligned.iloc[:, 1]

    x = np.clip(nu * (rt - vt), -700, 700)
    h = 1.0 / (1.0 + np.exp(x))

    loss = (alpha - h) * (rt - vt)
    return loss


def build_loss_panel(portfolio_returns, var_panel, alpha, smooth=False, nu=25.0):
    losses = {}

    for model in var_panel.columns:
        if smooth:
            losses[model] = smooth_var_loss_series(
                portfolio_returns,
                var_panel[model],
                alpha=alpha,
                nu=nu,
            )
        else:
            losses[model] = var_loss_series(
                portfolio_returns,
                var_panel[model],
                alpha=alpha,
            )

    return pd.concat(losses, axis=1).dropna(how="any")


