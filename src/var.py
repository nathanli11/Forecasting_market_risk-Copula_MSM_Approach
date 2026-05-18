"""Portfolio VaR forecasting for copula-MSM models.

This module follows the paper's convention:
    VaR_t(alpha) is a return quantile, usually negative.

That is:
    P(r_p,t <= VaR_t(alpha) | Omega_{t-1}) = alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm, t


EPS = 1e-10


def forecast_msm_copula_var_oos_fixed_params(
    msm_1,
    msm_2,
    returns_1: pd.Series,
    returns_2: pd.Series,
    copula_params: dict[str, float],
    copula: str = "student",
    pi: float = 0.5,
    alpha: float = 0.05,
    n_oos: int = 500,
    integration_nodes: int = 501,
    root_tol: float = 1e-4,
    verbose: bool = True,
) -> pd.Series:
    """One-step-ahead VaR forecasts with fixed MSM and copula parameters.

    The output follows the paper convention:
        VaR_t(alpha) = alpha-quantile of portfolio returns.

    Therefore VaR values are usually negative.

    This is still a fixed-parameter approximation:
        - MSM parameters are fixed.
        - copula parameters are fixed.
        - only MSM predictive state probabilities evolve over time.
    """
    try:
        from src.msm import make_msm_states, msm_filter_from_result
    except ImportError:
        from msm import make_msm_states, msm_filter_from_result

    _validate_var_inputs(alpha=alpha, pi=pi)

    if n_oos <= 0:
        raise ValueError("n_oos must be positive.")

    series_1 = pd.to_numeric(returns_1, errors="coerce").dropna()
    series_2 = pd.to_numeric(returns_2, errors="coerce").dropna()

    common_index = series_1.index.intersection(series_2.index)
    series_1 = series_1.loc[common_index]
    series_2 = series_2.loc[common_index]

    if n_oos >= len(common_index):
        raise ValueError("n_oos must be smaller than the sample size.")

    filter_1 = msm_filter_from_result(series_1, msm_1)
    filter_2 = msm_filter_from_result(series_2, msm_2)

    predicted_probs_1 = filter_1["predicted_probs"].to_numpy(dtype=float)
    predicted_probs_2 = filter_2["predicted_probs"].to_numpy(dtype=float)

    states_1 = make_msm_states(k=msm_1.k, m0=msm_1.params.m0)
    states_2 = make_msm_states(k=msm_2.k, m0=msm_2.params.m0)

    h_1 = np.sqrt(np.prod(states_1, axis=1))
    h_2 = np.sqrt(np.prod(states_2, axis=1))

    t_start = len(common_index) - n_oos
    forecast_index = common_index[t_start:]

    var_values = np.empty(n_oos, dtype=float)

    for i, _date in enumerate(forecast_index):
        t_idx = t_start + i

        var_values[i] = portfolio_var_root(
            state_probs_1=predicted_probs_1[t_idx],
            state_probs_2=predicted_probs_2[t_idx],
            sigma_1=msm_1.params.sigma,
            sigma_2=msm_2.params.sigma,
            h_1=h_1,
            h_2=h_2,
            copula_params=copula_params,
            mean_1=msm_1.mean_return,
            mean_2=msm_2.mean_return,
            copula=copula,
            pi=pi,
            alpha=alpha,
            integration_nodes=integration_nodes,
            root_tol=root_tol,
        )

        if verbose and (i + 1) % 50 == 0:
            print(
                f"  [{i + 1}/{n_oos}] "
                f"VaR {100 * alpha:.1f}% = {var_values[i]:.4f}"
            )

    return pd.Series(
        var_values,
        index=forecast_index,
        name=f"VaR_{int(alpha * 100)}_return",
    )


def forecast_msm_copula_var_oos_rolling(
    returns_1: pd.Series,
    returns_2: pd.Series,
    k_1: int,
    k_2: int,
    copula: str = "student",
    pi: float = 0.5,
    alpha: float = 0.05,
    n_insample: int = 1135,
    n_oos: int = 500,
    n_starts: int = 5,
    seed: int = 123,
    integration_nodes: int = 501,
    root_tol: float = 1e-4,
    verbose: bool = True,
) -> pd.Series:
    """True rolling VaR forecasts: MSM and copula re-estimated at each step.

    At each out-of-sample date t, re-estimates MSM (both assets) and the
    copula on the rolling window [t - n_insample, t - 1], then computes the
    one-step-ahead VaR.  This matches the rolling scheme described in the
    paper (Segnon & Trede, 2023, Section 4).

    Warning: this calls fit_msm twice and fit_copula once per step.
    With n_oos=500 this runs ~1000 MSM optimisations -- expect long runtime.
    Reduce n_starts to trade accuracy for speed.
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

    _validate_var_inputs(alpha=alpha, pi=pi)

    if n_oos <= 0:
        raise ValueError("n_oos must be positive.")
    if n_insample <= 0:
        raise ValueError("n_insample must be positive.")

    series_1 = pd.to_numeric(returns_1, errors="coerce").dropna()
    series_2 = pd.to_numeric(returns_2, errors="coerce").dropna()

    common_index = series_1.index.intersection(series_2.index)
    series_1 = series_1.loc[common_index]
    series_2 = series_2.loc[common_index]

    required = n_insample + n_oos
    if len(common_index) < required:
        raise ValueError(
            f"Need at least n_insample + n_oos = {required} observations, "
            f"got {len(common_index)}."
        )

    t_start = len(common_index) - n_oos
    forecast_index = common_index[t_start:]
    var_values = np.empty(n_oos, dtype=float)

    for i in range(n_oos):
        window_1 = series_1.iloc[i : t_start + i]
        window_2 = series_2.iloc[i : t_start + i]

        msm_1_i = fit_msm(window_1, k=k_1, n_starts=n_starts, seed=seed + i, verbose=False)
        msm_2_i = fit_msm(window_2, k=k_2, n_starts=n_starts, seed=seed + i, verbose=False)

        filter_1_i = msm_filter_from_result(window_1, msm_1_i)
        filter_2_i = msm_filter_from_result(window_2, msm_2_i)

        # One-step-ahead state probabilities for the forecast date:
        # P(s_t | r_{i}, ..., r_{t-1}) = filtered_probs[-1] @ A
        gammas_1 = renewal_probabilities_from_gamma_k(k=k_1, b=msm_1_i.params.b, gamma_k=msm_1_i.params.gamma_k)
        gammas_2 = renewal_probabilities_from_gamma_k(k=k_2, b=msm_2_i.params.b, gamma_k=msm_2_i.params.gamma_k)
        A_1 = transition_matrix_from_gammas(gammas_1)
        A_2 = transition_matrix_from_gammas(gammas_2)

        pred_probs_1 = filter_1_i["filtered_probs"].iloc[-1].to_numpy(dtype=float) @ A_1
        pred_probs_2 = filter_2_i["filtered_probs"].iloc[-1].to_numpy(dtype=float) @ A_2

        # Fit copula on rolling-window PIT values
        pit_1 = filter_1_i["pit"].to_numpy(dtype=float)
        pit_2 = filter_2_i["pit"].to_numpy(dtype=float)
        uniforms_i = pd.DataFrame({"asset_1": pit_1, "asset_2": pit_2})
        copula_result_i = fit_copula(uniforms_i, copula=copula, margin_model="msm")
        copula_params_i = copula_result_i.params

        states_1 = make_msm_states(k=k_1, m0=msm_1_i.params.m0)
        states_2 = make_msm_states(k=k_2, m0=msm_2_i.params.m0)
        h_1 = np.sqrt(np.prod(states_1, axis=1))
        h_2 = np.sqrt(np.prod(states_2, axis=1))

        var_values[i] = portfolio_var_root(
            state_probs_1=pred_probs_1,
            state_probs_2=pred_probs_2,
            sigma_1=msm_1_i.params.sigma,
            sigma_2=msm_2_i.params.sigma,
            h_1=h_1,
            h_2=h_2,
            copula_params=copula_params_i,
            mean_1=msm_1_i.mean_return,
            mean_2=msm_2_i.mean_return,
            copula=copula,
            pi=pi,
            alpha=alpha,
            integration_nodes=integration_nodes,
            root_tol=root_tol,
        )

        if verbose and (i + 1) % 10 == 0:
            print(
                f"  [{i + 1}/{n_oos}] "
                f"VaR {100 * alpha:.1f}% = {var_values[i]:.4f}"
            )

    return pd.Series(
        var_values,
        index=forecast_index,
        name=f"VaR_{int(alpha * 100)}_return",
    )


def portfolio_var_root(
    state_probs_1: np.ndarray,
    state_probs_2: np.ndarray,
    sigma_1: float,
    sigma_2: float,
    h_1: np.ndarray,
    h_2: np.ndarray,
    copula_params: dict[str, float],
    mean_1: float = 0.0,
    mean_2: float = 0.0,
    copula: str = "student",
    pi: float = 0.5,
    alpha: float = 0.05,
    integration_nodes: int = 501,
    root_tol: float = 1e-4,
) -> float:
    """Solve the paper's VaR equation numerically.

    Finds q such that:
        P(r_p <= q | Omega_{t-1}) - alpha = 0.

    Returns
    -------
    float
        Signed return VaR, usually negative.
    """
    _validate_var_inputs(alpha=alpha, pi=pi)

    if integration_nodes < 51:
        raise ValueError("integration_nodes should be at least 51.")

    if pi == 1.0:
        return float(
            mean_1
            + msm_conditional_quantile(
                np.array([alpha]),
                state_probs_1,
                sigma_1,
                h_1,
            )[0]
        )

    if pi == 0.0:
        return float(
            mean_2
            + msm_conditional_quantile(
                np.array([alpha]),
                state_probs_2,
                sigma_2,
                h_2,
            )[0]
        )

    lower, upper = _portfolio_var_bracket(
        state_probs_1=state_probs_1,
        state_probs_2=state_probs_2,
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        h_1=h_1,
        h_2=h_2,
        mean_1=mean_1,
        mean_2=mean_2,
        pi=pi,
    )

    def objective(q: float) -> float:
        return (
            portfolio_cdf_msm_copula(
                q=q,
                state_probs_1=state_probs_1,
                state_probs_2=state_probs_2,
                sigma_1=sigma_1,
                sigma_2=sigma_2,
                h_1=h_1,
                h_2=h_2,
                copula_params=copula_params,
                mean_1=mean_1,
                mean_2=mean_2,
                copula=copula,
                pi=pi,
                integration_nodes=integration_nodes,
            )
            - alpha
        )

    f_lower = objective(lower)
    f_upper = objective(upper)

    expansion = 1
    while not (f_lower <= 0.0 <= f_upper):
        width = upper - lower
        lower -= width
        upper += width
        f_lower = objective(lower)
        f_upper = objective(upper)
        expansion += 1

        if expansion > 10:
            raise RuntimeError(
                "Could not bracket the VaR root. "
                f"Last bracket=({lower}, {upper}), "
                f"objective=({f_lower}, {f_upper})."
            )

    return float(
        brentq(
            objective,
            lower,
            upper,
            xtol=root_tol,
            rtol=1e-6,
            maxiter=100,
        )
    )


def portfolio_cdf_msm_copula(
    q: float,
    state_probs_1: np.ndarray,
    state_probs_2: np.ndarray,
    sigma_1: float,
    sigma_2: float,
    h_1: np.ndarray,
    h_2: np.ndarray,
    copula_params: dict[str, float],
    mean_1: float = 0.0,
    mean_2: float = 0.0,
    copula: str = "student",
    pi: float = 0.5,
    integration_nodes: int = 501,
) -> float:
    """Numerically evaluate P(r_p <= q | Omega_{t-1}).

    This is a one-dimensional transformation of the paper's double integral.
    We integrate over u2 in [0, 1]:

        int P(U1 <= a(q, u2) | U2 = u2) du2.

    where:
        r2(u2) = mean_2 + F_2^{-1}(u2),
        threshold_r1 = [q - (1 - pi) r2(u2)] / pi,
        a(q, u2) = F_1(threshold_r1 - mean_1).
    """
    _validate_var_inputs(alpha=0.05, pi=pi)

    if not 0.0 < pi < 1.0:
        raise ValueError("portfolio_cdf_msm_copula requires pi in (0, 1).")

    nodes, weights = _unit_interval_gauss_legendre(integration_nodes)

    y2 = msm_conditional_quantile(
        u=nodes,
        state_probs=state_probs_2,
        sigma=sigma_2,
        h=h_2,
    )

    r2 = mean_2 + y2

    threshold_r1 = (q - (1.0 - pi) * r2) / pi
    threshold_y1 = threshold_r1 - mean_1

    u1_threshold = msm_conditional_cdf(
        y=threshold_y1,
        state_probs=state_probs_1,
        sigma=sigma_1,
        h=h_1,
    )

    conditional_probability = copula_conditional_cdf_u1_given_u2(
        u1=u1_threshold,
        u2=nodes,
        copula_params=copula_params,
        copula=copula,
    )

    return float(np.sum(weights * conditional_probability))


def msm_conditional_cdf(
    y: np.ndarray | float,
    state_probs: np.ndarray,
    sigma: float,
    h: np.ndarray,
) -> np.ndarray:
    """MSM conditional CDF for centered returns.

    F(y | Omega_{t-1}) = sum_j p_j Phi(y / (sigma h_j)).
    """
    y_values = np.asarray(y, dtype=float)

    probabilities = _normalize_probabilities(state_probs)
    h_values = _validate_h_sigma(h=h, sigma=sigma)

    cdf = np.sum(
        probabilities[:, None]
        * norm.cdf(y_values.reshape(1, -1) / (sigma * h_values[:, None])),
        axis=0,
    )

    return np.clip(cdf, EPS, 1.0 - EPS)


def msm_conditional_quantile(
    u: np.ndarray | float,
    state_probs: np.ndarray,
    sigma: float,
    h: np.ndarray,
    grid_size: int = 20_001,
    tail_std_multiplier: float = 9.0,
) -> np.ndarray:
    """Invert the MSM conditional CDF by interpolation."""
    uniforms = np.asarray(u, dtype=float)
    uniforms = np.clip(uniforms, EPS, 1.0 - EPS)

    probabilities = _normalize_probabilities(state_probs)
    h_values = _validate_h_sigma(h=h, sigma=sigma)

    max_scale = sigma * float(np.max(h_values))
    lower = -tail_std_multiplier * max_scale
    upper = tail_std_multiplier * max_scale

    y_grid = np.linspace(lower, upper, grid_size)

    cdf_grid = np.sum(
        probabilities[:, None]
        * norm.cdf(y_grid[None, :] / (sigma * h_values[:, None])),
        axis=0,
    )

    cdf_grid = np.maximum.accumulate(cdf_grid)
    cdf_grid = np.clip(cdf_grid, EPS, 1.0 - EPS)

    return np.interp(uniforms, cdf_grid, y_grid)


def copula_conditional_cdf_u1_given_u2(
    u1: np.ndarray,
    u2: np.ndarray,
    copula_params: dict[str, float],
    copula: str = "student",
) -> np.ndarray:
    """Compute P(U1 <= u1 | U2 = u2) for Gaussian or Student copula."""
    copula = copula.lower()

    u1 = np.clip(np.asarray(u1, dtype=float), EPS, 1.0 - EPS)
    u2 = np.clip(np.asarray(u2, dtype=float), EPS, 1.0 - EPS)

    rho = float(copula_params["rho"])

    if not -0.999 < rho < 0.999:
        raise ValueError("rho must be in (-0.999, 0.999).")

    if copula == "gaussian":
        z1 = norm.ppf(u1)
        z2 = norm.ppf(u2)
        denominator = np.sqrt(1.0 - rho**2)
        return norm.cdf((z1 - rho * z2) / denominator)

    if copula == "student":
        nu = float(copula_params["nu"])

        if nu <= 2.0:
            raise ValueError("nu must be greater than 2 for Student copula.")

        x1 = t.ppf(u1, df=nu)
        x2 = t.ppf(u2, df=nu)

        conditional_df = nu + 1.0
        conditional_mean = rho * x2
        conditional_scale = np.sqrt(
            ((nu + x2**2) * (1.0 - rho**2)) / (nu + 1.0)
        )

        return t.cdf(
            (x1 - conditional_mean) / conditional_scale,
            df=conditional_df,
        )

    raise ValueError("Unsupported copula. Use 'student' or 'gaussian'.")


def portfolio_var_mc(
    state_probs_1: np.ndarray,
    state_probs_2: np.ndarray,
    sigma_1: float,
    sigma_2: float,
    h_1: np.ndarray,
    h_2: np.ndarray,
    copula_params: dict[str, float],
    mean_1: float = 0.0,
    mean_2: float = 0.0,
    copula: str = "student",
    pi: float = 0.5,
    alpha: float = 0.05,
    n_samples: int = 50_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo fallback returning signed return VaR.

    This is not the main paper-like method. It is useful for checking that the
    root-based numerical VaR is reasonable.
    """
    _validate_var_inputs(alpha=alpha, pi=pi)

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rng = np.random.default_rng(seed)
    u1, u2 = _sample_bivariate_copula(
        copula_params=copula_params,
        copula=copula,
        n_samples=n_samples,
        rng=rng,
    )

    y1 = msm_conditional_quantile(
        u=u1,
        state_probs=state_probs_1,
        sigma=sigma_1,
        h=h_1,
    )

    y2 = msm_conditional_quantile(
        u=u2,
        state_probs=state_probs_2,
        sigma=sigma_2,
        h=h_2,
    )

    r1 = mean_1 + y1
    r2 = mean_2 + y2
    rp = pi * r1 + (1.0 - pi) * r2

    return float(np.quantile(rp, alpha))


def _sample_bivariate_copula(
    copula_params: dict[str, float],
    copula: str,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample uniforms from Gaussian or Student copula."""
    copula = copula.lower()
    rho = float(copula_params["rho"])

    if not -0.999 < rho < 0.999:
        raise ValueError("rho must be in (-0.999, 0.999).")

    corr = np.array([[1.0, rho], [rho, 1.0]])

    if copula == "gaussian":
        z = rng.multivariate_normal(
            mean=np.zeros(2),
            cov=corr,
            size=n_samples,
        )
        return norm.cdf(z[:, 0]), norm.cdf(z[:, 1])

    if copula == "student":
        nu = float(copula_params["nu"])

        if nu <= 2.0:
            raise ValueError("nu must be greater than 2 for Student copula.")

        z = rng.multivariate_normal(
            mean=np.zeros(2),
            cov=corr,
            size=n_samples,
        )
        chi2 = rng.chisquare(df=nu, size=n_samples)
        x = z * np.sqrt(nu / chi2[:, None])

        return t.cdf(x[:, 0], df=nu), t.cdf(x[:, 1], df=nu)

    raise ValueError("Unsupported copula. Use 'student' or 'gaussian'.")


def _portfolio_var_bracket(
    state_probs_1: np.ndarray,
    state_probs_2: np.ndarray,
    sigma_1: float,
    sigma_2: float,
    h_1: np.ndarray,
    h_2: np.ndarray,
    mean_1: float,
    mean_2: float,
    pi: float,
) -> tuple[float, float]:
    """Build a conservative bracket for the portfolio return quantile."""
    q_low = EPS
    q_high = 1.0 - EPS

    r1_low = mean_1 + msm_conditional_quantile(
        np.array([q_low]),
        state_probs_1,
        sigma_1,
        h_1,
    )[0]
    r1_high = mean_1 + msm_conditional_quantile(
        np.array([q_high]),
        state_probs_1,
        sigma_1,
        h_1,
    )[0]

    r2_low = mean_2 + msm_conditional_quantile(
        np.array([q_low]),
        state_probs_2,
        sigma_2,
        h_2,
    )[0]
    r2_high = mean_2 + msm_conditional_quantile(
        np.array([q_high]),
        state_probs_2,
        sigma_2,
        h_2,
    )[0]

    lower = pi * r1_low + (1.0 - pi) * r2_low
    upper = pi * r1_high + (1.0 - pi) * r2_high

    return float(lower), float(upper)


def _unit_interval_gauss_legendre(
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on (0, 1)."""
    x, w = np.polynomial.legendre.leggauss(n_nodes)

    nodes = 0.5 * (x + 1.0)
    weights = 0.5 * w

    nodes = np.clip(nodes, EPS, 1.0 - EPS)

    return nodes, weights


def _normalize_probabilities(state_probs: np.ndarray) -> np.ndarray:
    """Validate and normalize state probabilities."""
    probabilities = np.asarray(state_probs, dtype=float)

    if probabilities.ndim != 1:
        raise ValueError("state_probs must be one-dimensional.")

    if np.any(probabilities < 0):
        raise ValueError("state_probs must be non-negative.")

    total = probabilities.sum()

    if total <= 0:
        raise ValueError("state_probs must sum to a positive value.")

    return probabilities / total


def _validate_h_sigma(h: np.ndarray, sigma: float) -> np.ndarray:
    """Validate MSM h values and sigma."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    h_values = np.asarray(h, dtype=float)

    if h_values.ndim != 1:
        raise ValueError("h must be one-dimensional.")

    if np.any(h_values <= 0):
        raise ValueError("All h values must be positive.")

    return h_values


def _validate_var_inputs(alpha: float, pi: float) -> None:
    """Validate VaR level and portfolio weight."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")

    if not 0.0 <= pi <= 1.0:
        raise ValueError("pi must be in [0, 1].")


def forecast_historical_var(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    weights: np.ndarray | None = None,
    window_size: int = 1135,
    n_oos: int = 500,
) -> pd.Series:
    """Rolling historical simulation VaR as a signed return quantile."""
    _validate_var_inputs(alpha=alpha, pi=0.5)

    frame = returns.apply(pd.to_numeric, errors="coerce").dropna(how="any")

    if weights is None:
        weights = np.repeat(1.0 / frame.shape[1], frame.shape[1])

    weights = np.asarray(weights, dtype=float)

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1.")

    portfolio = frame @ weights

    if window_size + n_oos > len(portfolio):
        raise ValueError("window_size + n_oos cannot exceed sample size.")

    start = len(portfolio) - n_oos
    values = []
    dates = portfolio.index[start:]

    for pos in range(start, len(portfolio)):
        window = portfolio.iloc[pos - window_size:pos]
        values.append(float(np.quantile(window, alpha)))

    return pd.Series(
        values,
        index=dates,
        name=f"Historical VaR {int(alpha * 100)}%",
    )

def forecast_variance_covariance_var(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    weights: np.ndarray | None = None,
    window_size: int = 1135,
    n_oos: int = 500,
    include_mean: bool = True,
) -> pd.Series:
    """Rolling variance-covariance VaR as a signed return quantile."""
    _validate_var_inputs(alpha=alpha, pi=0.5)

    frame = returns.apply(pd.to_numeric, errors="coerce").dropna(how="any")

    if weights is None:
        weights = np.repeat(1.0 / frame.shape[1], frame.shape[1])

    weights = np.asarray(weights, dtype=float)

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1.")

    if window_size + n_oos > len(frame):
        raise ValueError("window_size + n_oos cannot exceed sample size.")

    z_alpha = norm.ppf(alpha)

    start = len(frame) - n_oos
    values = []
    dates = frame.index[start:]

    for pos in range(start, len(frame)):
        window = frame.iloc[pos - window_size:pos]

        mu = window.mean().to_numpy(dtype=float) if include_mean else np.zeros(frame.shape[1])
        cov = window.cov().to_numpy(dtype=float)

        portfolio_mean = float(weights @ mu)
        portfolio_var = float(weights @ cov @ weights)
        portfolio_vol = np.sqrt(max(portfolio_var, 0.0))

        values.append(portfolio_mean + portfolio_vol * z_alpha)

    return pd.Series(
        values,
        index=dates,
        name=f"Covariance VaR {int(alpha * 100)}%",
    )


def forecast_riskmetrics_var(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    weights: np.ndarray | None = None,
    lambda_: float = 0.94,
    window_size: int = 1135,
    n_oos: int = 500,
    include_mean: bool = False,
) -> pd.Series:
    """RiskMetrics EWMA VaR as a signed return quantile."""
    _validate_var_inputs(alpha=alpha, pi=0.5)

    if not 0.0 < lambda_ < 1.0:
        raise ValueError("lambda_ must be in (0, 1).")

    frame = returns.apply(pd.to_numeric, errors="coerce").dropna(how="any")

    if weights is None:
        weights = np.repeat(1.0 / frame.shape[1], frame.shape[1])

    weights = np.asarray(weights, dtype=float)

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1.")

    if window_size + n_oos > len(frame):
        raise ValueError("window_size + n_oos cannot exceed sample size.")

    z_alpha = norm.ppf(alpha)

    start = len(frame) - n_oos
    dates = frame.index[start:]

    initial_window = frame.iloc[start - window_size:start]
    sigma = initial_window.cov().to_numpy(dtype=float)

    values = []

    for pos in range(start, len(frame)):
        previous_return = frame.iloc[pos - 1].to_numpy(dtype=float).reshape(-1, 1)

        sigma = (
            lambda_ * sigma
            + (1.0 - lambda_) * (previous_return @ previous_return.T)
        )

        mu = (
            frame.iloc[pos - window_size:pos].mean().to_numpy(dtype=float)
            if include_mean
            else np.zeros(frame.shape[1])
        )

        portfolio_mean = float(weights @ mu)
        portfolio_var = float(weights @ sigma @ weights)
        portfolio_vol = np.sqrt(max(portfolio_var, 0.0))

        values.append(portfolio_mean + portfolio_vol * z_alpha)

    return pd.Series(
        values,
        index=dates,
        name=f"RiskMetrics VaR {int(alpha * 100)}%",
    )


def forecast_ccc_garch_var_fixed_params(
    returns: pd.DataFrame,
    garch_results: dict,
    alpha: float = 0.05,
    weights: np.ndarray | None = None,
    n_oos: int = 500,
    include_mean: bool = True,
) -> pd.Series:
    """CCC-GARCH VaR with fixed univariate GARCH parameters.

    VaR is returned as a signed return quantile.
    """
    _validate_var_inputs(alpha=alpha, pi=0.5)

    frame = returns.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    assets = list(frame.columns)

    if len(assets) != 2:
        raise ValueError("CCC-GARCH helper currently expects exactly two assets.")

    if weights is None:
        weights = np.array([0.5, 0.5])

    weights = np.asarray(weights, dtype=float)

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1.")

    z_alpha = norm.ppf(alpha)

    standardized = []
    volatilities = []
    means = []

    for asset in assets:
        result = garch_results[asset]
        model_result = result.model_result

        std_resid = model_result.std_resid.dropna()
        volatility = model_result.conditional_volatility.dropna()

        standardized.append(std_resid.rename(asset))
        volatilities.append(volatility.rename(asset))
        means.append(result.mean if include_mean else 0.0)

    standardized_frame = pd.concat(standardized, axis=1).dropna(how="any")
    rho = float(standardized_frame.corr().iloc[0, 1])

    corr = np.array(
        [
            [1.0, rho],
            [rho, 1.0],
        ]
    )

    volatility_frame = pd.concat(volatilities, axis=1).dropna(how="any")

    common_index = frame.index.intersection(volatility_frame.index)
    common_index = common_index[-n_oos:]

    values = []

    for date in common_index:
        sigma_values = volatility_frame.loc[date, assets].to_numpy(dtype=float)
        d_t = np.diag(sigma_values)

        covariance_t = d_t @ corr @ d_t

        mu = np.asarray(means, dtype=float)
        portfolio_mean = float(weights @ mu)
        portfolio_var = float(weights @ covariance_t @ weights)
        portfolio_vol = np.sqrt(max(portfolio_var, 0.0))

        values.append(portfolio_mean + portfolio_vol * z_alpha)

    return pd.Series(
        values,
        index=common_index,
        name=f"CCC-GARCH VaR {int(alpha * 100)}%",
    )


def portfolio_cdf_garch_copula(
    q: float,
    mu_1: float,
    mu_2: float,
    sigma_1: float,
    sigma_2: float,
    copula_params: dict[str, float],
    copula: str = "student",
    pi: float = 0.5,
    integration_nodes: int = 501,
) -> float:
    """Evaluate P(r_p <= q) for normal GARCH margins and Gaussian/Student copula."""
    _validate_var_inputs(alpha=0.05, pi=pi)

    if not 0.0 < pi < 1.0:
        raise ValueError("pi must be in (0, 1).")

    nodes, weights = _unit_interval_gauss_legendre(integration_nodes)

    z2 = norm.ppf(nodes)
    r2 = mu_2 + sigma_2 * z2

    threshold_r1 = (q - (1.0 - pi) * r2) / pi
    u1_threshold = norm.cdf((threshold_r1 - mu_1) / sigma_1)

    conditional_probability = copula_conditional_cdf_u1_given_u2(
        u1=u1_threshold,
        u2=nodes,
        copula_params=copula_params,
        copula=copula,
    )

    return float(np.sum(weights * conditional_probability))


def portfolio_var_garch_copula_root(
    mu_1: float,
    mu_2: float,
    sigma_1: float,
    sigma_2: float,
    copula_params: dict[str, float],
    copula: str = "student",
    pi: float = 0.5,
    alpha: float = 0.05,
    integration_nodes: int = 501,
    root_tol: float = 1e-4,
) -> float:
    """Solve VaR equation for normal GARCH margins and fitted copula."""
    _validate_var_inputs(alpha=alpha, pi=pi)

    if sigma_1 <= 0 or sigma_2 <= 0:
        raise ValueError("GARCH volatilities must be positive.")

    if pi == 1.0:
        return float(mu_1 + sigma_1 * norm.ppf(alpha))

    if pi == 0.0:
        return float(mu_2 + sigma_2 * norm.ppf(alpha))

    max_sigma = max(sigma_1, sigma_2)
    center = pi * mu_1 + (1.0 - pi) * mu_2

    lower = center - 12.0 * max_sigma
    upper = center + 12.0 * max_sigma

    def objective(q: float) -> float:
        return (
            portfolio_cdf_garch_copula(
                q=q,
                mu_1=mu_1,
                mu_2=mu_2,
                sigma_1=sigma_1,
                sigma_2=sigma_2,
                copula_params=copula_params,
                copula=copula,
                pi=pi,
                integration_nodes=integration_nodes,
            )
            - alpha
        )

    return float(
        brentq(
            objective,
            lower,
            upper,
            xtol=root_tol,
            rtol=1e-6,
            maxiter=100,
        )
    )


def forecast_garch_copula_var_fixed_params(
    returns: pd.DataFrame,
    garch_results: dict,
    copula_params: dict[str, float],
    copula: str = "student",
    alpha: float = 0.05,
    weights: np.ndarray | None = None,
    n_oos: int = 500,
    integration_nodes: int = 501,
    root_tol: float = 1e-4,
) -> pd.Series:
    """Student-Copula-GARCH VaR with fixed GARCH and copula parameters."""
    _validate_var_inputs(alpha=alpha, pi=0.5)

    frame = returns.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    assets = list(frame.columns)

    if len(assets) != 2:
        raise ValueError("This helper expects exactly two assets.")

    if weights is None:
        weights = np.array([0.5, 0.5])

    weights = np.asarray(weights, dtype=float)

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1.")

    pi = float(weights[0])

    vol_frames = []
    means = []

    for asset in assets:
        result = garch_results[asset]
        vol = result.model_result.conditional_volatility.dropna().rename(asset)
        vol_frames.append(vol)
        means.append(float(result.mean))

    volatility_frame = pd.concat(vol_frames, axis=1).dropna(how="any")

    common_index = frame.index.intersection(volatility_frame.index)
    common_index = common_index[-n_oos:]

    values = []

    for date in common_index:
        sigma_1 = float(volatility_frame.loc[date, assets[0]])
        sigma_2 = float(volatility_frame.loc[date, assets[1]])

        values.append(
            portfolio_var_garch_copula_root(
                mu_1=means[0],
                mu_2=means[1],
                sigma_1=sigma_1,
                sigma_2=sigma_2,
                copula_params=copula_params,
                copula=copula,
                pi=pi,
                alpha=alpha,
                integration_nodes=integration_nodes,
                root_tol=root_tol,
            )
        )

    return pd.Series(
        values,
        index=common_index,
        name=f"Student-Copula-GARCH VaR {int(alpha * 100)}%",
    )