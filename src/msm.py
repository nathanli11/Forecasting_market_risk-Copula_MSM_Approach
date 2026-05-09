"""Markov Switching Multifractal volatility model."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm


@dataclass(frozen=True)
class MSMParams:
    """Estimated MSM parameters."""

    m0: float
    sigma: float
    b: float
    gamma_k: float
    gamma_1: float
    k: int


@dataclass(frozen=True)
class MSMFitResult:
    """Container for MSM estimation results."""

    asset: str
    k: int
    params: MSMParams
    log_likelihood: float
    success: bool
    message: str
    nobs: int
    mean: float


def make_msm_states(k: int, m0: float) -> np.ndarray:
    """Return all 2^k MSM multiplier states.

    Each component takes value m0 or 2 - m0.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    if not 0.0 < m0 < 2.0:
        raise ValueError("m0 must lie in (0, 2).")

    values = [m0, 2.0 - m0]
    states = np.array(list(product(values, repeat=k)), dtype=float)
    return states


def renewal_probabilities_from_gamma_k(
    k: int,
    b: float,
    gamma_k: float,
) -> np.ndarray:
    """Compute gamma_1,...,gamma_k from b and gamma_k.

    The paper reports gamma_k. We invert the usual MSM relation:

        gamma_i = 1 - (1 - gamma_1) ** (b ** (i - 1))

    using gamma_k.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    if b <= 1.0:
        raise ValueError("b must be > 1.")
    if not 0.0 < gamma_k < 1.0:
        raise ValueError("gamma_k must lie in (0, 1).")

    if k == 1:
        gamma_1 = gamma_k
    else:
        gamma_1 = 1.0 - (1.0 - gamma_k) ** (1.0 / (b ** (k - 1)))

    gammas = np.array(
        [
            1.0 - (1.0 - gamma_1) ** (b**i)
            for i in range(k)
        ],
        dtype=float,
    )

    return np.clip(gammas, 1e-10, 1.0 - 1e-10)


def transition_matrix_from_gammas(gammas: np.ndarray) -> np.ndarray:
    """Build the 2^k x 2^k MSM transition matrix.

    If a component is not renewed, it keeps its current value.
    If it is renewed, it draws either low or high with probability 1/2.
    Thus:
        P(next bit = old bit) = 1 - gamma_i / 2
        P(next bit = other bit) = gamma_i / 2
    """
    gammas = np.asarray(gammas, dtype=float)
    k = len(gammas)

    bit_states = np.array(list(product([0, 1], repeat=k)), dtype=int)
    n_states = bit_states.shape[0]

    transition = np.ones((n_states, n_states), dtype=float)

    for i, gamma_i in enumerate(gammas):
        same = bit_states[:, [i]] == bit_states[:, i][None, :]
        component_transition = np.where(
            same,
            1.0 - gamma_i / 2.0,
            gamma_i / 2.0,
        )
        transition *= component_transition

    # Numerical safety
    transition /= transition.sum(axis=1, keepdims=True)

    return transition


def msm_loglikelihood(
    y: pd.Series | np.ndarray,
    k: int,
    m0: float,
    sigma: float,
    b: float,
    gamma_k: float,
) -> float:
    """Compute MSM log-likelihood by Hamilton filtering."""
    values = _clean_centered_returns(y)

    if sigma <= 0.0:
        return -np.inf

    states = make_msm_states(k=k, m0=m0)
    gammas = renewal_probabilities_from_gamma_k(k=k, b=b, gamma_k=gamma_k)
    transition = transition_matrix_from_gammas(gammas)

    h = np.sqrt(np.prod(states, axis=1))
    state_sigmas = sigma * h

    if np.any(state_sigmas <= 0) or np.any(~np.isfinite(state_sigmas)):
        return -np.inf

    n_states = states.shape[0]
    # Initial distribution: equal probability for all states.
    predicted_probs = np.full(n_states, 1.0 / n_states)

    loglik = 0.0
    log_state_sigmas = np.log(state_sigmas)
    log_sqrt_2pi = 0.5 * np.log(2.0 * np.pi)

    for obs in values:
        # log_state_densities = norm.logpdf(
        #     obs,
        #     loc=0.0,
        #     scale=state_sigmas,
        # )
        z = obs / state_sigmas
        log_state_densities = (
            -log_sqrt_2pi
            - log_state_sigmas
            - 0.5 * z**2
        )

        log_joint = np.log(predicted_probs + 1e-300) + log_state_densities
        #log_density = logsumexp(log_joint)
        max_log_joint = np.max(log_joint)
        log_density = max_log_joint + np.log(np.exp(log_joint - max_log_joint).sum())

        if not np.isfinite(log_density):
            return -np.inf

        loglik += log_density

        # Update p_{t|t}
        filtered_probs = np.exp(log_joint - log_density)

        # Predict p_{t+1|t}
        #predicted_probs = predict_next_probabilities(filtered_probs, gammas)

        predicted_probs = filtered_probs @ transition
        predicted_probs = np.clip(predicted_probs, 1e-300, 1.0)
        predicted_probs /= predicted_probs.sum()

    return float(loglik)


def msm_filter(
    returns: pd.Series | np.ndarray,
    k: int,
    m0: float,
    sigma: float,
    b: float,
    gamma_k: float,
    mean: float = 0.0,
    clip_cdf: float = 1e-10,
) -> dict[str, pd.Series | pd.DataFrame | float]:
    """Run the MSM Hamilton filter and return conditional objects.

    Parameters
    ----------
    returns:
        Uncentered return series. If you already have centered returns, pass
        mean=0.0.
    k:
        Number of MSM volatility components.
    m0, sigma, b, gamma_k:
        MSM parameters.
    mean:
        Constant conditional mean used to center returns.
    clip_cdf:
        Numerical clipping level for PIT values.

    Returns
    -------
    dict
        Dictionary containing predicted probabilities, filtered probabilities,
        conditional densities, conditional CDFs, PIT values, conditional
        volatility and log-likelihood.
    """
    series = _as_series_with_index(returns)
    centered = series - mean
    values = centered.to_numpy(dtype=float)

    states = make_msm_states(k=k, m0=m0)
    gammas = renewal_probabilities_from_gamma_k(k=k, b=b, gamma_k=gamma_k)
    transition = transition_matrix_from_gammas(gammas)

    h = np.sqrt(np.prod(states, axis=1))
    state_sigmas = sigma * h

    if np.any(state_sigmas <= 0) or np.any(~np.isfinite(state_sigmas)):
        raise ValueError("Invalid MSM state standard deviations.")

    n_obs = values.shape[0]
    n_states = states.shape[0]

    predicted_probs = np.full(n_states, 1.0 / n_states)

    predicted_probs_store = np.empty((n_obs, n_states), dtype=float)
    filtered_probs_store = np.empty((n_obs, n_states), dtype=float)
    densities = np.empty(n_obs, dtype=float)
    cdfs = np.empty(n_obs, dtype=float)
    conditional_volatility = np.empty(n_obs, dtype=float)

    loglik = 0.0

    log_state_sigmas = np.log(state_sigmas)
    log_sqrt_2pi = 0.5 * np.log(2.0 * np.pi)

    for t, obs in enumerate(values):
        predicted_probs_store[t, :] = predicted_probs
        z = obs / state_sigmas
        log_state_densities = (
            -log_sqrt_2pi
            - log_state_sigmas
            - 0.5 * z**2
        )
        log_joint = np.log(predicted_probs + 1e-300) + log_state_densities
        max_log_joint = np.max(log_joint)
        log_density = max_log_joint + np.log(
            np.exp(log_joint - max_log_joint).sum()
        )
        density_t = float(np.exp(log_density))
        densities[t] = density_t
        loglik += float(log_density)
        # Conditional CDF:
        # F(y_t | Omega_{t-1}) = sum_j p_{t|t-1}(j) Phi(y_t / sigma_j)
        state_cdfs = norm.cdf(z)
        cdf_t = float(np.sum(predicted_probs * state_cdfs))
        cdfs[t] = np.clip(cdf_t, clip_cdf, 1.0 - clip_cdf)
        # Conditional volatility:
        # sqrt(E[sigma_t^2 | Omega_{t-1}])
        conditional_variance_t = float(
            np.sum(predicted_probs * state_sigmas**2)
        )
        conditional_volatility[t] = np.sqrt(conditional_variance_t)
        # Bayesian update:
        # p_{t|t}(j) = p_{t|t-1}(j) f_j(y_t) / f(y_t | Omega_{t-1})
        filtered_probs = np.exp(log_joint - log_density)
        filtered_probs_store[t, :] = filtered_probs
        # Prediction:
        # p_{t+1|t} = p_{t|t} P
        predicted_probs = filtered_probs @ transition
        predicted_probs = np.clip(predicted_probs, 1e-300, 1.0)
        predicted_probs /= predicted_probs.sum()

    state_columns = [f"state_{i}" for i in range(n_states)]

    return {
        "centered_returns": pd.Series(
            centered.to_numpy(),
            index=series.index,
            name=series.name,
        ),
        "predicted_probs": pd.DataFrame(
            predicted_probs_store,
            index=series.index,
            columns=state_columns,
        ),
        "filtered_probs": pd.DataFrame(
            filtered_probs_store,
            index=series.index,
            columns=state_columns,
        ),
        "densities": pd.Series(
            densities,
            index=series.index,
            name=series.name,
        ),
        "cdfs": pd.Series(
            cdfs,
            index=series.index,
            name=series.name,
        ),
        "pit": pd.Series(
            cdfs,
            index=series.index,
            name=series.name,
        ),
        "conditional_volatility": pd.Series(
            conditional_volatility,
            index=series.index,
            name=series.name,
        ),
        "log_likelihood": float(loglik),
        "states": pd.DataFrame(states, columns=[f"M_{i+1}" for i in range(k)]),
        "gammas": pd.Series(gammas, index=[f"gamma_{i+1}" for i in range(k)]),
        "state_sigmas": pd.Series(state_sigmas, index=state_columns),
    }


def msm_conditional_cdf(
    returns: pd.Series | np.ndarray,
    k: int,
    m0: float,
    sigma: float,
    b: float,
    gamma_k: float,
    mean: float = 0.0,
    clip_cdf: float = 1e-10,
) -> pd.Series:
    """Return the MSM conditional CDF values F(y_t | Omega_{t-1})."""
    filtered = msm_filter(
        returns=returns,
        k=k,
        m0=m0,
        sigma=sigma,
        b=b,
        gamma_k=gamma_k,
        mean=mean,
        clip_cdf=clip_cdf,
    )
    return filtered["cdfs"]


def msm_probability_integral_transform(
    returns: pd.Series,
    fit_result: MSMFitResult,
    clip_cdf: float = 1e-10,
) -> pd.Series:
    """Compute MSM PIT values from an estimated MSMFitResult.

    The output is:
        u_t = F_MSM(r_t - mu_hat | Omega_{t-1})
    """
    params = fit_result.params

    pit = msm_conditional_cdf(
        returns=returns,
        k=fit_result.k,
        m0=params.m0,
        sigma=params.sigma,
        b=params.b,
        gamma_k=params.gamma_k,
        mean=fit_result.mean,
        clip_cdf=clip_cdf,
    )

    pit.name = returns.name
    return pit


def msm_filter_from_result(
    returns: pd.Series,
    fit_result: MSMFitResult,
    clip_cdf: float = 1e-10,
) -> dict[str, pd.Series | pd.DataFrame | float]:
    """Run the MSM filter using a stored MSMFitResult."""
    params = fit_result.params

    return msm_filter(
        returns=returns,
        k=fit_result.k,
        m0=params.m0,
        sigma=params.sigma,
        b=params.b,
        gamma_k=params.gamma_k,
        mean=fit_result.mean,
        clip_cdf=clip_cdf,
    )


def build_msm_pit_frame(
    returns: pd.DataFrame,
    fit_results: dict[str, MSMFitResult],
    clip_cdf: float = 1e-10,
) -> pd.DataFrame:
    """Build a DataFrame of MSM PIT values for several assets."""
    pits = {}

    for asset in returns.columns:
        if asset not in fit_results:
            raise KeyError(f"Missing MSM fit result for asset: {asset}")

        pits[asset] = msm_probability_integral_transform(
            returns=returns[asset],
            fit_result=fit_results[asset],
            clip_cdf=clip_cdf,
        )

    pit_frame = pd.concat(pits, axis=1).dropna(how="any")
    pit_frame.index.name = returns.index.name or "date"

    return pit_frame


def _as_series_with_index(values: pd.Series | np.ndarray) -> pd.Series:
    """Convert array-like values to a clean Series while preserving index if possible."""
    if isinstance(values, pd.Series):
        series = pd.to_numeric(values, errors="coerce").dropna()
        return series.astype(float)

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if array.size == 0:
        raise ValueError("At least one observation is required.")

    return pd.Series(array, dtype=float)


def fit_msm(
    returns: pd.Series,
    k: int,
    n_starts: int = 20,
    seed: int = 123,
) -> MSMFitResult:
    """Estimate MSM parameters for one return series and one k."""
    series = pd.to_numeric(returns, errors="coerce").dropna()
    asset = str(series.name or "asset")

    # The MSM is estimated on centered percentage returns.
    y = series - series.mean()

    rng = np.random.default_rng(seed)

    best_result = None
    best_loglik = -np.inf

    bounds = [
        (0.05, 1.95),   # m0
        (1e-4, 10.0),   # sigma
        (1.0001, 50.0), # b
        (1e-5, 0.999),  # gamma_k
    ]

    initial_points = _initial_points_for_msm(
        y=y,
        k=k,
        n_starts=n_starts,
        rng=rng,
    )

    # for x0 in initial_points:
    #     opt = minimize(
    #         _negative_loglikelihood,
    #         x0=x0,
    #         args=(y.to_numpy(dtype=float), k),
    #         method="Nelder-Mead",
    #         options={
    #             "maxiter": 5000,
    #             "xatol": 1e-5,
    #             "fatol": 1e-5,
    #         },
    #     )
    #     # Nelder-Mead does not enforce bounds, so evaluate only valid points.
    #     x = opt.x
    #     if not _params_in_bounds(x, bounds):
    #         continue
    #     loglik = -_negative_loglikelihood(x, y.to_numpy(dtype=float), k)
    #     # Optional local refinement with bounds
    #     opt_bounded = minimize(
    #         _negative_loglikelihood,
    #         x0=x,
    #         args=(y.to_numpy(dtype=float), k),
    #         method="L-BFGS-B",
    #         bounds=bounds,
    #         options={
    #             "maxiter": 2000,
    #             "ftol": 1e-8,
    #         },
    #     )
    #     if opt_bounded.success:
    #         x = opt_bounded.x
    #         loglik = -float(opt_bounded.fun)
    #         opt = opt_bounded

    #     if np.isfinite(loglik) and loglik > best_loglik:
    #         best_loglik = loglik
    #         best_result = opt

    y_array = y.to_numpy(dtype=float)
    for start_id, x0 in enumerate(initial_points, start=1):
        print(f"  start {start_id}/{len(initial_points)}: x0={x0}")
        opt = minimize(
            _negative_loglikelihood,
            x0=x0,
            args=(y_array, k),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 300,
                "maxls": 20,
                "ftol": 1e-6,
                "gtol": 1e-5,
            },
        )
        loglik = -float(opt.fun)
        print(
            f"    success={opt.success}, "
            f"loglik={loglik:.3f}, "
            f"nit={getattr(opt, 'nit', None)}, "
            f"nfev={getattr(opt, 'nfev', None)}"
        )
        if np.isfinite(loglik) and loglik > best_loglik:
            best_loglik = loglik
            best_result = opt

    if best_result is None:
        raise RuntimeError(f"MSM estimation failed for {asset}, k={k}.")

    m0, sigma, b, gamma_k = best_result.x
    gammas = renewal_probabilities_from_gamma_k(k=k, b=b, gamma_k=gamma_k)

    params = MSMParams(
        m0=float(m0),
        sigma=float(sigma),
        b=float(b),
        gamma_k=float(gamma_k),
        gamma_1=float(gammas[0]),
        k=k,
    )

    return MSMFitResult(
        asset=asset,
        k=k,
        params=params,
        log_likelihood=float(best_loglik),
        success=bool(best_result.success),
        message=str(best_result.message),
        nobs=int(series.shape[0]),
        mean=float(series.mean()),
    )


def fit_msm_grid(
    returns: pd.DataFrame,
    k_values: range | list[int] = range(1, 8),
    n_starts: int = 20,
    seed: int = 123,
) -> pd.DataFrame:
    """Estimate MSM for each asset and each k, then return a comparison table."""
    rows = []

    for asset in returns.columns:
        for k in k_values:
            print(f"Estimating MSM: asset={asset}, k={k}, n_starts={n_starts}")
            result = fit_msm(
                returns=returns[asset],
                k=int(k),
                n_starts=n_starts,
                seed=seed + 1000 * int(k),
            )

            rows.append(
                {
                    "asset": result.asset,
                    "k": result.k,
                    "mean": result.mean,
                    "m0": result.params.m0,
                    "sigma": result.params.sigma,
                    "b": result.params.b,
                    "gamma_1": result.params.gamma_1,
                    "gamma_k": result.params.gamma_k,
                    "log_likelihood": result.log_likelihood,
                    "success": result.success,
                    "message": result.message,
                    "nobs": result.nobs,
                }
            )

    return pd.DataFrame(rows)


def predict_next_probabilities(
    filtered_probs: np.ndarray,
    gammas: np.ndarray,
) -> np.ndarray:
    """Predict next MSM state probabilities without building the full transition matrix.

    This is mathematically equivalent to multiplying by the 2^k x 2^k transition
    matrix, but much faster because the transition is a Kronecker product of
    k independent 2 x 2 transitions.
    """
    gammas = np.asarray(gammas, dtype=float)
    k = len(gammas)

    probs = filtered_probs.reshape((2,) * k)

    for axis, gamma_i in enumerate(gammas):
        transition_i = np.array(
            [
                [1.0 - gamma_i / 2.0, gamma_i / 2.0],
                [gamma_i / 2.0, 1.0 - gamma_i / 2.0],
            ],
            dtype=float,
        )

        probs = np.tensordot(probs, transition_i, axes=([axis], [0]))
        probs = np.moveaxis(probs, -1, axis)

    next_probs = probs.reshape(-1)
    next_probs = np.clip(next_probs, 1e-300, 1.0)
    next_probs /= next_probs.sum()

    return next_probs


def _negative_loglikelihood(
    params: np.ndarray,
    y: np.ndarray,
    k: int,
) -> float:
    m0, sigma, b, gamma_k = params

    if not (
        0.0 < m0 < 2.0
        and sigma > 0.0
        and b > 1.0
        and 0.0 < gamma_k < 1.0
    ):
        return 1e50

    loglik = msm_loglikelihood(
        y=y,
        k=k,
        m0=float(m0),
        sigma=float(sigma),
        b=float(b),
        gamma_k=float(gamma_k),
    )

    if not np.isfinite(loglik):
        return 1e50

    return -float(loglik)


# def _initial_points_for_msm(
#     y: pd.Series,
#     k: int,
#     n_starts: int,
#     rng: np.random.Generator,
# ) -> list[np.ndarray]:
#     """Generate starting values for numerical optimization."""
#     sample_sigma = float(y.std(ddof=1))
#     sample_sigma = max(sample_sigma, 1e-2)

#     deterministic = [
#         np.array([1.50, sample_sigma, 2.0, 0.10]),
#         np.array([1.50, sample_sigma, 5.0, 0.20]),
#         np.array([1.40, sample_sigma, 10.0, 0.10]),
#         np.array([1.60, sample_sigma, 10.0, 0.20]),
#         np.array([1.30, sample_sigma, 20.0, 0.10]),
#     ]

#     random_points = []

#     for _ in range(max(0, n_starts - len(deterministic))):
#         random_points.append(
#             np.array(
#                 [
#                     rng.uniform(1.1, 1.8),          # m0
#                     sample_sigma * rng.uniform(0.6, 1.6),  # sigma
#                     rng.uniform(1.2, 30.0),         # b
#                     rng.uniform(0.02, 0.95),        # gamma_k
#                 ],
#                 dtype=float,
#             )
#         )

#     return deterministic + random_points

def _initial_points_for_msm(
    y: pd.Series,
    k: int,
    n_starts: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate exactly n_starts starting values for numerical optimization."""
    sample_sigma = float(y.std(ddof=1))
    sample_sigma = max(sample_sigma, 1e-2)

    deterministic = [
        np.array([1.50, sample_sigma, 2.0, 0.10]),
        np.array([1.50, sample_sigma, 5.0, 0.20]),
        np.array([1.40, sample_sigma, 10.0, 0.10]),
        np.array([1.60, sample_sigma, 10.0, 0.20]),
        np.array([1.30, sample_sigma, 20.0, 0.10]),
    ]

    points = deterministic[: min(n_starts, len(deterministic))]

    while len(points) < n_starts:
        points.append(
            np.array(
                [
                    rng.uniform(1.1, 1.8),
                    sample_sigma * rng.uniform(0.6, 1.6),
                    rng.uniform(1.2, 30.0),
                    rng.uniform(0.02, 0.95),
                ],
                dtype=float,
            )
        )

    return points


def _params_in_bounds(
    params: np.ndarray,
    bounds: list[tuple[float, float]],
) -> bool:
    return all(low <= value <= high for value, (low, high) in zip(params, bounds))


def _clean_centered_returns(y: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("MSM estimation requires at least one observation.")

    return values