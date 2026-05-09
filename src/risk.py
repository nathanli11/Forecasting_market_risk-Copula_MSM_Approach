"""Core risk, dependence, and backtesting helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller

def summary_statistics(
    returns: pd.DataFrame | pd.Series,
    arch_lags: tuple[int, ...] = (1, 5, 10),
    hill_tail_fraction: float = 0.05,
) -> pd.DataFrame:
    """Return descriptive statistics used in Table 1 of the paper.

    The input should be percentage log returns, i.e. 100 * log(P_t / P_{t-1}).
    """
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    rows = []

    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()

        row: dict[str, float] = {
            "count": float(series.shape[0]),
            "Mean": float(series.mean()),
            "Std": float(series.std(ddof=1)),
            "Skewness": float(series.skew()),
            # pandas returns excess kurtosis, so add 3 for Pearson kurtosis
            "Kurtosis": float(series.kurtosis() + 3.0),
            "Hurst": hurst_exponent(series),
            "Tail index": hill_tail_index(
                series,
                fraction=hill_tail_fraction,
            ),
        }

        for lag in arch_lags:
            arch_stat, arch_pvalue = arch_lm_test(series, lags=lag)
            row[f"Arch({lag})"] = arch_stat
            row[f"Arch({lag}) p-value"] = arch_pvalue

        jb_result = stats.jarque_bera(series)
        row["JB"] = float(jb_result.statistic)
        row["JB p-value"] = float(jb_result.pvalue)

        adf_result = adfuller(series, autolag="AIC")
        # The paper reports positive ADF values, so we store abs(test statistic)
        row["ADF"] = float(abs(adf_result[0]))
        row["ADF p-value"] = float(adf_result[1])

        rows.append(pd.Series(row, name=column))

    return pd.DataFrame(rows)


def arch_lm_test(series: pd.Series, lags: int) -> tuple[float, float]:
    """Return Engle ARCH LM statistic and p-value."""
    clean = pd.to_numeric(series, errors="coerce").dropna()

    if clean.empty:
        raise ValueError("ARCH test requires at least one observation.")

    lm_stat, lm_pvalue, _, _ = het_arch(clean, nlags=lags)

    return float(lm_stat), float(lm_pvalue)


def hill_tail_index(
    series: pd.Series,
    fraction: float = 0.05,
) -> float:
    """Estimate the tail index using Hill's estimator on absolute returns.

    The result is sensitive to the fraction of observations used in the tail.
    The paper reports values around 3 to 4.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    values = np.abs(clean.to_numpy(dtype=float))
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if values.size < 20:
        return float("nan")

    values = np.sort(values)

    k = int(np.floor(fraction * values.size))
    k = max(k, 5)
    k = min(k, values.size - 1)

    tail_values = values[-k:]
    threshold = values[-k - 1]

    hill_gamma = np.mean(np.log(tail_values) - np.log(threshold))

    if hill_gamma <= 0:
        return float("nan")

    return float(1.0 / hill_gamma)


def hurst_exponent(
    series: pd.Series,
    min_lag: int = 2,
    max_lag: int = 100,
) -> float:
    """Estimate the Hurst exponent using the log-log variance method."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)

    if values.size < 20:
        return float("nan")

    max_lag = min(max_lag, values.size // 4)

    if max_lag <= min_lag:
        return float("nan")

    lags = np.arange(min_lag, max_lag + 1)
    tau = []

    for lag in lags:
        diff = values[lag:] - values[:-lag]
        tau.append(np.sqrt(np.std(diff, ddof=1)))

    tau = np.asarray(tau, dtype=float)
    valid = np.isfinite(tau) & (tau > 0)

    if valid.sum() < 2:
        return float("nan")

    slope, _ = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)

    return float(slope)


def format_table_1(stats_table: pd.DataFrame) -> pd.DataFrame:
    """Format Table 1 values with statistics and p-values."""
    main_columns = [
        "Mean",
        "Std",
        "Skewness",
        "Kurtosis",
        "Hurst",
        "Tail index",
    ]

    test_columns = [
        "Arch(1)",
        "Arch(5)",
        "Arch(10)",
        "JB",
        "ADF",
    ]

    formatted = pd.DataFrame(index=stats_table.index)

    for column in main_columns:
        formatted[column] = stats_table[column].map(lambda value: f"{value:.3f}")

    for column in test_columns:
        pvalue_column = f"{column} p-value"
        formatted[column] = [
            f"{stat:.3f}\n({pvalue:.3f})"
            for stat, pvalue in zip(
                stats_table[column],
                stats_table[pvalue_column],
            )
        ]

    return formatted


def realized_volatility(
    returns: pd.Series,
    window: int = 20,
    annualization: int = 252,
) -> pd.Series:
    """Rolling annualized realized volatility."""
    if window <= 1:
        raise ValueError("window must be greater than 1.")
    return returns.rolling(window).std() * np.sqrt(annualization)


def standardize_returns(returns: pd.Series) -> pd.Series:
    """Center and scale a return series."""
    cleaned = returns.dropna()
    std = cleaned.std()
    if std == 0:
        raise ValueError("Cannot standardize a constant return series.")
    return (cleaned - cleaned.mean()) / std


def pseudo_observations(data: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Convert observations to empirical CDF ranks in the open unit interval."""
    frame = data.to_frame() if isinstance(data, pd.Series) else data
    ranks = frame.rank(method="average", pct=False)
    return ranks / (len(frame) + 1.0)


def gaussian_copula_correlation(uniforms: pd.DataFrame) -> pd.DataFrame:
    """Estimate a Gaussian copula correlation matrix from pseudo-observations."""
    if not ((uniforms > 0) & (uniforms < 1)).all().all():
        raise ValueError("Gaussian copula inputs must lie strictly between 0 and 1.")
    normal_scores = pd.DataFrame(
        stats.norm.ppf(uniforms),
        index=uniforms.index,
        columns=uniforms.columns,
    )
    return normal_scores.corr()


def simulate_gaussian_copula(
    correlation: pd.DataFrame | np.ndarray,
    n_samples: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate uniforms from a Gaussian copula."""
    rng = np.random.default_rng(seed)
    corr = np.asarray(correlation, dtype=float)
    draws = rng.multivariate_normal(np.zeros(corr.shape[0]), corr, size=n_samples)
    return stats.norm.cdf(draws)


def historical_var(returns: pd.Series | np.ndarray, alpha: float = 0.01) -> float:
    """Historical VaR as a positive loss number at tail probability alpha."""
    _validate_alpha(alpha)
    values = _clean_returns(returns)
    quantile = np.quantile(values, alpha)
    return float(max(-quantile, 0.0))


def gaussian_var(
    returns: pd.Series | np.ndarray,
    alpha: float = 0.01,
    mean: float | None = None,
    volatility: float | None = None,
) -> float:
    """Parametric Gaussian VaR as a positive loss number."""
    _validate_alpha(alpha)
    values = _clean_returns(returns)
    mu = float(values.mean()) if mean is None else float(mean)
    sigma = float(values.std(ddof=1)) if volatility is None else float(volatility)
    if sigma < 0:
        raise ValueError("volatility must be non-negative.")
    quantile = mu + sigma * stats.norm.ppf(alpha)
    return float(max(-quantile, 0.0))


def portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
) -> pd.Series:
    """Compute weighted portfolio returns from aligned asset returns."""
    weight_array = np.asarray(weights, dtype=float)
    if returns.shape[1] != len(weight_array):
        raise ValueError("Number of weights must match number of return columns.")
    if not np.isclose(weight_array.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")
    return returns @ weight_array


def var_exceedances(returns: pd.Series, var_forecasts: pd.Series | float) -> pd.Series:
    """Indicator for losses exceeding positive VaR forecasts."""
    forecasts = (
        pd.Series(var_forecasts, index=returns.index)
        if np.isscalar(var_forecasts)
        else var_forecasts.reindex(returns.index)
    )
    losses = -returns
    return losses.gt(forecasts).astype(int)


def violation_rate(returns: pd.Series, var_forecasts: pd.Series | float) -> float:
    """Observed VaR violation frequency."""
    return float(var_exceedances(returns, var_forecasts).mean())


def kupiec_pof_test(
    exceedances: pd.Series | np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Kupiec unconditional coverage likelihood-ratio test."""
    hits = np.asarray(exceedances, dtype=int)
    if hits.size == 0:
        raise ValueError("At least one exceedance indicator is required.")
    x = int(hits.sum())
    n = int(hits.size)
    phat = x / n

    if x == 0 or x == n:
        log_l_unrestricted = 0.0
    else:
        log_l_unrestricted = x * math.log(phat) + (n - x) * math.log(1 - phat)

    log_l_restricted = x * math.log(alpha) + (n - x) * math.log(1 - alpha)
    statistic = -2 * (log_l_restricted - log_l_unrestricted)
    pvalue = 1 - stats.chi2.cdf(statistic, df=1)
    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "violations": float(x),
    }


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def _clean_returns(returns: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("At least one return observation is required.")
    return values
