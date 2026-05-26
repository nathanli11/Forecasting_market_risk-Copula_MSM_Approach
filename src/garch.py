"""GARCH marginal models for copula-GARCH replication."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


@dataclass(frozen=True)
class GARCHFitResult:
    """Container for an estimated GARCH(1,1) marginal model."""
    asset: str
    model_result: object
    mean: float
    omega: float
    alpha: float
    beta: float
    mean_se: float
    omega_se: float
    alpha_se: float
    beta_se: float
    log_likelihood: float
    aic: float
    bic: float
    nobs: int
    dist: str


def fit_garch_11(
    returns: pd.Series,
    mean: str = "Constant",
    dist: str = "normal",
    rescale: bool = False,
) -> GARCHFitResult:
    """Estimate a Gaussian GARCH(1,1) model for one return series.

    The input should be percentage returns, as in the paper.
    """
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError(
            "Install the `arch` package to fit GARCH models: pip install arch"
        ) from exc

    series = pd.to_numeric(returns, errors="coerce").dropna()
    asset = str(series.name or "asset")

    model = arch_model(
        series,
        mean=mean,
        vol="GARCH",
        p=1,
        q=1,
        dist=dist,
        rescale=rescale,
    )

    result = model.fit(disp="off")

    params = result.params
    std_errors = result.std_err

    mean_param_name = "mu"

    return GARCHFitResult(
        asset=asset,
        model_result=result,
        mean=float(params.get(mean_param_name, 0.0)),
        omega=float(params["omega"]),
        alpha=float(params["alpha[1]"]),
        beta=float(params["beta[1]"]),
        mean_se=float(std_errors.get(mean_param_name, np.nan)),
        omega_se=float(std_errors["omega"]),
        alpha_se=float(std_errors["alpha[1]"]),
        beta_se=float(std_errors["beta[1]"]),
        log_likelihood=float(result.loglikelihood),
        aic=float(result.aic),
        bic=float(result.bic),
        nobs=int(result.nobs),
        dist=dist,
    )


def fit_garch_marginals(
    returns: pd.DataFrame,
    mean: str = "Constant",
    dist: str = "normal",
    rescale: bool = False,
) -> dict[str, GARCHFitResult]:
    """Estimate GARCH(1,1) for each column of a returns DataFrame."""
    results = {}

    for asset in returns.columns:
        results[asset] = fit_garch_11(
            returns=returns[asset],
            mean=mean,
            dist=dist,
            rescale=rescale,
        )

    return results


def garch_standardized_residuals(
    fit_result: GARCHFitResult,
) -> pd.Series:
    """Return standardized residuals from a fitted GARCH model."""
    result = fit_result.model_result
    residuals = result.std_resid.dropna()
    residuals.name = fit_result.asset
    return residuals


def garch_conditional_volatility(
    fit_result: GARCHFitResult,
) -> pd.Series:
    """Return conditional volatility from a fitted GARCH model."""
    result = fit_result.model_result
    volatility = result.conditional_volatility.dropna()
    volatility.name = fit_result.asset
    return volatility


def garch_probability_integral_transform(
    fit_result: GARCHFitResult,
    clip_cdf: float = 1e-10,
) -> pd.Series:
    """Compute GARCH PIT values using standardized residuals.

    Under normal innovations:
        u_t = Phi(epsilon_t)
    """
    standardized_residuals = garch_standardized_residuals(fit_result)
    pit = pd.Series(
        norm.cdf(standardized_residuals.to_numpy(dtype=float)),
        index=standardized_residuals.index,
        name=fit_result.asset,
    )

    pit = pit.clip(lower=clip_cdf, upper=1.0 - clip_cdf)
    return pit


def build_garch_pit_frame(
    fit_results: dict[str, GARCHFitResult],
    clip_cdf: float = 1e-10,
) -> pd.DataFrame:
    """Build a DataFrame of GARCH PIT values for all assets."""
    pits = {
        asset: garch_probability_integral_transform(result, clip_cdf=clip_cdf)
        for asset, result in fit_results.items()
    }

    pit_frame = pd.concat(pits, axis=1).dropna(how="any")
    pit_frame.index.name = "date"

    return pit_frame


def build_garch_volatility_frame(
    fit_results: dict[str, GARCHFitResult],
) -> pd.DataFrame:
    """Build a DataFrame of GARCH conditional volatilities."""
    volatilities = {
        asset: garch_conditional_volatility(result)
        for asset, result in fit_results.items()
    }

    volatility_frame = pd.concat(volatilities, axis=1).dropna(how="any")
    volatility_frame.index.name = "date"

    return volatility_frame


def garch_diagnostics(
    fit_result: GARCHFitResult,
    arch_lags: tuple[int, ...] = (1, 5, 10),
    ljungbox_lags: tuple[int, ...] = (2, 4, 8),
) -> dict[str, float]:
    """Compute ARCH-LM and Ljung-Box diagnostics for standardized residuals."""
    residuals = garch_standardized_residuals(fit_result).dropna()

    diagnostics: dict[str, float] = {}

    for lag in arch_lags:
        stat, pvalue, _, _ = het_arch(residuals, nlags=lag)
        diagnostics[f"Arch({lag})"] = float(stat)
        diagnostics[f"Arch({lag}) p-value"] = float(pvalue)

    lb = acorr_ljungbox(
        residuals,
        lags=list(ljungbox_lags),
        return_df=True,
    )

    for lag in ljungbox_lags:
        diagnostics[f"Q({lag})"] = float(lb.loc[lag, "lb_stat"])
        diagnostics[f"Q({lag}) p-value"] = float(lb.loc[lag, "lb_pvalue"])

    return diagnostics


def garch_fit_result_to_dict(
    fit_result: GARCHFitResult,
) -> dict[str, float | int | str]:
    """Convert a GARCHFitResult to a flat dictionary."""
    row = {
        "asset": fit_result.asset,
        "mean": fit_result.mean,
        "mean_se": fit_result.mean_se,
        "omega": fit_result.omega,
        "omega_se": fit_result.omega_se,
        "alpha": fit_result.alpha,
        "alpha_se": fit_result.alpha_se,
        "beta": fit_result.beta,
        "beta_se": fit_result.beta_se,
        "log_likelihood": fit_result.log_likelihood,
        "aic": fit_result.aic,
        "bic": fit_result.bic,
        "nobs": fit_result.nobs,
        "dist": fit_result.dist,
    }

    row.update(garch_diagnostics(fit_result))
    return row


def garch_results_table(
    fit_results: dict[str, GARCHFitResult],
) -> pd.DataFrame:
    """Return a table with GARCH parameters and diagnostics."""
    rows = [
        garch_fit_result_to_dict(result)
        for result in fit_results.values()
    ]
    return pd.DataFrame(rows)


def format_garch_table_3(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Format GARCH results in a style close to Table 3 of the paper."""
    formatted = pd.DataFrame(index=table["asset"])

    for param in ["omega", "alpha", "beta"]:
        se_col = f"{param}_se"
        formatted[param] = [
            f"{value:.3f} [{se:.3f}]"
            for value, se in zip(table[param], table[se_col])
        ]

    for test in ["Arch(1)", "Arch(5)", "Arch(10)", "Q(2)", "Q(4)", "Q(8)"]:
        pvalue_col = f"{test} p-value"
        formatted[test] = [
            f"{stat:.3f} ({pvalue:.3f})"
            for stat, pvalue in zip(table[test], table[pvalue_col])
        ]

    return formatted