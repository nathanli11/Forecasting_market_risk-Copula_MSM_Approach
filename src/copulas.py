"""Bivariate copula estimation for copula-MSM and copula-GARCH models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, multivariate_t, norm, t


EPS = 1e-10


@dataclass(frozen=True)
class CopulaFitResult:
    """Container for an estimated bivariate copula."""

    copula: str
    margin_model: str
    params: dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    nobs: int
    success: bool
    message: str


def _validate_uniforms(uniforms: pd.DataFrame) -> pd.DataFrame:
    """Validate and clip a two-column PIT DataFrame."""
    if uniforms.shape[1] != 2:
        raise ValueError("Copula estimation expects exactly two uniform series.")

    frame = uniforms.copy()
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    frame = frame.clip(lower=EPS, upper=1.0 - EPS)

    if frame.empty:
        raise ValueError("At least one observation is required.")

    return frame


def gaussian_copula_logpdf(uniforms: pd.DataFrame, rho: float) -> np.ndarray:
    """Gaussian copula log-density."""
    frame = _validate_uniforms(uniforms)

    if not -0.999 < rho < 0.999:
        return np.full(len(frame), -np.inf)

    z = norm.ppf(frame.to_numpy(dtype=float))
    corr = np.array([[1.0, rho], [rho, 1.0]])

    joint = multivariate_normal.logpdf(
        z,
        mean=np.zeros(2),
        cov=corr,
    )

    marginals = norm.logpdf(z[:, 0]) + norm.logpdf(z[:, 1])
    return joint - marginals


def student_copula_logpdf(
    uniforms: pd.DataFrame,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Student-t copula log-density."""
    frame = _validate_uniforms(uniforms)

    if not -0.999 < rho < 0.999 or nu <= 2.01:
        return np.full(len(frame), -np.inf)

    x = t.ppf(frame.to_numpy(dtype=float), df=nu)
    shape = np.array([[1.0, rho], [rho, 1.0]])

    joint = multivariate_t.logpdf(
        x,
        loc=np.zeros(2),
        shape=shape,
        df=nu,
    )

    marginals = t.logpdf(x[:, 0], df=nu) + t.logpdf(x[:, 1], df=nu)
    return joint - marginals


def clayton_copula_logpdf(uniforms: pd.DataFrame, theta: float) -> np.ndarray:
    """Clayton copula log-density."""
    frame = _validate_uniforms(uniforms)

    if theta <= 0:
        return np.full(len(frame), -np.inf)

    u = frame.iloc[:, 0].to_numpy(dtype=float)
    v = frame.iloc[:, 1].to_numpy(dtype=float)

    s = u ** (-theta) + v ** (-theta) - 1.0

    log_density = (
        np.log(theta + 1.0)
        + (-theta - 1.0) * (np.log(u) + np.log(v))
        + (-2.0 - 1.0 / theta) * np.log(s)
    )

    return log_density


def rotated_clayton_copula_logpdf(
    uniforms: pd.DataFrame,
    theta: float,
) -> np.ndarray:
    """Survival, or 180-degree rotated, Clayton copula log-density."""
    frame = _validate_uniforms(uniforms)
    rotated = 1.0 - frame
    return clayton_copula_logpdf(rotated, theta=theta)


def gumbel_copula_logpdf(uniforms: pd.DataFrame, theta: float) -> np.ndarray:
    """Gumbel copula log-density."""
    frame = _validate_uniforms(uniforms)

    if theta < 1.0:
        return np.full(len(frame), -np.inf)

    u = frame.iloc[:, 0].to_numpy(dtype=float)
    v = frame.iloc[:, 1].to_numpy(dtype=float)

    x = -np.log(u)
    y = -np.log(v)

    a = x**theta + y**theta
    s = a ** (1.0 / theta)

    log_density = (
        -s
        - np.log(u)
        - np.log(v)
        + (theta - 1.0) * (np.log(x) + np.log(y))
        + (1.0 / theta - 2.0) * np.log(a)
        + np.log(s + theta - 1.0)
    )

    return log_density


def rotated_gumbel_copula_logpdf(
    uniforms: pd.DataFrame,
    theta: float,
) -> np.ndarray:
    """Survival, or 180-degree rotated, Gumbel copula log-density."""
    frame = _validate_uniforms(uniforms)
    rotated = 1.0 - frame
    return gumbel_copula_logpdf(rotated, theta=theta)


def frank_copula_logpdf(uniforms: pd.DataFrame, theta: float) -> np.ndarray:
    """Frank copula log-density."""
    frame = _validate_uniforms(uniforms)

    if abs(theta) < 1e-6:
        return np.zeros(len(frame))

    u = frame.iloc[:, 0].to_numpy(dtype=float)
    v = frame.iloc[:, 1].to_numpy(dtype=float)

    numerator = (
        theta
        * (1.0 - np.exp(-theta))
        * np.exp(-theta * (u + v))
    )

    denominator = (
        (1.0 - np.exp(-theta))
        - (1.0 - np.exp(-theta * u)) * (1.0 - np.exp(-theta * v))
    ) ** 2

    density = numerator / denominator

    if np.any(~np.isfinite(density)) or np.any(density <= 0):
        return np.full(len(frame), -np.inf)

    return np.log(density)


def plackett_copula_logpdf(uniforms: pd.DataFrame, theta: float) -> np.ndarray:
    """Plackett copula log-density.

    theta = 1 corresponds to independence.
    """
    frame = _validate_uniforms(uniforms)

    if theta <= 0:
        return np.full(len(frame), -np.inf)

    if abs(theta - 1.0) < 1e-6:
        return np.zeros(len(frame))

    u = frame.iloc[:, 0].to_numpy(dtype=float)
    v = frame.iloc[:, 1].to_numpy(dtype=float)

    a = theta - 1.0
    denominator_base = (
        (1.0 + a * (u + v)) ** 2
        - 4.0 * theta * a * u * v
    )

    numerator = theta * (1.0 + a * (u + v - 2.0 * u * v))
    if np.any(denominator_base <= 0) or np.any(~np.isfinite(denominator_base)):
        return np.full(len(frame), -np.inf)

    density = numerator / (denominator_base ** 1.5)

    if np.any(~np.isfinite(density)) or np.any(density <= 0):
        return np.full(len(frame), -np.inf)

    return np.log(density)


def _copula_logpdf_from_vector(
    copula: str,
    uniforms: pd.DataFrame,
    x: np.ndarray,
) -> np.ndarray:
    """Dispatch log-density evaluation from an optimizer parameter vector."""
    copula = copula.lower()

    if copula == "gaussian":
        return gaussian_copula_logpdf(uniforms, rho=float(x[0]))

    if copula == "student":
        return student_copula_logpdf(
            uniforms,
            rho=float(x[0]),
            nu=float(x[1]),
        )

    if copula == "plackett":
        return plackett_copula_logpdf(uniforms, theta=float(x[0]))

    if copula == "clayton":
        return clayton_copula_logpdf(uniforms, theta=float(x[0]))

    if copula == "rotated_clayton":
        return rotated_clayton_copula_logpdf(uniforms, theta=float(x[0]))

    if copula == "frank":
        return frank_copula_logpdf(uniforms, theta=float(x[0]))

    if copula == "gumbel":
        return gumbel_copula_logpdf(uniforms, theta=float(x[0]))

    if copula == "rotated_gumbel":
        return rotated_gumbel_copula_logpdf(uniforms, theta=float(x[0]))

    raise ValueError(f"Unknown copula: {copula}")


def _negative_loglikelihood(
    x: np.ndarray,
    copula: str,
    uniforms: pd.DataFrame,
) -> float:
    logpdf = _copula_logpdf_from_vector(copula, uniforms, x)

    if np.any(~np.isfinite(logpdf)):
        return 1e50

    return -float(np.sum(logpdf))


def _initial_points_and_bounds(
    copula: str,
    uniforms: pd.DataFrame,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Return starting values and bounds for a copula."""
    frame = _validate_uniforms(uniforms)
    z = norm.ppf(frame.to_numpy(dtype=float))
    rho0 = float(np.corrcoef(z[:, 0], z[:, 1])[0, 1])
    rho0 = float(np.clip(rho0, -0.95, 0.95))

    copula = copula.lower()

    if copula == "gaussian":
        return [np.array([rho0])], [(-0.999, 0.999)]

    if copula == "student":
        return [
            np.array([rho0, 5.0]),
            np.array([rho0, 10.0]),
            np.array([rho0, 20.0]),
        ], [(-0.999, 0.999), (2.01, 100.0)]
    
    if copula == "plackett":
        return [
            np.array([20.0]),
            np.array([50.0]),
            np.array([80.0]),
            np.array([150.0]),
        ], [(1e-4, 500.0)]

    if copula in {"clayton", "rotated_clayton"}:
        return [
            np.array([1.0]),
            np.array([2.0]),
            np.array([4.0]),
            np.array([8.0]),
        ], [(1e-4, 50.0)]

    if copula == "frank":
        return [
            np.array([5.0]),
            np.array([10.0]),
            np.array([20.0]),
            np.array([30.0]),
            np.array([50.0]),
        ], [(-100.0, 100.0)]

    if copula in {"gumbel", "rotated_gumbel"}:
        return [
            np.array([1.2]),
            np.array([2.0]),
            np.array([4.0]),
            np.array([6.0]),
            np.array([10.0]),
        ], [(1.0001, 50.0)]

    raise ValueError(f"Unknown copula: {copula}")


def _params_dict(copula: str, x: np.ndarray) -> dict[str, float]:
    copula = copula.lower()

    if copula == "gaussian":
        return {"rho": float(x[0])}

    if copula == "student":
        return {"rho": float(x[0]), "nu": float(x[1])}

    if copula == "plackett":
        return {"theta": float(x[0])}

    if copula in {"clayton", "rotated_clayton"}:
        return {"theta": float(x[0])}

    if copula == "frank":
        return {"theta": float(x[0])}

    if copula in {"gumbel", "rotated_gumbel"}:
        return {"theta": float(x[0])}

    raise ValueError(f"Unknown copula: {copula}")


def fit_copula(
    uniforms: pd.DataFrame,
    copula: str,
    margin_model: str,
) -> CopulaFitResult:
    """Estimate one bivariate copula by maximum likelihood."""
    frame = _validate_uniforms(uniforms)

    initial_points, bounds = _initial_points_and_bounds(copula, frame)

    best_opt = None
    best_fun = np.inf
    fallback_opt = None
    fallback_fun = np.inf

    for x0 in initial_points:
        opt = minimize(
            _negative_loglikelihood,
            x0=x0,
            args=(copula, frame),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-10},
        )
        if np.isfinite(opt.fun) and opt.fun < fallback_fun:
            fallback_fun = float(opt.fun)
            fallback_opt = opt
        if opt.success and np.isfinite(opt.fun) and opt.fun < best_fun:
            best_fun = float(opt.fun)
            best_opt = opt

    if best_opt is None:
        best_opt = fallback_opt
    if best_opt is None:
        raise RuntimeError(f"Copula estimation failed for {copula}.")

    log_likelihood = -float(best_opt.fun)
    n_params = len(best_opt.x)
    nobs = int(frame.shape[0])

    return CopulaFitResult(
        copula=copula,
        margin_model=margin_model,
        params=_params_dict(copula, best_opt.x),
        log_likelihood=log_likelihood,
        aic=-2.0 * log_likelihood + 2.0 * n_params,
        bic=-2.0 * log_likelihood + np.log(nobs) * n_params,
        nobs=nobs,
        success=bool(best_opt.success),
        message=str(best_opt.message),
    )


def fit_all_copulas(
    uniforms: pd.DataFrame,
    margin_model: str,
    copulas: tuple[str, ...] = (
        "gaussian",
        "student",
        "plackett",
        "clayton",
        "rotated_clayton",
        "frank",
        "gumbel",
        "rotated_gumbel",
    ),
) -> list[CopulaFitResult]:
    """Estimate several copulas on the same PIT data."""
    results = []

    for copula in copulas:
        result = fit_copula(
            uniforms=uniforms,
            copula=copula,
            margin_model=margin_model,
        )
        results.append(result)

    return results


def fit_copula_grid(
    pit_by_model: dict[str, pd.DataFrame],
    copulas: tuple[str, ...] = (
        "gaussian",
        "student",
        "plackett",
        "clayton",
        "rotated_clayton",
        "frank",
        "gumbel",
        "rotated_gumbel",
    ),
) -> list[CopulaFitResult]:
    """Estimate copulas for several margin models, e.g. MSM and GARCH."""
    results = []

    for margin_model, uniforms in pit_by_model.items():
        results.extend(
            fit_all_copulas(
                uniforms=uniforms,
                margin_model=margin_model,
                copulas=copulas,
            )
        )

    return results


def copula_fit_result_to_dict(
    result: CopulaFitResult,
) -> dict[str, float | int | str | bool]:
    """Convert a CopulaFitResult to a flat dictionary."""
    row: dict[str, float | int | str | bool] = {
        "margin_model": result.margin_model,
        "copula": result.copula,
        "log_likelihood": result.log_likelihood,
        "aic": result.aic,
        "bic": result.bic,
        "nobs": result.nobs,
        "success": result.success,
        "message": result.message,
    }

    row.update(result.params)
    return row


def copula_results_table(
    results: list[CopulaFitResult],
) -> pd.DataFrame:
    """Return a table of copula estimates."""
    return pd.DataFrame(
        [copula_fit_result_to_dict(result) for result in results]
    )


def format_copula_table_4(table: pd.DataFrame) -> pd.DataFrame:
    """Format copula estimates close to the paper's Table 4."""
    rows = []

    for _, row in table.iterrows():
        params = []

        for name in ["rho", "nu", "theta"]:
            if name in row and pd.notna(row[name]):
                params.append(f"{name}={row[name]:.3f}")

        rows.append(
            {
                "margin_model": row["margin_model"],
                "copula": row["copula"],
                "parameters": ", ".join(params),
                "Log(L)": f"{row['log_likelihood']:.3f}",
                "AIC": f"{row['aic']:.3f}",
                "BIC": f"{row['bic']:.3f}",
            }
        )

    return pd.DataFrame(rows)


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
        norm.ppf(uniforms),
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
    return norm.cdf(draws)