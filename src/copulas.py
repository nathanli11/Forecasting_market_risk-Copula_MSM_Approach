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
    density = np.maximum(density, EPS)

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
    density = numerator / (denominator_base ** 1.5)
    density = np.maximum(density, EPS)

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


def _initial_and_bounds(
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

    initial_points, bounds = _initial_and_bounds(copula, frame)

    best_opt = None
    best_fun = np.inf

    for x0 in initial_points:
        opt = minimize(
            _negative_loglikelihood,
            x0=x0,
            args=(copula, frame),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-10},
        )

        if np.isfinite(opt.fun) and opt.fun < best_fun:
            best_fun = float(opt.fun)
            best_opt = opt

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
 
def portfolio_var_mc(
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
    n_samples: int = 10_000,
    seed: int | None = None,
    )-> float:
    """
    Portfolio VaR at level alpha via Monte Carlo — Eq. (12)-(13).
 
    Steps:
        1. Sample (u1, u2) from the fitted copula
        2. Map to returns via MSM inverse CDF
        3. r_p = pi*r1 + (1-pi)*r2
        4. VaR = quantile(r_p, alpha)
 
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
        VaR_t(alpha) — signed return (negative = loss).
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
    
    r1 = _msm_quantile(u1, state_probs_1, sigma_1, h_1)
    r2 = _msm_quantile(u2, state_probs_2, sigma_2, h_2)
    r_p = pi * r1 + (1.0 - pi)*r2
    return float(np.quantile(r_p, alpha))


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
 