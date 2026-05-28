"""Formatting and replication-table helpers for VaR backtesting outputs."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.risk import christoffersen_lr_test, gmm_duration_test, spa_pvalue_for_basis
from src.var import build_loss_panel, portfolio_returns, var_exceedances

BENCH_ORDER = ["Historical", "RiskMetrics", "Var-Covar", "CCC-GARCH"]
COPULA_ORDER = ["Normal", "Student", "Plackett", "Clayton", "rotClayton", "SJC", "Frank", "Gumbel", "rotGumbel"]
STAT_ORDER = ["EFV", "uc", "ind", "cc"]


def parse_model_name(model: str) -> tuple[str, str]:
    """Map an internal model name to the group/submodel labels used in Tables 5-8."""
    if model in ["Historical", "RiskMetrics", "Variance-Covariance", "CCC-GARCH"]:
        group = "Bench"
        submodel = {
            "Historical": "Historical",
            "RiskMetrics": "RiskMetrics",
            "Variance-Covariance": "Var-Covar",
            "CCC-GARCH": "CCC-GARCH",
        }[model]
    elif model.startswith("Copula-GARCH"):
        group = "Copula-GARCH"
        submodel = model.replace("Copula-GARCH ", "")
    elif model.startswith("Copula-MSM"):
        group = "Copula-MSM"
        submodel = model.replace("Copula-MSM ", "")
    else:
        group = "Other"
        submodel = model

    submodel = submodel.replace("Gaussian", "Normal")
    submodel = submodel.replace("Rotated Clayton", "rotClayton")
    submodel = submodel.replace("Rotated Gumbel", "rotGumbel")
    return group, submodel


def make_lr_table(returns: pd.DataFrame, var_panel: pd.DataFrame, alpha: float, weights=(0.5, 0.5)) -> pd.DataFrame:
    """Return EFV and Christoffersen LR p-values for each VaR model."""
    rp = portfolio_returns(returns, weights).reindex(var_panel.index)
    rows = {}
    for model in var_panel.columns:
        hits = var_exceedances(rp, var_panel[model])
        test = christoffersen_lr_test(hits, alpha)
        rows[model] = {
            "EFV": hits.mean(),
            "uc": test["uc_pvalue"],
            "ind": test["ind_pvalue"],
            "cc": test["cc_pvalue"],
        }
    return pd.DataFrame(rows).T


def format_lr_table(lr_table: pd.DataFrame, msm_submodels: Iterable[str] | None = None) -> pd.DataFrame:
    """Format Tables 5-6 with a MultiIndex column layout matching the paper."""
    df = lr_table.reset_index().rename(columns={"index": "model"})
    long = df.melt(id_vars="model", value_vars=STAT_ORDER, var_name="stat", value_name="value")
    long[["group", "submodel"]] = long["model"].apply(lambda x: pd.Series(parse_model_name(str(x))))
    out = long.pivot(index="stat", columns=["group", "submodel"], values="value")

    msm = list(msm_submodels) if msm_submodels is not None else COPULA_ORDER
    column_order = (
        [("Bench", c) for c in BENCH_ORDER]
        + [("Copula-GARCH", c) for c in COPULA_ORDER]
        + [("Copula-MSM", c) for c in msm]
    )
    out = out.reindex(index=STAT_ORDER, columns=pd.MultiIndex.from_tuples(column_order))
    return out.map(lambda x: f"{x:.3f}" if pd.notna(x) else "")


def make_gmm_table(returns: pd.DataFrame, var_panel: pd.DataFrame, alpha: float, weights=(0.5, 0.5)) -> pd.DataFrame:
    """Return GMM duration-based p-values for one VaR panel."""
    rp = portfolio_returns(returns, weights).reindex(var_panel.index)
    rows = ["Juc(1)", "Jcc(2)", "Jcc(3)", "Jcc(4)", "Jind(2)", "Jind(3)", "Jind(4)"]
    out = {}
    for model in var_panel.columns:
        hits = var_exceedances(rp, var_panel[model])
        out[model] = {
            "Juc(1)": gmm_duration_test(hits, alpha, p=1, kind="uc"),
            "Jcc(2)": gmm_duration_test(hits, alpha, p=2, kind="cc"),
            "Jcc(3)": gmm_duration_test(hits, alpha, p=3, kind="cc"),
            "Jcc(4)": gmm_duration_test(hits, alpha, p=4, kind="cc"),
            "Jind(2)": gmm_duration_test(hits, alpha, p=2, kind="ind"),
            "Jind(3)": gmm_duration_test(hits, alpha, p=3, kind="ind"),
            "Jind(4)": gmm_duration_test(hits, alpha, p=4, kind="ind"),
        }
    return pd.DataFrame(out).reindex(rows)


def rename_model_for_table9(name: str) -> str:
    """Rename model labels to the basis-model labels used in Table 9."""
    mapping = {
        "Historical": "Hist",
        "Hist": "Hist",
        "RiskMetrics": "RiskMetrics",
        "Variance-Covariance": "Covariance",
        "Var-Covar": "Covariance",
        "VarCov": "Covariance",
        "CCC-GARCH": "CCC-GARCH",
    }
    if name in mapping:
        return mapping[name]
    s = str(name)
    replacements = {
        "Gaussian": "Normal", "gaussian": "Normal", "student": "Student",
        "plackett": "Plackett", "clayton": "Clayton", "rotated_clayton": "rotClayton",
        "Rotated Clayton": "rotClayton", "sjc": "SJC", "frank": "Frank",
        "gumbel": "Gumbel", "rotated_gumbel": "rotGumbel", "Rotated Gumbel": "rotGumbel",
    }
    for prefix, suffix in [("Copula-GARCH ", "-GARCH"), ("Copula-MSM ", "-MSM")]:
        if s.startswith(prefix):
            cop = s.replace(prefix, "")
            for old, new in replacements.items():
                cop = cop.replace(old, new)
            return f"{cop}{suffix}"
    return s


def order_table9_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Order Table 9 rows as in the paper."""
    order = [
        "Hist", "RiskMetrics", "Covariance", "CCC-GARCH",
        "Normal-GARCH", "Student-GARCH", "Plackett-GARCH", "Clayton-GARCH",
        "rotClayton-GARCH", "SJC-GARCH", "Frank-GARCH", "Gumbel-GARCH", "rotGumbel-GARCH",
        "Normal-MSM", "Student-MSM", "Plackett-MSM", "Clayton-MSM",
        "rotClayton-MSM", "SJC-MSM", "Frank-MSM", "Gumbel-MSM", "rotGumbel-MSM",
    ]
    return df.reindex([x for x in order if x in df.index])


def make_spa_table9(
    portfolio_returns: pd.Series,
    var_5_panel: pd.DataFrame,
    var_1_panel: pd.DataFrame,
    B: int = 5000,
    block_p: float = 0.1,
    nu: float = 25.0,
    seed: int = 123,
    nw_lag: int | None = None,
) -> pd.DataFrame:
    """Compute the Table 9 SPA p-values for VaR and smooth VaR losses."""
    loss_5 = build_loss_panel(portfolio_returns, var_5_panel, alpha=0.05, smooth=False)
    sloss_5 = build_loss_panel(portfolio_returns, var_5_panel, alpha=0.05, smooth=True, nu=nu)
    loss_1 = build_loss_panel(portfolio_returns, var_1_panel, alpha=0.01, smooth=False)
    sloss_1 = build_loss_panel(portfolio_returns, var_1_panel, alpha=0.01, smooth=True, nu=nu)

    common = set(loss_5.columns) & set(sloss_5.columns) & set(loss_1.columns) & set(sloss_1.columns)
    common_models = [m for m in loss_5.columns if m in common]

    rows = {}
    for i, model in enumerate(common_models):
        rows[rename_model_for_table9(model)] = {
            "VaRl(5%)": spa_pvalue_for_basis(loss_5[common_models], model, B=B, block_p=block_p, seed=seed + 10_000 * i + 1, nw_lag=nw_lag),
            "SVaRl(5%)": spa_pvalue_for_basis(sloss_5[common_models], model, B=B, block_p=block_p, seed=seed + 10_000 * i + 2, nw_lag=nw_lag),
            "VaRl(1%)": spa_pvalue_for_basis(loss_1[common_models], model, B=B, block_p=block_p, seed=seed + 10_000 * i + 3, nw_lag=nw_lag),
            "SVaRl(1%)": spa_pvalue_for_basis(sloss_1[common_models], model, B=B, block_p=block_p, seed=seed + 10_000 * i + 4, nw_lag=nw_lag),
        }
    return order_table9_rows(pd.DataFrame(rows).T)
