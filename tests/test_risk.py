import numpy as np
import pandas as pd
import pytest

from src.copulas import (
    gaussian_copula_correlation,
    pseudo_observations,
    simulate_gaussian_copula,
)
from src.risk import (
    christoffersen_lr_test,
    gaussian_var,
    historical_var,
    kupiec_pof_test,
    portfolio_returns,
    var_exceedances,
    violation_rate,
)


def test_var_helpers_use_positive_loss_convention() -> None:
    returns = pd.Series([-0.05, -0.02, 0.01, 0.03])

    assert np.isclose(historical_var(returns, alpha=0.25), 0.0275)
    assert gaussian_var(returns, alpha=0.05, mean=0.0, volatility=0.02) > 0


def test_portfolio_returns_validates_weights() -> None:
    returns = pd.DataFrame({"a": [0.01], "b": [0.02]})

    with pytest.raises(ValueError, match="sum to 1"):
        portfolio_returns(returns, [0.4, 0.4])


def test_var_backtesting_helpers() -> None:
    returns = pd.Series([-0.02, 0.01, -0.05])
    forecasts = pd.Series([0.01, 0.01, 0.04])

    hits = var_exceedances(returns, forecasts)

    assert hits.tolist() == [1, 0, 1]
    assert np.isclose(violation_rate(returns, 0.03), 1 / 3)

    test = kupiec_pof_test(pd.Series([0, 1, 0, 0, 1]), alpha=0.05)

    assert set(test) == {
        "statistic",
        "pvalue",
        "violations",
        "nobs",
        "violation_rate",
    }
    assert test["violations"] == 2
    assert test["nobs"] == 5
    assert np.isclose(test["violation_rate"], 2 / 5)


def test_christoffersen_lr_test() -> None:
    hits = pd.Series([0, 0, 1, 0, 0, 1, 0, 0, 0, 1])

    test = christoffersen_lr_test(hits, alpha=0.05)

    assert set(test) == {
        "efv",
        "uc_stat",
        "uc_pvalue",
        "ind_stat",
        "ind_pvalue",
        "cc_stat",
        "cc_pvalue",
        "violations",
        "nobs",
        "n00",
        "n01",
        "n10",
        "n11",
    }

    assert test["violations"] == 3
    assert test["nobs"] == 10
    assert 0.0 <= test["uc_pvalue"] <= 1.0
    assert 0.0 <= test["ind_pvalue"] <= 1.0
    assert 0.0 <= test["cc_pvalue"] <= 1.0


def test_kupiec_handles_zero_violations() -> None:
    hits = pd.Series(np.zeros(500, dtype=int))

    test = kupiec_pof_test(hits, alpha=0.05)

    assert test["violations"] == 0
    assert test["statistic"] > 0
    assert 0.0 <= test["pvalue"] <= 1.0


def test_gaussian_copula_helpers() -> None:
    data = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [3.0, 2.0, 1.0],
        }
    )

    uniforms = pseudo_observations(data)

    assert ((uniforms > 0) & (uniforms < 1)).all().all()

    corr = gaussian_copula_correlation(uniforms)

    assert corr.shape == (2, 2)
    assert np.isclose(corr.loc["a", "a"], 1.0)

    draws = simulate_gaussian_copula(np.eye(2), n_samples=10, seed=123)

    assert draws.shape == (10, 2)
    assert np.logical_and(draws > 0, draws < 1).all()