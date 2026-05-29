# Replication Notes

## Objective

Replicate the paper's portfolio market-risk forecasting workflow:

1. Build clean asset return series.
2. Estimate marginal volatility models, including GARCH baselines and MSM models.
3. Transform marginal innovations into pseudo-observations.
4. Estimate dependence through copulas.
5. Produce one-step-ahead VaR forecasts.
6. Backtest VaR forecasts and compare competing models.

## Current Structure

- `src/data.py` contains data loading, Yahoo downloads, and return construction.
- `src/msm.py` contains MSM estimation and PIT construction.
- `src/garch.py` contains GARCH marginals and diagnostics.
- `src/copulas.py` contains copula likelihoods and estimation.
- `src/var.py` contains rolling VaR forecasting.
- `src/risk.py` contains summary statistics and backtesting helpers.
- `src/reporting.py` contains table builders for the replication outputs.
- `src/plotting.py` contains Plotly helpers.
- `notebooks/replication_workflow_final.ipynb` is the main workflow notebook.
- `notebooks/replication_workflow.ipynb` is kept as an exploratory notebook.

## Current Conventions

- Returns are percentage log returns, not decimals.
- VaR forecasts are signed return quantiles.
- A violation occurs when `portfolio_return < VaR`.
- Rolling VaR series are saved under `data/processed/var_forecasts/`.
- Tables and figures are written under `reports/tables/` and `reports/figures/`.

## Paper Mapping

- Verify the asset universe and sample period against the paper.
- Verify the MSM `k` grid and the copula families used for each margin.
- Verify the rolling window, out-of-sample length, and alpha levels.
- Verify which tables should be reproduced from the current code paths.
- Keep a short list of any deviations from the paper, if they remain intentional.
