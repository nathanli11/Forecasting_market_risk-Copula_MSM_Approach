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

- `data.py` contains data loading, Yahoo downloads, and return construction.
- `risk.py` contains statistics, GARCH, copula helpers, VaR, and backtesting.
- `plotting.py` contains Plotly helpers.
- `replication_workflow.ipynb` is the only notebook kept for exploration.

## Current Conventions

- Returns are represented as decimals, not percentages.
- VaR forecasts are positive loss numbers.
- A violation occurs when `-return > VaR`.
- Raw and generated data stay outside Git by default.
