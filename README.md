# Copula-MSM VaR Replication

Replication project for Segnon and Trede's copula-Markov switching multifractal
approach to portfolio market-risk forecasting.

The repository is centered on the reusable Python package in `src/` and the
workflow notebooks in `notebooks/`. The final notebook orchestrates the full
replication; the exploratory notebook is kept for reference.

## Repository Layout

```text
.
|-- data/
|   |-- raw/                         # Raw price CSV files
|   `-- processed/                   # Returns, PIT series, saved VaR panels
|       `-- var_forecasts/           # Rolling VaR outputs saved by the workflow
|-- notebooks/
|   |-- replication_workflow.ipynb   # Exploratory workflow
|   `-- replication_workflow_final.ipynb
|-- references/
|   |-- bibliography.bib
|   `-- paper_notes.md
|-- reports/
|   |-- figures/                     # Plotly figures exported as HTML
|   `-- tables/                      # CSV and LaTeX tables
|-- src/
|   |-- __init__.py                  # Public package API
|   |-- config.py                    # Project paths and output directories
|   |-- data.py                      # Data loading, downloads, returns
|   |-- msm.py                       # MSM estimation, filtering, PITs
|   |-- garch.py                     # GARCH(1,1) margins and diagnostics
|   |-- copulas.py                   # Copula likelihoods and estimation
|   |-- var.py                       # Rolling VaR forecasting functions
|   |-- risk.py                      # Summary stats and backtesting tests
|   |-- reporting.py                 # Table-building helpers
|   |-- plotting.py                  # Plotly figures
|   `-- utils.py                     # Small I/O helpers for saved VaR panels
|-- environment.yml
|-- requirements.txt
|-- requirements-dev.txt
`-- pyproject.toml
```

## What The Workflow Reproduces

The final notebook follows the paper's empirical design:

1. Load NASDAQ Composite and S&P 500 prices.
2. Compute percentage log returns: `100 * log(P_t / P_{t-1})`.
3. Estimate MSM marginal models for several values of `k`.
4. Estimate Gaussian GARCH(1,1) marginal models.
5. Transform marginal residuals and filtered returns into PIT series.
6. Estimate the copulas used in the paper for MSM and GARCH margins.
7. Compute one-step-ahead rolling VaR forecasts with historical simulation,
   variance-covariance, RiskMetrics, CCC-GARCH, Copula-GARCH, and Copula-MSM.
8. Evaluate VaR forecasts with Christoffersen LR tests, GMM duration tests, and
   Hansen SPA tests.

The VaR backtesting scheme uses a fixed rolling window with:

```python
WINDOW_SIZE = 1135
N_OOS = 500
ALPHA_5 = 0.05
ALPHA_1 = 0.01
WEIGHTS = [0.5, 0.5]
```

VaR is stored as a signed return quantile. A violation occurs when:

```python
portfolio_return_t < VaR_t
```

## Setup With Conda

The conda environment is meant to be a ready-to-run development setup.

```powershell
conda env create -f environment.yml
conda activate copula-msm-var
```

If you want a Jupyter kernel entry, register it once after activation:

```powershell
python -m ipykernel install --user --name copula-msm-var --display-name "Python (copula-msm-var)"
```

## Setup Without Conda

Use `requirements-dev.txt` for a full editable development install. It pulls in
the runtime dependencies from `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

If you only want the runtime dependencies, install `requirements.txt` instead.

## Input Data

The workflow expects the following raw files by default:

```text
data/raw/nasdaqcom_yahoo_close.csv
data/raw/sp500_yahoo_close.csv
```

Each file should contain a date column and a price column. The helper
`load_price_csv` automatically detects the date column and the first numeric price
column if they are not specified explicitly.

## Running The Workflow

Open and run:

```text
notebooks/replication_workflow_final.ipynb
```

The exploratory notebook remains available at `notebooks/replication_workflow.ipynb`.

The workflow saves intermediate and final outputs into:

```text
data/processed/
data/processed/var_forecasts/
reports/figures/
reports/tables/
```

The slowest block is the rolling Copula-MSM VaR section. For a quicker run, keep:

```python
MSM_COPULAS_TO_RUN = ["student", "gaussian"]
```

For the fuller replication, extend it to all supported copulas:

```python
MSM_COPULAS_TO_RUN = list(SUPPORTED_COPULAS)
```

## Main Outputs

Key outputs written by the workflow include:

```text
reports/tables/analysis_descriptive.csv
reports/tables/table_1_statistics.csv
reports/tables/table_1_statistics.tex
reports/tables/table_2_msm_estimates_raw.csv
reports/tables/table_2_msm_estimates.tex
reports/tables/table_3_garch_estimates_raw.csv
reports/tables/table_3_garch_estimates_formatted.csv
reports/tables/table_3_garch_estimates.tex
reports/tables/table_4_copula_estimates_raw.csv
reports/tables/table_4_copula_estimates_formatted.csv
reports/tables/table_4_copula_estimates.tex
reports/tables/table_5_lr_var_5pct_raw.csv
reports/tables/table_5_lr_var_5pct_formatted.csv
reports/tables/table_5_lr_var_5pct.tex
reports/tables/table_6_lr_var_1pct_raw.csv
reports/tables/table_6_lr_var_1pct_formatted.csv
reports/tables/table_6_lr_var_1pct.tex
reports/tables/table_7_gmm_var_5pct_raw.csv
reports/tables/table_8_gmm_var_1pct_raw.csv
reports/tables/table_9_spa_formatted.csv
```

## Notes On The Public API

The package uses an explicit `src/__init__.py`; it does not rely on `import *`.
That keeps notebook imports stable and avoids exposing private helpers.

Typical imports are:

```python
from src import (
    load_price_csv,
    log_returns,
    align_return_frame,
    fit_msm_grid,
    fit_garch_marginals,
    fit_copula_grid,
    forecast_msm_copula_var_rolling,
    make_lr_table,
    make_gmm_table,
    make_spa_table9,
)
```

## Reproducibility

Copula-MSM VaR forecasts are slow because the model is re-estimated inside each
rolling window and the portfolio VaR is solved numerically. The warm-start logic in
`var.py` only changes optimizer initialization; it does not change the rolling
window, likelihood, copula estimation, or VaR equation.

For long runs, save each VaR series as soon as it is computed. The notebook uses
`save_var`, `load_var`, and `concat_var_series` for this purpose.

## Development Checks

```powershell
python -m pytest
python -m ruff check .
python -m black --check .
```

Plotting uses Plotly. HTML exports work directly. Static image exports require
`kaleido`.
