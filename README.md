# Copula-MSM VaR Replication

Final replication project for Segnon and Trede's Copula-Markov Switching
Multifractal approach to portfolio market-risk forecasting.

The project is organized around a small `src/` package and one final workflow
notebook. The notebook is written in English and is intended to orchestrate the
replication steps; reusable logic should remain in the Python modules.

## Repository layout

```text
copula-msm-var-replication/
|-- data/
|   |-- raw/                         # Raw price CSV files
|   `-- processed/                   # Cleaned prices, returns, PIT series
|-- reports/
|   |-- figures/                     # Plotly figures exported as HTML
|   `-- tables/                      # CSV and LaTeX tables
|       `-- var_forecasts/           # Saved rolling VaR series
|-- references/                      # Replicated paper and related references
|-- src/
|   |-- __init__.py                  # Explicit public API
|   |-- config.py                    # Project paths and output directories
|   |-- data.py                      # Data loading, downloads, returns
|   |-- msm.py                       # MSM estimation, filtering, PITs
|   |-- garch.py                     # GARCH(1,1) margins and diagnostics
|   |-- copulas.py                   # Copula likelihoods and estimation
|   |-- var.py                       # Rolling VaR forecasting functions
|   |-- risk.py                      # Summary stats and backtesting tests
|   |-- reporting.py                 # Tables 5-9 helpers
|   |-- plotting.py                  # Plotly figures
|   `-- utils.py                     # Small I/O helpers for saved VaR panels
|-- replication_workflow_final_english.ipynb
|-- README.md
```

## Methodology reproduced

The final notebook follows the paper's empirical design:

1. Load NASDAQ Composite and S&P 500 prices.
2. Compute percentage log returns: `100 * log(P_t / P_{t-1})`.
3. Estimate MSM marginal models for several values of `k`.
4. Estimate Gaussian GARCH(1,1) marginal models.
5. Transform marginal residuals/filtered returns into PIT series.
6. Estimate the copulas used in the paper for MSM and GARCH margins.
7. Compute one-step-ahead rolling VaR forecasts with:
   - Historical Simulation,
   - Variance-Covariance,
   - RiskMetrics,
   - CCC-GARCH,
   - Copula-GARCH,
   - Copula-MSM.
8. Evaluate VaR forecasts using:
   - Christoffersen LR tests,
   - GMM duration-based tests,
   - Hansen SPA tests.

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

## Setup with Conda

```powershell
conda env create -f environment.yml
conda activate copula-msm-var
python -m ipykernel install --user --name copula-msm-var --display-name "Python (copula-msm-var)"
```

## Setup without Conda

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If development dependencies are separated in `requirements-dev.txt`, install them
with:

```powershell
python -m pip install -r requirements-dev.txt
```

## Input data expected by the final notebook

The notebook expects the following raw files by default:

```text
data/raw/nasdaqcom_yahoo_close.csv
data/raw/sp500_yahoo_close.csv
```

Each file should contain a date column and a price column. The helper
`load_price_csv` automatically detects the date column and the first numeric price
column if they are not specified explicitly.

## Running the final workflow

Open and run:

```text
replication_workflow_final_english.ipynb
```

The workflow saves intermediate and final outputs into:

```text
data/processed/
reports/figures/
reports/tables/
reports/tables/var_forecasts/
```

The most computationally expensive block is the rolling Copula-MSM VaR block. For
a quick run, keep:

```python
MSM_COPULAS_TO_RUN = ["student", "gaussian"]
```

For a fuller replication, extend it to all supported copulas:

```python
MSM_COPULAS_TO_RUN = list(SUPPORTED_COPULAS)
```

## Main output files

The workflow produces:

```text
reports/tables/table_1_statistics_formatted.csv
reports/tables/table_2_msm_estimates_raw.csv
reports/tables/table_3_garch_estimates_formatted.csv
reports/tables/table_4_copula_estimates_formatted.csv
reports/tables/table_5_lr_var_5pct_formatted.csv
reports/tables/table_6_lr_var_1pct_formatted.csv
reports/tables/table_7_gmm_var_5pct_raw.csv
reports/tables/table_8_gmm_var_1pct_raw.csv
reports/tables/table_9_spa_formatted.csv
```

It also exports LaTeX versions of the LR tables:

```text
reports/tables/table_5_lr_var_5pct.tex
reports/tables/table_6_lr_var_1pct.tex
```

## Public API

The package uses an explicit `src/__init__.py`; it does not rely on `import *`.
This keeps the notebook imports stable and avoids exposing private helper
functions unintentionally.

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

## Notes on reproducibility and runtime

Copula-MSM VaR forecasts are slow because the model is re-estimated inside each
rolling window and the portfolio VaR is solved numerically. The warm-start logic in
`var.py` only changes optimizer initialization; it does not change the rolling
window, likelihood, copula estimation, or VaR equation.

For long runs, save each VaR series as soon as it is computed. The notebook uses
`save_var`, `load_var`, and `concat_var_series` for this purpose.

## Development checks

```powershell
python -m pytest
python -m ruff check .
python -m black --check .
```

Plotting uses Plotly. HTML exports work directly. Static image exports require
`kaleido`.
