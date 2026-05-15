# Copula MSM VaR Replication

Replication project for Segnon and Trede's copula Markov-switching multifractal
approach to portfolio market-risk forecasting.

The project is intentionally compact: one small Python package, one workflow
notebook, and a few scripts that can be rerun from the command line.

## Layout

```text
copula-msm-var-replication/
|-- data/
|   |-- raw/              # Downloaded price CSVs
|   `-- processed/        # Return datasets built from raw prices
|-- notebooks/
|   `-- replication_workflow.ipynb
|-- src/
|   |-- config.py         # Project paths
|   |-- data.py           # CSV loading, Yahoo/FRED-style downloads, returns
|   |-- risk.py           # Statistics, GARCH, copulas, VaR, backtesting
|   `-- plotting.py       # Plotly figures
|-- scripts/
|   |-- download_data.py
|   |-- build_dataset.py
|   `-- run_analysis.py
|-- tests/
|-- reports/
`-- references/
```

## Setup With Conda

```powershell
conda env create -f environment.yml
conda activate copula-msm-var
python -m ipykernel install --user --name copula-msm-var --display-name "Python (copula-msm-var)"
```

## Setup Without Conda

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
```

## Pipeline

1. Download index prices into `data/raw/`.
2. Build aligned log-return series into `data/processed/returns.csv`.
3. Run descriptive statistics and simple VaR/backtesting outputs into `reports/`.
4. Use `notebooks/replication_workflow.ipynb` for inspection, figures, and model
   development.
5. Move durable code back into `src/` when it becomes reusable.

## Download Data

FRED:

```powershell
python scripts/download_data.py NASDAQCOM --source fred --start 2008-01-01 --end 2024-12-31 --output data/raw/NASDAQCOM.csv
```

Yahoo Finance, with FRED-like two-column output:

```powershell
python scripts/download_data.py NASDAQCOM --source yahoo --start 2008-01-01 --end 2024-12-31 --field close --output data/raw/NASDAQCOM_yahoo.csv
python scripts/download_data.py SP500 --source yahoo --start 2008-01-01 --end 2024-12-31 --field close --output data/raw/SP500_yahoo.csv
```

Yahoo mappings:

- `NASDAQCOM` -> `^IXIC`
- `SP500` -> `^GSPC`

To compare close and adjusted close in one wider CSV:

```powershell
python scripts/download_data.py SP500 --source yahoo --start 2008-01-01 --end 2024-12-31 --fields close adj_close --output data/raw/SP500_yahoo_close_adj_close.csv
```

## Build Returns

```powershell
python scripts/build_dataset.py data/raw/NASDAQCOM.csv data/raw/SP500_yahoo.csv --output data/processed/returns.csv
```

## Run Basic Analysis

```powershell
python scripts/run_analysis.py --input data/processed/returns.csv
python scripts/run_analysis.py --input data/processed/returns.csv --var-column NASDAQCOM --window 500 --alpha 0.01
```

## Development Checks

```powershell
python -m pytest
python -m ruff check .
python -m black --check .
```

Plotting uses Plotly. HTML and JSON exports work directly; static image exports
use `kaleido`.
