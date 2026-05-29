# Segnon And Trede Paper Notes

Use this file to keep the paper-to-code mapping tight while the replication is
being finalized.

## Bibliographic Anchor

- Segnon, J. and Trede, M. Copula-Markov switching multifractal market-risk
  forecasting paper used as the target replication.

## Facts To Confirm From The Paper

- Asset universe and sample period.
- Return definition and data cleaning rules.
- MSM parameterization and estimation method.
- Copula families and selection criteria.
- VaR levels and forecasting windows.
- Backtesting tests and loss functions.
- Tables and figures to reproduce.

## Code Map

- `src/data.py` for prices and returns.
- `src/msm.py` for MSM margins and PITs.
- `src/garch.py` for GARCH margins.
- `src/copulas.py` for copula estimation.
- `src/var.py` for rolling VaR.
- `src/reporting.py` for tables 1 to 9.
- `notebooks/replication_workflow_final.ipynb` for the end-to-end workflow.

## Open Checks

- Confirm whether the paper uses percentage log returns or simple returns.
- Confirm whether the VaR sign convention matches the current code.
- Confirm the exact copula list and whether rotated versions are required.
- Confirm the exact names of the reproduced tables and whether any are omitted.
