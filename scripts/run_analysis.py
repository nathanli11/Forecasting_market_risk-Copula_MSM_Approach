"""Run lightweight analysis tables from a processed return dataset."""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import REPORTS_DIR, ensure_project_dirs
from src.risk import (
    historical_var,
    kupiec_pof_test,
    summary_statistics,
    var_exceedances,
    violation_rate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/returns.csv")
    parser.add_argument("--var-column", default=None)
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--output-prefix", default=REPORTS_DIR / "analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    returns = pd.read_csv(args.input, index_col=0, parse_dates=True)

    output_prefix = str(args.output_prefix)
    summary_statistics(returns).to_csv(f"{output_prefix}_descriptive.csv")
    print(f"Wrote {output_prefix}_descriptive.csv")

    if args.var_column is None:
        return

    series = returns[args.var_column].dropna()
    forecasts = series.rolling(args.window).apply(
        lambda window: historical_var(window, alpha=args.alpha),
        raw=False,
    )
    forecasts = forecasts.dropna().rename(f"{args.var_column}_hist_var")
    forecasts.to_csv(f"{output_prefix}_var.csv")

    aligned = pd.concat([series, forecasts], axis=1).dropna()
    hits = var_exceedances(aligned.iloc[:, 0], aligned.iloc[:, 1])
    test = kupiec_pof_test(hits, alpha=args.alpha)
    test["violation_rate"] = violation_rate(aligned.iloc[:, 0], aligned.iloc[:, 1])
    pd.Series(test).to_csv(f"{output_prefix}_backtest.csv")
    print(f"Wrote {output_prefix}_var.csv")
    print(f"Wrote {output_prefix}_backtest.csv")


if __name__ == "__main__":
    main()
