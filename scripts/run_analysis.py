"""Run lightweight analysis tables from a processed return dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import TABLES_DIR, ensure_project_dirs
from src.risk import (
    christoffersen_lr_test,
    historical_var,
    kupiec_pof_test,
    summary_statistics,
    var_exceedances,
    violation_rate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        default="data/processed/returns.csv",
        help="Path to a CSV file containing return series.",
    )
    parser.add_argument(
        "--var-column",
        default=None,
        help="Column on which to run a rolling historical VaR backtest.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=500,
        help="Rolling window length for historical VaR.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="VaR tail probability.",
    )
    parser.add_argument(
        "--output-prefix",
        default=str(TABLES_DIR / "analysis"),
        help="Output prefix for generated CSV files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    returns = pd.read_csv(input_path, index_col=0, parse_dates=True)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    descriptive_path = output_prefix.with_name(output_prefix.name + "_descriptive.csv")

    summary_statistics(returns).to_csv(descriptive_path)
    print(f"Wrote {descriptive_path}")

    if args.var_column is None:
        return

    if args.var_column not in returns.columns:
        available = ", ".join(returns.columns)
        raise ValueError(
            f"Unknown --var-column '{args.var_column}'. "
            f"Available columns: {available}."
        )

    series = returns[args.var_column].dropna()

    forecasts = series.rolling(args.window).apply(
        lambda window: historical_var(window, alpha=args.alpha),
        raw=False,
    )

    forecasts = forecasts.dropna().rename(f"{args.var_column}_hist_var")

    var_path = output_prefix.with_name(output_prefix.name + "_var.csv")
    forecasts.to_csv(var_path)

    aligned = pd.concat([series, forecasts], axis=1).dropna()

    hits = var_exceedances(
        returns=aligned.iloc[:, 0],
        var_forecasts=aligned.iloc[:, 1],
    )

    kupiec = kupiec_pof_test(hits, alpha=args.alpha)
    christoffersen = christoffersen_lr_test(hits, alpha=args.alpha)

    backtest = {
        "violation_rate": violation_rate(aligned.iloc[:, 0], aligned.iloc[:, 1]),
        **{f"kupiec_{key}": value for key, value in kupiec.items()},
        **{f"christoffersen_{key}": value for key, value in christoffersen.items()},
    }

    backtest_path = output_prefix.with_name(output_prefix.name + "_backtest.csv")
    pd.Series(backtest).to_csv(backtest_path)

    print(f"Wrote {var_path}")
    print(f"Wrote {backtest_path}")


if __name__ == "__main__":
    main()