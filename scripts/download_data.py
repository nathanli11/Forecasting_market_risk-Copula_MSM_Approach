"""Download price data into data/raw.

Example:
    python scripts/download_data.py NASDAQCOM --source fred
    python scripts/download_data.py SP500 --source yahoo --start 2008-01-01
"""

from __future__ import annotations

import argparse

import pandas as pd
from pandas_datareader import data as pdr

from src.config import RAW_DATA_DIR, ensure_project_dirs
from src.data import (
    download_yahoo_index,
    download_yahoo_index_prices,
    resolve_yahoo_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="Market data symbol, e.g. NASDAQCOM.")
    parser.add_argument(
        "--source",
        default="fred",
        choices=["fred", "yahoo"],
        help="Data source.",
    )
    parser.add_argument("--start", default="1990-01-01", help="Start date.")
    parser.add_argument("--end", default=None, help="End date.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument(
        "--field",
        default="close",
        choices=["close", "adj_close"],
        help="Yahoo field for FRED-like output.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=["close", "adj_close"],
        default=None,
        help="Yahoo fields for a wider CSV, e.g. --fields close adj_close.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    if args.source == "yahoo":
        frame = _download_yahoo(args)
        output = args.output or _default_yahoo_output(args)
        pd.DataFrame(frame).to_csv(output, index=False)
    else:
        frame = pdr.DataReader(args.symbol, args.source, args.start, args.end)
        output = args.output or RAW_DATA_DIR / f"{args.symbol.lower()}.csv"
        pd.DataFrame(frame).rename_axis("observation_date").to_csv(output)
    print(f"Wrote {output}")


def _download_yahoo(args: argparse.Namespace) -> pd.DataFrame:
    if args.fields:
        return download_yahoo_index_prices(
            args.symbol,
            start=args.start,
            end=args.end,
            fields=tuple(args.fields),
        )
    return download_yahoo_index(
        args.symbol,
        start=args.start,
        end=args.end,
        field=args.field,
    )


def _default_yahoo_output(args: argparse.Namespace) -> str:
    spec = resolve_yahoo_index(args.symbol)
    if args.fields:
        suffix = "_".join(args.fields)
    else:
        suffix = str(args.field)
    return str(RAW_DATA_DIR / f"{spec.fred_series.lower()}_yahoo_{suffix}.csv")


if __name__ == "__main__":
    main()
