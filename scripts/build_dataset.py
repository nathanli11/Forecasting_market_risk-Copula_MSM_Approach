"""Build a processed return dataset from one or more price CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROCESSED_DATA_DIR, ensure_project_dirs
from src.data import align_return_frame, load_price_csv, log_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", nargs="+", help="Input price CSV files.")
    parser.add_argument("--output", default=PROCESSED_DATA_DIR / "returns.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    returns = {}
    for csv_file in args.csv_files:
        path = Path(csv_file)
        prices = load_price_csv(path)
        returns[path.stem] = log_returns(prices)
    frame = align_return_frame(returns)
    frame.to_csv(args.output)
    print(f"Wrote {args.output} with shape {frame.shape}")


if __name__ == "__main__":
    main()
