"""Small input/output helpers for saved VaR forecast panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_var(series: pd.Series, output_dir: str | Path, filename: str) -> Path:
    """Save one VaR forecast series as CSV and return the output path."""
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame(name=series.name or "VaR").to_csv(path)
    return path


def load_var(output_dir: str | Path, filename: str, column_name: str | None = None) -> pd.Series:
    """Load one saved VaR forecast series from a CSV file."""
    path = Path(output_dir) / filename
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    series = frame.iloc[:, 0]
    if column_name is not None:
        series.name = column_name
    return series


def concat_var_series(series_by_name: dict[str, pd.Series], expected_length: int | None = None) -> pd.DataFrame:
    """Concatenate named VaR forecasts and align them on common dates."""
    frame = pd.concat({name: series for name, series in series_by_name.items()}, axis=1).dropna(how="any")
    if expected_length is not None and len(frame) != expected_length:
        raise ValueError(f"Expected {expected_length} observations, got {len(frame)}.")
    return frame
