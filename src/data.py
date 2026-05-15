"""Data download, loading, and return construction helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

PriceField = Literal["close", "adj_close"]
YahooDownloader = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class YahooIndexSpec:
    """Mapping between FRED-style series names and Yahoo tickers."""

    fred_series: str
    yahoo_symbol: str
    description: str


YAHOO_INDEXES: dict[str, YahooIndexSpec] = {
    "NASDAQCOM": YahooIndexSpec("NASDAQCOM", "^IXIC", "NASDAQ Composite"),
    "SP500": YahooIndexSpec("SP500", "^GSPC", "S&P 500"),
}

_ALIASES = {
    "NASDAQ": "NASDAQCOM",
    "NASDAQCOM": "NASDAQCOM",
    "NASDAQ_COMPOSITE": "NASDAQCOM",
    "IXIC": "NASDAQCOM",
    "^IXIC": "NASDAQCOM",
    "SP500": "SP500",
    "S&P500": "SP500",
    "S&P_500": "SP500",
    "SANDP500": "SP500",
    "GSPC": "SP500",
    "^GSPC": "SP500",
}

_YAHOO_FIELD_NAMES: dict[PriceField, str] = {
    "close": "Close",
    "adj_close": "Adj Close",
}


def load_price_csv(
    path: str | Path,
    date_column: str | None = None,
    price_column: str | None = None,
) -> pd.Series:
    """Load a price series from a CSV file.

    If the date or price columns are not provided, the function uses the first
    column that looks like a date and the first numeric column respectively.
    """
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV file is empty: {path}")

    if date_column is None:
        date_column = _infer_date_column(frame)
    if price_column is None:
        price_column = _infer_price_column(frame, exclude={date_column})

    series = pd.Series(
        frame[price_column].to_numpy(),
        index=pd.to_datetime(frame[date_column]),
    )
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    series.name = price_column
    return series


def simple_returns(prices: pd.Series | Iterable[float]) -> pd.Series:
    """Compute simple percentage returns from prices."""
    series = _as_series(prices)
    return series.pct_change().dropna()


def log_returns(prices: pd.Series | Iterable[float]) -> pd.Series:
    """Compute log returns from strictly positive prices."""
    series = _as_series(prices)
    if (series <= 0).any():
        raise ValueError("Log returns require strictly positive prices.")
    return np.log(series).diff().dropna()


def align_return_frame(series_by_name: dict[str, pd.Series]) -> pd.DataFrame:
    """Combine named return series on their common dates."""
    if not series_by_name:
        raise ValueError("At least one return series is required.")
    frame = pd.concat(series_by_name, axis=1).dropna(how="any")
    frame.index.name = "date"
    return frame


def resolve_yahoo_index(identifier: str) -> YahooIndexSpec:
    """Resolve a FRED series name or Yahoo ticker to a known index."""
    key = _normalise_identifier(identifier)
    try:
        return YAHOO_INDEXES[_ALIASES[key]]
    except KeyError as exc:
        known = ", ".join(sorted(YAHOO_INDEXES))
        raise ValueError(
            f"Unknown Yahoo index '{identifier}'. Known indexes: {known}."
        ) from exc


def download_yahoo_index(
    identifier: str,
    start: str | None = None,
    end: str | None = None,
    field: PriceField = "close",
    downloader: YahooDownloader | None = None,
) -> pd.DataFrame:
    """Download one Yahoo index field with the same schema as a FRED CSV."""
    _validate_fields((field,))
    spec = resolve_yahoo_index(identifier)
    raw = _download_raw_yahoo(spec, start=start, end=end, downloader=downloader)
    series = _select_price_series(raw, spec, field)
    return _series_to_fred_like_frame(series, spec.fred_series)


def download_yahoo_index_prices(
    identifier: str,
    start: str | None = None,
    end: str | None = None,
    fields: Iterable[PriceField] = ("close", "adj_close"),
    downloader: YahooDownloader | None = None,
) -> pd.DataFrame:
    """Download one or more Yahoo price fields for an index."""
    fields = _validate_fields(fields)
    spec = resolve_yahoo_index(identifier)
    raw = _download_raw_yahoo(spec, start=start, end=end, downloader=downloader)
    output = pd.DataFrame()

    for field in fields:
        series = _select_price_series(raw, spec, field)
        frame = _series_to_fred_like_frame(series, f"{spec.fred_series}_{field}")
        if output.empty:
            output = frame
        else:
            output = output.merge(frame, on="observation_date", how="outer")

    return output.sort_values("observation_date").reset_index(drop=True)


def save_yahoo_index_csv(
    identifier: str,
    output_path: str | Path,
    start: str | None = None,
    end: str | None = None,
    field: PriceField = "close",
    fields: Iterable[PriceField] | None = None,
    downloader: YahooDownloader | None = None,
) -> Path:
    """Download Yahoo index data and save it as a CSV file."""
    if fields is None:
        frame = download_yahoo_index(
            identifier,
            start=start,
            end=end,
            field=field,
            downloader=downloader,
        )
    else:
        frame = download_yahoo_index_prices(
            identifier,
            start=start,
            end=end,
            fields=fields,
            downloader=downloader,
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _as_series(values: pd.Series | Iterable[float]) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(list(values), dtype=float)


def _infer_date_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        lowered = str(column).lower()
        if "date" in lowered or "time" in lowered:
            return str(column)
    return str(frame.columns[0])


def _infer_price_column(frame: pd.DataFrame, exclude: set[str]) -> str:
    candidates = [column for column in frame.columns if str(column) not in exclude]
    numeric_candidates = [
        str(column)
        for column in candidates
        if pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    if not numeric_candidates:
        raise ValueError("Could not infer a numeric price column.")
    return numeric_candidates[0]


def _download_raw_yahoo(
    spec: YahooIndexSpec,
    start: str | None,
    end: str | None,
    downloader: YahooDownloader | None,
) -> pd.DataFrame:
    if downloader is None:
        downloader = _default_yfinance_download

    raw = downloader(
        spec.yahoo_symbol,
        start=start,
        end=_inclusive_end_to_yfinance_end(end),
    )
    if raw.empty:
        raise ValueError(
            f"Yahoo Finance returned no rows for {spec.yahoo_symbol} "
            f"between {start} and {end}."
        )
    return _flatten_yfinance_columns(raw, spec.yahoo_symbol)


def _default_yfinance_download(
    yahoo_symbol: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "Install the `yfinance` package to download Yahoo Finance data."
        ) from exc

    _configure_yfinance_cache(yf)
    return yf.download(
        yahoo_symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )


def _configure_yfinance_cache(yf_module) -> None:
    cache_dir = Path(os.getenv("YFINANCE_CACHE_DIR", _default_yfinance_cache_dir()))
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf_module, "set_tz_cache_location"):
        yf_module.set_tz_cache_location(str(cache_dir))


def _default_yfinance_cache_dir() -> Path:
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "src" / "yfinance_cache"
    return Path.home() / ".cache" / "src" / "yfinance_cache"


def _select_price_series(
    raw: pd.DataFrame,
    spec: YahooIndexSpec,
    field: PriceField,
) -> pd.Series:
    yahoo_field = _YAHOO_FIELD_NAMES[field]
    if yahoo_field not in raw.columns:
        available = ", ".join(str(column) for column in raw.columns)
        raise ValueError(
            f"Yahoo field '{yahoo_field}' is missing for {spec.yahoo_symbol}. "
            f"Available columns: {available}."
        )

    series = pd.to_numeric(raw[yahoo_field], errors="coerce").dropna()
    series.name = spec.fred_series
    return series.sort_index()


def _series_to_fred_like_frame(series: pd.Series, column_name: str) -> pd.DataFrame:
    dates = pd.to_datetime(series.index).strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "observation_date": dates,
            column_name: series.to_numpy(),
        }
    )


def _flatten_yfinance_columns(frame: pd.DataFrame, yahoo_symbol: str) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    flattened = frame.copy()
    for level in range(flattened.columns.nlevels):
        values = flattened.columns.get_level_values(level)
        if yahoo_symbol in values:
            return flattened.xs(yahoo_symbol, axis=1, level=level)

    while isinstance(flattened.columns, pd.MultiIndex):
        dropped = False
        for level in range(flattened.columns.nlevels):
            values = flattened.columns.get_level_values(level)
            if len(set(values)) == 1:
                flattened.columns = flattened.columns.droplevel(level)
                dropped = True
                break
        if not dropped:
            break

    if isinstance(flattened.columns, pd.MultiIndex):
        flattened.columns = [
            "_".join(str(part) for part in column if part)
            for column in flattened.columns
        ]
    return flattened


def _inclusive_end_to_yfinance_end(end: str | None) -> str | None:
    if end is None:
        return None
    return (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _validate_fields(fields: Iterable[str]) -> tuple[PriceField, ...]:
    selected = tuple(fields)
    if not selected:
        raise ValueError("At least one Yahoo price field is required.")

    unknown = sorted(set(selected) - set(_YAHOO_FIELD_NAMES))
    if unknown:
        known = ", ".join(_YAHOO_FIELD_NAMES)
        raise ValueError(
            f"Unknown Yahoo price fields {unknown}. Known fields: {known}."
        )
    return selected


def _normalise_identifier(identifier: str) -> str:
    return (
        identifier.strip()
        .upper()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "")
    )
