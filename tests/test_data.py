from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data import (
    align_return_frame,
    download_yahoo_index,
    download_yahoo_index_prices,
    log_returns,
    resolve_yahoo_index,
    save_yahoo_index_csv,
    simple_returns,
)


def test_returns_helpers() -> None:
    prices = pd.Series([100.0, 110.0, 121.0])
    assert np.allclose(simple_returns(prices).to_numpy(), [0.10, 0.10])
    assert np.isclose(log_returns(prices).iloc[0], np.log(1.1))


def test_align_return_frame_keeps_common_dates() -> None:
    dates = pd.date_range("2020-01-01", periods=3)
    left = pd.Series([0.1, 0.2, 0.3], index=dates)
    right = pd.Series([0.4, 0.5], index=dates[1:])
    result = align_return_frame({"left": left, "right": right})
    assert list(result.index) == list(dates[1:])
    assert list(result.columns) == ["left", "right"]


def test_resolve_yahoo_index_accepts_fred_and_yahoo_aliases() -> None:
    assert resolve_yahoo_index("NASDAQCOM").yahoo_symbol == "^IXIC"
    assert resolve_yahoo_index("^GSPC").fred_series == "SP500"
    assert resolve_yahoo_index("S&P 500").fred_series == "SP500"


def test_download_yahoo_index_returns_fred_like_schema() -> None:
    calls = []

    def fake_downloader(
        symbol: str,
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        calls.append((symbol, start, end))
        return pd.DataFrame(
            {"Close": [100.0, 101.0], "Adj Close": [99.5, 100.5]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )

    result = download_yahoo_index(
        "NASDAQCOM",
        start="2020-01-01",
        end="2020-01-03",
        field="close",
        downloader=fake_downloader,
    )

    assert calls == [("^IXIC", "2020-01-01", "2020-01-04")]
    assert result.columns.tolist() == ["observation_date", "NASDAQCOM"]
    assert result["NASDAQCOM"].tolist() == [100.0, 101.0]


def test_download_yahoo_index_prices_can_return_close_and_adjusted_close() -> None:
    def fake_downloader(
        symbol: str,
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {"Close": [4500.0], "Adj Close": [4499.5]},
            index=pd.to_datetime(["2020-01-02"]),
        )

    result = download_yahoo_index_prices(
        "SP500",
        fields=("close", "adj_close"),
        downloader=fake_downloader,
    )

    assert result.iloc[0].to_dict() == {
        "observation_date": "2020-01-02",
        "SP500_close": 4500.0,
        "SP500_adj_close": 4499.5,
    }


def test_save_yahoo_index_csv_writes_fred_like_file() -> None:
    def fake_downloader(
        symbol: str,
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        return pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2020-01-02"]))

    captured = {}

    def fake_to_csv(self: pd.DataFrame, path: Path, index: bool) -> None:
        captured["frame"] = self.copy()
        captured["path"] = path
        captured["index"] = index

    output = Path("nasdaq.csv")
    with patch.object(pd.DataFrame, "to_csv", fake_to_csv):
        saved = save_yahoo_index_csv(
            "NASDAQCOM",
            output,
            field="close",
            downloader=fake_downloader,
        )

    assert saved == output
    assert captured["path"] == output
    assert captured["index"] is False
    assert captured["frame"].to_dict("list") == {
        "observation_date": ["2020-01-02"],
        "NASDAQCOM": [100.0],
    }


def test_unknown_index_raises_helpful_error() -> None:
    with pytest.raises(ValueError, match="Known indexes"):
        resolve_yahoo_index("CAC40")
