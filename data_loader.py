"""
Get datasets from local CSV (NASDAQ) and Yahoo Finance (SP500).
Process data transformations.

Sources :
    http://research.stlouisfed.org/fred2/series
    -> NASDAQ Composite : local CSV downloaded from FRED (NASDAQCOM.csv)
    -> S&P 500          : ^GSPC via Yahoo Finance

"""

import warnings
import numpy as np
import pandas as pd
import os
import yfinance as yf

warnings.filterwarnings("ignore")


FRED_SERIES = {
    "NASDAQ": "NASDAQCOM",
    "SP500":  "SP500",
}

class DataLoader:
    """
    Get, clean and process dataset (prices and returns).

    Parameters
    nasdaq_csv : str
        Path to NASDAQCOM.csv downloaded from FRED.
        Defaults to "data/NASDAQCOM.csv".
    sp500_csv : str
        Path to SP500.csv (exported after first download()).
        Defaults to "data/SP500.csv".
    start : str
        Start date "YYYY-MM-DD".
    end : str
        End date "YYYY-MM-DD".
    T_expected : int | None
        Expected number of trading days (validation after load).

    Attributes (after load() or download())
    prices : pd.DataFrame
        Daily closing prices aligned on common trading dates.
    returns : pd.DataFrame
        Log-returns x100 : r_t = 100 * ln(P_t / P_{t-1}).
    """

    def __init__(
        self,
        nasdaq_csv: str = "data/NASDAQCOM.csv",
        sp500_csv: str  = "data/SP500.csv",
        start: str      = "2009-04-15",
        end: str        = "2015-10-12",
        T_expected: int | None = 1635,
    ):
        self.nasdaq_csv = nasdaq_csv
        self.sp500_csv  = sp500_csv
        self.start      = start
        self.end        = end
        self.T_expected = T_expected

        self.prices: pd.DataFrame | None  = None
        self.returns: pd.DataFrame | None = None

    def load(self) -> "DataLoader":
        """
        Default entry point. Load both series from local CSV files,
        compute returns and validate T.
        Returns self for method chaining.

        Requires prices to have been downloaded at least once via download().
        """
        nasdaq = self._load_nasdaq_csv()
        sp500 = self._load_sp500_csv()

        self.prices = pd.concat([nasdaq, sp500], axis=1).dropna()
        self.returns = self._compute_returns(self.prices)
        self._validate_T()
        return self

    def download(self) -> "DataLoader":
        """
        Download SP500 from Yahoo Finance, load NASDAQ from local CSV,
        export SP500 to CSV, compute returns and validate T.
        Use this only on first run or to refresh data.
        Returns self for method chaining.
        """
        nasdaq = self._load_nasdaq_csv()
        sp500 = self._download_sp500()

        self.prices = pd.concat([nasdaq, sp500], axis=1).dropna()
        self.returns = self._compute_returns(self.prices)
        self._validate_T()
        return self

    @classmethod
    def from_csv(cls, prices_path: str, returns_path: str) -> "DataLoader":
        """
        Reload a DataLoader from prices/returns CSV files previously
        saved by save(). No network access required.

        Parameters
        prices_path  : str  path to prices.csv
        returns_path : str  path to returns.csv
        """
        instance = object.__new__(cls)
        instance.nasdaq_csv = ""
        instance.sp500_csv = ""
        instance.start = ""
        instance.end = ""
        instance.T_expected = None
        instance.prices = pd.read_csv(prices_path,  index_col=0, parse_dates=True)
        instance.returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        return instance

    def save(self, directory: str = ".") -> None:
        """Save prices.csv and returns.csv to directory."""
        self._check_loaded()
        os.makedirs(directory, exist_ok=True)
        self.prices.to_csv(f"{directory}/prices.csv")
        self.returns.to_csv(f"{directory}/returns.csv")

    def _load_nasdaq_csv(self) -> pd.Series:
        """
        Load NASDAQ prices from a local FRED CSV file (NASDAQCOM.csv).

        FRED CSV format:
            DATE,NASDAQCOM
            2009-04-15,1652.54
            2009-04-16,.          <- public holidays appear as "."
            ...
        """
        df = pd.read_csv(
            self.nasdaq_csv,
            index_col=0,
            parse_dates=True,
            na_values=".",      # FRED encodes missing values as "."
        )
        series = df.iloc[:, 0]                      # first (and only) value column
        series = series.loc[self.start:self.end]    # filter to requested window
        series = series.dropna()                    # drop public holidays
        series.name = "NASDAQ"
        return series

    def _load_sp500_csv(self) -> pd.Series:
        """
        Load SP500 prices from local CSV previously exported by download().
        """
        df = pd.read_csv(
            self.sp500_csv,
            index_col=0,
            parse_dates=True,
        )
        series = df.iloc[:, 0]
        series = series.loc[self.start:self.end]
        series = series.dropna()
        series.name = "SP500"
        return series

    def _download_sp500(self) -> pd.Series:
        """
        Download SP500 daily closing prices from Yahoo Finance (^GSPC)
        and export them to sp500_csv.

        FRED limits SP500 history to 10 years by contractual agreement
        with S&P Dow Jones Indices, so Yahoo Finance is used instead.
        """
        raw = yf.download(
            "^GSPC",
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
        )["Close"].squeeze().dropna()
        raw.name = "SP500"
        os.makedirs(os.path.dirname(self.sp500_csv) or ".", exist_ok=True)
        raw.to_csv(self.sp500_csv, index=True, header=True)
        return raw

    def _compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Log-returns x100 — Eq. (25) of the paper:
            r_t = 100 * ln(P_t / P_{t-1})
        """
        rets = 100 * np.log(prices / prices.shift(1)).dropna()
        rets.columns = ["r_" + col for col in prices.columns]
        return rets

    def _validate_T(self) -> None:
        """Check that the number of observations matches the paper."""
        T = len(self.returns)
        if self.T_expected is not None and T != self.T_expected:
            warnings.warn(
                f"[DataLoader] T = {T} but the paper reports {self.T_expected}. "
                "Check the date range and any residual NaN values.",
                UserWarning,
                stacklevel=2,
            )
        else:
            print(f"[DataLoader] T = {T} observations OK")

    def _check_loaded(self) -> None:
        """Raise if data has not been loaded yet."""
        if self.returns is None or self.prices is None:
            raise RuntimeError(
                "Data not loaded. Call load() or download() first."
            )

    def __repr__(self) -> str:
        T = len(self.returns) if self.returns is not None else "—"
        return (
            f"DataLoader("
            f"start='{self.start}', end='{self.end}', T={T})"
        )