"""
Data loading and preprocessing utilities for the TD3 training pipeline.

Provides the DataProcessor class for loading per-ticker CSV files,
assembling them into a multi-ticker panel DataFrame, and converting
the panel into 3D numpy arrays suitable for environment consumption.

Author: Peter Likavec
"""

import os
from typing import List, Optional, Union

import pandas as pd
import numpy as np


class DataProcessor:
    def __init__(
        self,
        data_dir: str,
        date_col: str = "date",
        file_pattern: str = "{ticker}/normalized.csv",
    ):
        self.data_dir = data_dir
        self.date_col = date_col
        self.file_pattern = file_pattern

    def _file_path(self, ticker: str) -> str:
        """Builds a full file path for a given ticker."""
        return os.path.join(self.data_dir, self.file_pattern.format(ticker=ticker))

    def load_single_ticker(
        self,
        ticker: str,
        filter_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Loads CSV for a single ticker and returns a MultiIndex DataFrame."""

        path = self._file_path(ticker)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file for ticker={ticker}: {path}")

        # load CSV and parse date column, then sort and set as index
        df = pd.read_csv(path, parse_dates=[self.date_col])
        df = df.sort_values(self.date_col).set_index(self.date_col)

        # drop unwanted columns if filter_cols is provided
        if filter_cols is not None:
            missing = [c for c in filter_cols if c not in df.columns]
            if missing:
                # raise early if any requested column doesn't exist
                raise ValueError(f"Columns not found in {ticker}: {missing}")
            df = df.drop(columns=filter_cols)

        # wrap columns in MultiIndex so each column is identified by (ticker, feature)
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        return df

    def load_panel(
        self,
        tickers: List[str],
        filter_cols: Optional[List[str]] = None,
        join: str = "inner",
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Loads multiple tickers and concatenates them into a single panel DataFrame."""

        frames = []

        for t in tickers:
            df_t = self.load_single_ticker(t, filter_cols=filter_cols)

            # skip tickers with unexpected shape (data quality check)
            if df_t.shape != (2243, 34):
                print(f"Skipping {t} shape {df_t.shape}")
                continue

            if df_t.empty:
                print(f"WARNING: {t} is empty")

            frames.append(df_t)

        # concatenate all ticker DataFrames side by side (MultiIndex columns)
        panel_df = pd.concat(frames, axis=1, join=join)

        # optionally slice by date range
        if start is not None:
            panel_df = panel_df[panel_df.index >= pd.to_datetime(start)]

        if end is not None:
            panel_df = panel_df[panel_df.index <= pd.to_datetime(end)]

        # sort columns alphabetically by (ticker, feature)
        panel_df = panel_df.sort_index(axis=1)
        return panel_df

    def load_data(self, app_config) -> tuple:
        """Loads panel data and converts it into 3D arrays for features and prices."""

        df = self.load_panel(
            tickers=app_config.ticker_config.tickers,
            start=app_config.start_date,
            end=app_config.end_date,
        )

        # slice out feature columns and price columns from the panel
        df_features = df.loc[:, (slice(None), app_config.filter_in)]
        df_prices = df.loc[:, (slice(None), ['close'])]

        # convert both to 3D arrays (T, N, F)
        data_3d_features, _, tickers, features = self.to_3d(df_features)
        data_3d_prices, _, _, _ = self.to_3d(df_prices)
        all_dates = df.index.get_level_values(0).unique().strftime('%Y-%m-%d').tolist()

        return data_3d_features, data_3d_prices, tickers, features, all_dates

    @staticmethod
    def to_3d(panel_df):
        """Converts a 2D panel DataFrame into a 3D numpy array (T, N, F)."""

        if not isinstance(panel_df.columns, pd.MultiIndex) or panel_df.columns.nlevels != 2:
            raise ValueError("panel_df must have MultiIndex columns (ticker, feature)")

        tickers = list(panel_df.columns.get_level_values(0).unique())

        # find features that exist across all tickers
        feature_sets = [set(panel_df[t].columns) for t in tickers]
        common_features = sorted(set.intersection(*feature_sets))
        if len(common_features) == 0:
            raise ValueError("No shared features across tickers")

        # keep only shared features to ensure consistent shape
        panel_df = panel_df.loc[:, (slice(None), common_features)]

        dates = list(panel_df.index)
        T = len(dates)
        N = len(tickers)
        F = len(common_features)

        # fill 3D array: axis 0 = time, axis 1 = ticker, axis 2 = feature
        data = np.zeros((T, N, F), dtype=float)

        for i, t in enumerate(tickers):
            sub_df = panel_df[t][common_features]

            if sub_df.shape[1] != F:
                raise ValueError(f"{t} feature mismatch {sub_df.shape[1]} vs {F}")

            data[:, i, :] = sub_df.values

        return data, dates, tickers, common_features
