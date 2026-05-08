"""
Data loading and preprocessing utilities for the TD3 training pipeline.

Provides the DataProcessor class for loading per-ticker CSV files,
assembling them into a multi-ticker panel DataFrame, and converting
the panel into 3D numpy arrays suitable for environment consumption.

Author: Peter Likavec
"""

import os
from typing import List, Optional, Union, Tuple, Any

import pandas as pd
import numpy as np
from numpy import dtype, ndarray


class DataProcessor:
    """Utility class for loading and preprocessing stock data from CSV files."""

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
        """
        Builds a full file path for a given ticker.

        Args:
            ticker: The stock ticker symbol to build the file path for.

        Returns:
            Name of the CSV file corresponding to the ticker, constructed using the file pattern.
        """
        return os.path.join(self.data_dir, self.file_pattern.format(ticker=ticker))

    def _load_single_ticker(
        self,
        ticker: str,
        filter_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Loads a single ticker's CSV file into a DataFrame, optionally filtering out unwanted columns.

        Args:
            ticker: Name of the ticker to load.
            filter_cols: Columns to drop from the DataFrame after loading. If None, no columns are dropped.

        Returns:
            DataFrame containing the ticker's data, indexed by date, with columns wrapped in a MultiIndex (ticker, feature).
        """

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
        """
        Loads multiple tickers and concatenates them into a single panel DataFrame with MultiIndex columns.

        Args:
            tickers: List of ticker symbols to load.
            filter_cols: Columns to drop from each ticker's DataFrame after loading. If None, no columns are dropped.
            join: How to handle non-overlapping dates across tickers when concatenating.
            start: Optional start date to slice the panel DataFrame after concatenation.
            end: Optional end date to slice the panel DataFrame after concatenation.

        Returns:
            A DataFrame indexed by date, with MultiIndex columns (ticker, feature), containing the concatenated data for all tickers.
        """

        frames = []

        for t in tickers:
            frames.append(self._load_single_ticker(t, filter_cols=filter_cols))

        # concatenate all ticker DataFrames side by side (MultiIndex columns)
        panel_df = pd.concat(frames, axis=1, join=join)

        # optionally slice by date range
        panel_df = panel_df[
            (panel_df.index >= pd.to_datetime(start) if start is not None else True)
            & (panel_df.index <= pd.to_datetime(end) if end is not None else True)
            ]

        # sort columns alphabetically by (ticker, feature)
        panel_df = panel_df.sort_index(axis=1)
        return panel_df

    def load_data(self, app_config) -> Tuple[ndarray[tuple[int, int, int], dtype[Any]], ndarray[tuple[int, int, int], dtype[Any]], list[Any], list[Any], list[str]]:
        """
        Load and preprocess data according to the provided application configuration,
        returning 3D numpy arrays for features and prices, along with metadata.

        Args:
            app_config: The application configuration object containing ticker selection, date range, and feature filtering information.

        Returns:
            Tuple containing the 3D numpy array of features (T, N, F), the 3D numpy array of prices (T, N, 1),
            the list of tickers, the list of features, and the list of all dates in the panel.
        """

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
    def to_3d(panel_df: pd.DataFrame) -> Tuple[ndarray[tuple[int, int, int], dtype[Any]], list[str], list[str], list[str]]:
        """
        Converts a panel DataFrame with MultiIndex columns (ticker, feature) into a 3D numpy array.

        The resulting array has shape (T, N, F) where:
        - T = number of time steps (dates)
        - N = number of tickers
        - F = number of features

        Args:
            panel_df: DataFrame with MultiIndex columns (ticker, feature) and datetime index. Must have the same features for all tickers.

        Returns:
            A tuple containing the 3D numpy array of shape (T, N, F), the list of dates, the list of tickers, and the list of features.
        """

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
