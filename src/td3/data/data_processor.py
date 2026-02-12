import os
from typing import List, Optional, Union, Tuple

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

    # ------------------------------------------------------------------

    def _file_path(self, ticker: str) -> str:
        return os.path.join(self.data_dir, self.file_pattern.format(ticker=ticker))

    # ------------------------------------------------------------------

    def load_single_ticker(
        self,
        ticker: str,
        filter_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Načíta všetky stĺpce pre daný ticker (CSV), voliteľne ich prefiltroval.
        Výstup má MultiIndex stĺpce: (ticker, column_name).
        """
        path = self._file_path(ticker)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file for ticker={ticker}: {path}")

        df = pd.read_csv(path, parse_dates=[self.date_col])
        df = df.sort_values(self.date_col).set_index(self.date_col)

        # filter stĺpcov ak chceš len niektoré (napr. ["Open","High","Low","Close"])
        if filter_cols is not None:
            missing = [c for c in filter_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Columns not found in {ticker}: {missing}")
            # df = df[filter_cols]
            df = df.drop(columns=filter_cols)

        # spravíme MultiIndex: (ticker, feature)
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])

        return df

    # ------------------------------------------------------------------

    def load_panel(
            self,
            tickers: List[str],
            filter_cols: Optional[List[str]] = None,
            join: str = "inner",
            start: Optional[Union[str, pd.Timestamp]] = None,
            end: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:

        frames = []

        for t in tickers:
            df_t = self.load_single_ticker(t, filter_cols=filter_cols)

            # print(f"{t} shape:", df_t.shape)

            if df_t.shape != (2243, 34):
                print(f"Skipping {t} shape {df_t.shape}")
                continue

            if df_t.empty:
                print(f"WARNING: {t} is empty")

            frames.append(df_t)

        print("Loaded frames:", len(frames))

        panel_df = pd.concat(frames, axis=1, join=join)

        print("After concat:", panel_df.shape)

        if start is not None:
            panel_df = panel_df[panel_df.index >= pd.to_datetime(start)]
            print("After start filter:", panel_df.shape)

        if end is not None:
            panel_df = panel_df[panel_df.index <= pd.to_datetime(end)]
            print("After end filter:", panel_df.shape)

        panel_df = panel_df.sort_index(axis=1)

        print("Final panel:", panel_df.shape)

        return panel_df

    # ------------------------------------------------------------------

    def to_3d(self, panel_df):

        if not isinstance(panel_df.columns, pd.MultiIndex) or panel_df.columns.nlevels != 2:
            raise ValueError("panel_df must have MultiIndex columns (ticker, feature)")

        tickers = list(panel_df.columns.get_level_values(0).unique())

        # --- find common features across ALL tickers ---
        feature_sets = [set(panel_df[t].columns) for t in tickers]
        common_features = sorted(set.intersection(*feature_sets))

        if len(common_features) == 0:
            raise ValueError("No shared features across tickers")

        # filter dataframe to shared features only
        panel_df = panel_df.loc[:, (slice(None), common_features)]

        dates = list(panel_df.index)
        T = len(dates)
        N = len(tickers)
        F = len(common_features)

        data = np.zeros((T, N, F), dtype=float)

        for i, t in enumerate(tickers):
            sub_df = panel_df[t][common_features]

            if sub_df.shape[1] != F:
                raise ValueError(f"{t} feature mismatch {sub_df.shape[1]} vs {F}")

            data[:, i, :] = sub_df.values

        return data, dates, tickers, common_features

