import os
from typing import List, Optional, Union, Tuple

import pandas as pd
import numpy as np


class DataProcessor:
    """
    Načíta dáta pre viaceré tickery, všetky stĺpce,
    zarovná podľa dátumu a vie ich prehodiť na 3D tvar (time, ticker, feature).
    """

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
        """
        Načíta všetky tickery, všetky stĺpce (alebo vybrané), zarovná podľa dátumu
        a vráti DataFrame s MultiIndex stĺpcami (ticker, feature).
        """
        frames = []
        for t in tickers:
            df_t = self.load_single_ticker(t, filter_cols=filter_cols)
            frames.append(df_t)

        panel_df = pd.concat(frames, axis=1, join=join)

        if start is not None:
            panel_df = panel_df[panel_df.index >= pd.to_datetime(start)]
        if end is not None:
            panel_df = panel_df[panel_df.index <= pd.to_datetime(end)]

        # pre istotu zoradíme MultiIndex (ticker, feature)
        panel_df = panel_df.sort_index(axis=1)

        return panel_df

    # ------------------------------------------------------------------

    def to_3d(
        self,
        panel_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, List[pd.Timestamp], List[str], List[str]]:
        """
        Z MultiIndex DataFrame (ticker, feature) spraví 3D numpy array.

        Výstup:
          data: shape = (T, N, F)
          dates: zoznam dátumov (T)
          tickers: zoznam tickerov (N)
          features: zoznam feature názvov (F)
        """
        if not isinstance(panel_df.columns, pd.MultiIndex) or panel_df.columns.nlevels != 2:
            raise ValueError("panel_df must have MultiIndex columns with levels (ticker, feature)")

        tickers = list(panel_df.columns.get_level_values(0).unique())
        features = list(panel_df.columns.get_level_values(1).unique())
        dates = list(panel_df.index)

        T = len(dates)
        N = len(tickers)
        F = len(features)

        data = np.zeros((T, N, F), dtype=float)

        # naplníme data[:, i, j] pre každý ticker a feature
        for i, t in enumerate(tickers):
            sub_df = panel_df[t]            # DataFrame s columns = features
            # zabezpečíme rovnaké poradie features
            sub_df = sub_df[features]
            data[:, i, :] = sub_df.values

        return data, dates, tickers, features