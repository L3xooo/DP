import pytest

from td3.data.data_processor import DataProcessor
import numpy as np


def test_data_processor_load_data():
    dp = DataProcessor(data_dir="./tests/fixtures", file_pattern="{ticker}.csv")
    df = dp.load_panel(
        tickers=["aa", "bb", "cc"],
        start="2024-01-01",
        end="2024-01-30",
    )

    data_3d_features, _, tickers, features = dp.to_3d(df)

    assert not df.empty
    assert tickers == ["aa", "bb", "cc"]
    assert features == ["a", "b", "c", "d", "e", "f"]


def test_data_processor_get_only_specific_column():
    dp = DataProcessor(data_dir="./tests/fixtures", file_pattern="{ticker}.csv")
    df = dp.load_panel(
        tickers=["aa", "bb", "cc"],
        start="2024-01-01",
        end="2024-01-30",
    )
    df_a = df.loc[:, (slice(None), ['a'])]
    data_3d_a, _, _, _ = dp.to_3d(df_a)

    assert data_3d_a.shape == (30, 3, 1)
    assert np.all(data_3d_a == 1), "Expected all values in column 'a' to be 1"


@pytest.mark.parametrize(
    "start,end,expected_T,expected_value",
    [
        ("2024-01-01", "2024-01-30", 30, 1),
        ("2024-1-31", "2024-02-19", 20, 2),
    ],
    ids=["jan_30", "jan_31"],
)
def test_data_processor_3d_shape_and_contains_value(start, end, expected_T, expected_value):
    dp = DataProcessor(data_dir="./tests/fixtures", file_pattern="{ticker}.csv")
    df = dp.load_panel(
        tickers=["aa", "bb", "cc"],
        start=start,
        end=end,
    )

    data_3d_features, _, tickers, features = dp.to_3d(df)

    assert data_3d_features.shape == (expected_T, 3, 6)
    assert np.any(data_3d_features == expected_value), f"Expected to contain value={expected_value}"
