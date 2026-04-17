from unittest.mock import MagicMock

from td3.data.data_processor import DataProcessor

FIXTURES_DIR = "./tests/fixtures"

def test_data_processor_load_panel():
    dp = DataProcessor(data_dir=FIXTURES_DIR, file_pattern="{ticker}.csv")
    df = dp.load_panel(
        tickers=["aa", "bb", "cc"],
        start="2024-01-01",
        end="2024-01-30",
    )

    data_3d_features, _, tickers, features = dp.to_3d(df)

    assert not df.empty
    assert tickers == ["aa", "bb", "cc"]
    assert features == ["a", "b", "c", "close", "d", "e"]


def test_data_processor_load_panel_with_filter():
    dp = DataProcessor(data_dir=FIXTURES_DIR, file_pattern="{ticker}.csv")
    df = dp.load_panel(
        tickers=["aa", "bb", "cc"],
        filter_cols=["c", "d", "e", "close"],
        start="2024-01-01",
        end="2024-01-30",
    )

    data_3d_features, _, tickers, features = dp.to_3d(df)

    assert not df.empty
    assert tickers == ["aa", "bb", "cc"]
    assert features == ["a", "b"]



def test_data_processor_load_data():
    mock_config = MagicMock()
    mock_config.data_dir = FIXTURES_DIR
    mock_config.ticker_config.tickers = ["aa", "bb", "cc"]
    mock_config.filter_in = ["a", "b", "c", "d", "e"]
    mock_config.start_date = "2024-01-01"
    mock_config.end_date = "2024-02-19"

    dp = DataProcessor(data_dir=mock_config.data_dir, file_pattern="{ticker}.csv")
    data_3d_features, data_3d_prices, tickers, features, dates = dp.load_data(mock_config)

    assert tickers == ["aa", "bb", "cc"]
    assert data_3d_features.ndim == 3
    assert data_3d_features.shape[0] == len(dates)
    assert data_3d_features.shape[1] == 3
    assert data_3d_prices.ndim == 3
    assert data_3d_prices.shape[2] == 1
    assert len(dates) > 0