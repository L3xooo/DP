from td3.data.data_processor import DataProcessor


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

def test_data_processor_load_data_with_filter():
    dp = DataProcessor(data_dir="./tests/fixtures", file_pattern="{ticker}.csv")
    df = dp.load_panel(
        tickers=["aa", "bb", "cc"],
        filter_cols=["c", "d", "e", "f"],
        start="2024-01-01",
        end="2024-01-30",
    )

    data_3d_features, _, tickers, features = dp.to_3d(df)

    assert not df.empty
    assert tickers == ["aa", "bb", "cc"]
    assert features == ["a", "b"]