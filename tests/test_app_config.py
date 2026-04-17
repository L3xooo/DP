import pytest
from pathlib import Path

from td3.config.app_config import AppConfig

CONFIG_PATH = Path("tests/fixtures/config/test_config.json")


@pytest.mark.parametrize("start_date, end_date", [
    (None, None),
    ("2022-01-01", None),
    (None, "2023-01-01"),
])
def test_raises_type_error_when_start_date_or_end_date_is_missing(start_date, end_date):
    with pytest.raises(TypeError):
        AppConfig.load_from_train_config(CONFIG_PATH, start_date=start_date, end_date=end_date)

def test_start_date_after_end_date_raises():
    with pytest.raises(ValueError):
        AppConfig.load_from_train_config(
            CONFIG_PATH,
            start_date="2024-01-01",
            end_date="2023-01-01",
        )