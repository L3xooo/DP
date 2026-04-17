import pytest
from td3.utils.date_utils import check_if_later_date


@pytest.mark.parametrize("sooner_date, later_date", [
    ("2022-01-01", "2022-01-02"),
    ("2022-01-01", "2023-01-01"),
    ("2019-01-18", "2024-01-01"),
])
def test_check_if_later_date_valid(sooner_date, later_date):
    check_if_later_date(sooner_date, later_date)


@pytest.mark.parametrize("sooner_date, later_date", [
    ("2022-01-02", "2022-01-01"),
    ("2023-01-01", "2022-01-01"),
    ("2022-01-01", "2022-01-01"),
])
def test_check_if_later_date_raises(sooner_date, later_date):
    with pytest.raises(ValueError, match="must be strictly later than sooner_date"):
        check_if_later_date(sooner_date, later_date)