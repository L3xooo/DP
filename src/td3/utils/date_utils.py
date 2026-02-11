from datetime import datetime

def check_if_later_date(date_a: str, date_b: str) -> None:
    """Raise ValueError if date_a (YYYY-MM-DD) is not later than date_b."""
    a = datetime.strptime(date_a, "%Y-%m-%d").date()
    b = datetime.strptime(date_b, "%Y-%m-%d").date()

    if a <= b:
        raise ValueError