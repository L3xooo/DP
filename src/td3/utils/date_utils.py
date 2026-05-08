"""
Date validation utilities for the TD3 training pipeline.

Provides helper functions for validating and comparing date strings
used in environment configuration and data loading.

Author: Peter Likavec
"""

from datetime import datetime


def check_if_later_date(sooner_date: str, later_date: str) -> None:
    """Validates that later_date is strictly later than sooner_date.

    Args:
        sooner_date: The earlier of the two dates, in YYYY-MM-DD format.
        later_date: The date that must be strictly later than sooner_date, in YYYY-MM-DD format.

    Raises:
        ValueError: If later_date is not strictly later than sooner_date.
    """
    a = datetime.strptime(sooner_date, "%Y-%m-%d").date()
    b = datetime.strptime(later_date, "%Y-%m-%d").date()

    if b <= a:
        raise ValueError(
            f"later_date ({later_date}) must be strictly later than sooner_date ({sooner_date})."
        )
