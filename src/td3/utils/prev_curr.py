"""
Generic container for tracking previous and current state values.

Provides the PrevCurr dataclass for maintaining a rolling two-state
buffer, useful for tracking transitions in environment observations.

Author: Peter Likavec
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class PrevCurr(Generic[T]):
    """Generic container holding a previous and current value of the same type."""

    prev: T
    curr: T

    def set_prev(self, value: T) -> None:
        """Set the previous value."""
        self.prev = value

    def set_curr(self, value: T) -> None:
        """Set the current value."""
        self.curr = value

    def set_both(self, value: T) -> None:
        """Set both previous and current values to the same value."""
        self.prev = value
        self.curr = value

    def set_prev_from_curr(self) -> None:
        """Set the previous value to the current value, effectively shifting the buffer."""
        self.prev = self.curr

    def update(self, new_curr: T) -> None:
        """Update the current value and shift the previous value to the old current."""
        self.prev = self.curr
        self.curr = new_curr
