from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

@dataclass
class PrevCurr(Generic[T]):
    prev: T
    curr: T

    def set_prev(self, value: T) -> None:
        self.prev = value

    def set_curr(self, value: T) -> None:
        self.curr = value

    def set_both(self, value: T) -> None:
        self.prev = value
        self.curr = value

    def set_prev_from_curr(self) -> None:
        self.prev = self.curr

    def update(self, new_curr: T) -> None:
        self.prev = self.curr
        self.curr = new_curr