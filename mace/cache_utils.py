from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class BoundedLRUCache(OrderedDict[K, V], Generic[K, V]):
    def __init__(self, *, max_entries: int) -> None:
        super().__init__()
        self.max_entries = int(max_entries)

    def get_lru(self, key: K) -> V | None:
        value = self.get(key)
        if value is not None:
            self.move_to_end(key)
        return value

    def store(self, key: K, value: V) -> None:
        if self.max_entries <= 0:
            self.pop(key, None)
            return
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.max_entries:
            self.popitem(last=False)
