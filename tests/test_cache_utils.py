from mace.cache_utils import BoundedLRUCache


def test_bounded_lru_cache_evicts_least_recently_used_entry():
    cache = BoundedLRUCache(max_entries=2)

    cache.store("first", 1)
    cache.store("second", 2)
    assert cache.get_lru("first") == 1
    cache.store("third", 3)

    assert list(cache) == ["first", "third"]
    assert "second" not in cache


def test_bounded_lru_cache_zero_capacity_does_not_store_entries():
    cache = BoundedLRUCache(max_entries=0)

    cache.store("first", 1)

    assert list(cache) == []
