from pathlib import Path

from dga_fraudulents_dataset.dedup import SQLiteDedupStore


def test_sqlite_dedup_insert(tmp_path: Path):
    store = SQLiteDedupStore(tmp_path / "dedup.sqlite3")
    s1 = store.insert_many([
        ("a.com", "x"),
        ("b.com", "x"),
    ])
    assert s1.inserted == 2
    s2 = store.insert_many([
        ("a.com", "y"),
        ("c.com", "x"),
    ])
    assert s2.inserted == 1
    assert store.count_unique() == 3
    store.close()

    reopened = SQLiteDedupStore(tmp_path / "dedup.sqlite3")
    duplicate = reopened.insert_many([("a.com", "other")])
    assert duplicate.inserted == 0
    assert reopened.count_unique() == 3
    assert reopened.count_by_algorithm() == {"x": 3}
    reopened.close()
