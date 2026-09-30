from pathlib import Path

from dga_fraudulents_dataset.dedup import SQLiteDedupStore
from dga_fraudulents_dataset.writer import export_csv


def test_export_csv(tmp_path: Path):
    db = SQLiteDedupStore(tmp_path / "dedup.sqlite3")
    db.insert_many([("a.com", "algo"), ("b.net", "algo")])
    db.commit()
    out = tmp_path / "out.csv"
    total = export_csv(db, out)
    db.close()

    content = out.read_text(encoding="utf-8").splitlines()
    assert content[0] == "domain,algorithm_code"
    assert total == 2
    assert len(content) == 3
