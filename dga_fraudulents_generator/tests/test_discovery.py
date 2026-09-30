from pathlib import Path

from dga_fraudulents_dataset.discovery import discover_and_inspect


def test_discovery_and_inspection(tmp_path: Path):
    root = tmp_path / "algos"
    a1 = root / "algo1"
    a1.mkdir(parents=True)
    (a1 / "dga.py").write_text(
        "def dga(seed, date):\n    return 'a1example.com'\n",
        encoding="utf-8",
    )
    (a1 / "example_domains.txt").write_text("foo.com\nbar.net\n", encoding="utf-8")

    inspections = discover_and_inspect(root)
    assert len(inspections) == 1
    ins = inspections[0]
    assert ins.algorithm_code == "algo1"
    assert ins.callable_name == "dga"
    assert ins.entrypoint and ins.entrypoint.endswith("dga.py")
    assert ins.requires_seed is True
    assert ins.requires_date is True
    assert ".com" in ins.inferred_tlds


def test_placeholder_comments_only_is_rejected(tmp_path: Path):
    root = tmp_path / "algos"
    a1 = root / "placeholder"
    a1.mkdir(parents=True)
    (a1 / "dga.py").write_text(
        "# TODO placeholder implementation pending\n# pending\n",
        encoding="utf-8",
    )

    inspections = discover_and_inspect(root)
    ins = inspections[0]
    assert ins.status == "discarded"
    assert ins.discard_reason == "placeholder_module"
    assert "validation:placeholder_module" in ins.notes


def test_no_callable_and_no_real_logic_is_rejected(tmp_path: Path):
    root = tmp_path / "algos"
    a1 = root / "nocallable"
    a1.mkdir(parents=True)
    (a1 / "dga.py").write_text(
        "import math\n\nCONST = 5\n",
        encoding="utf-8",
    )

    inspections = discover_and_inspect(root)
    ins = inspections[0]
    assert ins.status == "discarded"
    assert ins.discard_reason == "missing_implementation"
    assert "validation:missing_implementation" in ins.notes


def test_placeholder_word_in_comment_does_not_change_discard_reason(tmp_path: Path):
    root = tmp_path / "algos"
    algorithm = root / "constant_only"
    algorithm.mkdir(parents=True)
    (algorithm / "dga.py").write_text(
        "# TODO: improve documentation of this constant\nVALUE = 5\n",
        encoding="utf-8",
    )

    inspection = discover_and_inspect(root)[0]
    assert inspection.status == "discarded"
    assert inspection.discard_reason == "missing_implementation"


def test_redirect_stub_is_rejected_with_unresolved_redirect(tmp_path: Path):
    root = tmp_path / "algos"
    a1 = root / "redirected"
    a1.mkdir(parents=True)
    (a1 / "dga.py").write_text(
        "# moved to https://example.com/real/dga.py\n",
        encoding="utf-8",
    )

    inspections = discover_and_inspect(root)
    ins = inspections[0]
    assert ins.status == "discarded"
    assert ins.discard_reason == "unresolved_redirect"
    assert "validation:unresolved_redirect" in ins.notes
