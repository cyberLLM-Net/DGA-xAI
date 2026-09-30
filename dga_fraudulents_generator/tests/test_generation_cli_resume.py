from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.cli import main
from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import run_pipeline
from dga_fraudulents_dataset.utils import read_json


GOOD_ALGO = """
def dga(seed, nr):
    return f"{seed}{nr}.example.com"
"""

BAD_ALGO = """
def dga(seed):
    raise RuntimeError('boom')
"""


def _write_algorithms(root: Path) -> None:
    good = root / "goodalgo"
    bad = root / "badalgo"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    (good / "dga.py").write_text(GOOD_ALGO, encoding="utf-8")
    (good / "example_domains.txt").write_text("a.com\nb.net\n", encoding="utf-8")
    (bad / "dga.py").write_text(BAD_ALGO, encoding="utf-8")


def test_run_pipeline_and_resume(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_algorithms(root)

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=100,
        batch_size=20,
        checkpoint_every=25,
    )
    payload = run_pipeline(cfg)
    assert payload["unique_domains"] == 100
    assert (out / "udcdga_dga_domains.csv").exists()
    assert (out / "udcdga_dga_domains_stats.json").exists()
    assert (out / "udcdga_dga_generation_plan.json").exists()

    cfg2 = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=100,
        batch_size=20,
        checkpoint_every=25,
        resume=True,
    )
    payload2 = run_pipeline(cfg2)
    assert payload2["unique_domains"] >= 100


def test_cli_dry_run(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_algorithms(root)

    rc = main(
        [
            "--algorithms-root",
            str(root),
            "--output-dir",
            str(out),
            "--target-count",
            "200",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert (out / "udcdga_dga_domains.csv").exists()


def test_unresolved_redirect_discarded_without_attempts(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    good = root / "goodalgo"
    expiro = root / "expiro"
    good.mkdir(parents=True)
    expiro.mkdir(parents=True)
    (good / "dga.py").write_text(GOOD_ALGO, encoding="utf-8")
    (expiro / "dga.py").write_text(
        "# moved to https://github.com/example/real-expiro/dga.py\n",
        encoding="utf-8",
    )

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=40,
        batch_size=10,
        checkpoint_every=10,
    )
    run_pipeline(cfg)
    stats = read_json(out / "udcdga_dga_domains_stats.json")
    plan = read_json(out / "udcdga_dga_generation_plan.json")

    expiro_stats = stats["distribution_by_algorithm"]["expiro"]
    assert expiro_stats["status"] == "discarded"
    assert expiro_stats["discard_reason"] == "unresolved_redirect"
    assert expiro_stats["attempts_total"] == 0
    assert expiro_stats["generated_total"] == 0
    assert all("low_yield" not in str(item) for item in stats["errors_by_algorithm"].get("expiro", []))

    expiro_plan = [a for a in plan["algorithms"] if a["algorithm_code"] == "expiro"][0]
    assert expiro_plan["discard_reason"] == "unresolved_redirect"
