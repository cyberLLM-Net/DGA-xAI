from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import (
    GenerationError,
    _adaptive_batch_request,
    run_pipeline,
)
from dga_fraudulents_dataset.models import AlgorithmInspection, AlgorithmPlan
from dga_fraudulents_dataset.monerodownloader_adapter import MoneroDownloaderAdapter
from dga_fraudulents_dataset.utils import read_json


def _real_monerodownloader_inspection(*, strategy: dict | None = None) -> AlgorithmInspection:
    mpath = Path(__file__).resolve().parents[1] / "dga_algorithms" / "monerodownloader"
    return AlgorithmInspection(
        algorithm_code="monerodownloader",
        path=str(mpath),
        strategy="python_function",
        entrypoint=str(mpath / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
    )


def _write_tiny_monerodownloader(root: Path, *, labels_per_day: int = 6, day_sensitive: bool = True) -> Path:
    m = root / "monerodownloader"
    m.mkdir(parents=True, exist_ok=True)
    day_line = (
        "        sld=hashlib.md5(f'{magic}-{days}-{nr}'.encode('ascii')).hexdigest()[:8]\n"
        if day_sensitive
        else "        sld=hashlib.md5(f'{magic}-CONST-{nr}'.encode('ascii')).hexdigest()[:8]\n"
    )
    script = "".join(
        [
            "from datetime import datetime\n",
            "import hashlib\n",
            "tlds=['.org','.tickets','.blackfriday','.hosting','.feedback']\n",
            "magic='x'\n",
            "special='const0'\n",
            "def dga(date, back=0):\n",
            "  epoch=datetime(1970,1,1)\n",
            "  days=(date-epoch).days\n",
            "  for j in range(back+1):\n",
            f"    for nr in range({labels_per_day}):\n",
            "      if nr==0:\n",
            "        sld=special\n",
            "      else:\n",
            day_line,
            "      for tld in tlds:\n",
            "        yield f'{sld}{tld}'\n",
            "    days -= 1\n",
        ]
    )
    (m / "dga.py").write_text(script, encoding="utf-8")
    return m


def test_monerodownloader_output_has_finite_combinatorial_structure():
    adapter = MoneroDownloaderAdapter(
        _real_monerodownloader_inspection(strategy={"max_day_offsets": 3, "profile_days": 3}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r = adapter.generate(50)
    tlds = {("." + d.split(".", 1)[1]) for d in r.domains}
    assert tlds == {".org", ".tickets", ".blackfriday", ".hosting", ".feedback"}
    assert r.last_effective_params["structured_space"] is True


def test_monerodownloader_capacity_estimate_is_computed():
    adapter = MoneroDownloaderAdapter(
        _real_monerodownloader_inspection(strategy={"max_day_offsets": 8, "profile_days": 4}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    p = adapter.profile()
    assert p["structured_finite_combination"] is True
    assert p["distinct_base_labels_estimate"] > 0
    assert p["estimated_total_combination_space"] >= p["sample_unique"]
    assert p["estimated_max_unique_capacity"] > 0


def test_monerodownloader_batch_shrinks_near_exhaustion():
    cfg = AppConfig(algorithms_root=Path("."), output_dir=Path("."), batch_size=5000)
    plan = AlgorithmPlan(
        algorithm_code="monerodownloader",
        path="/tmp/monerodownloader",
        category="seed_based",
        strategy="structured_date_index",
        entrypoint="/tmp/monerodownloader/dga.py",
        callable_name="dga",
        required_params=["date", "back"],
        default_params={},
        requires_seed=False,
        requires_date=True,
        target_count=10000,
        planned_quota=10000,
        effective_quota=10000,
        maximum_effective_quota=10000,
        unique_valid_count=9300,
        profiling={"structured_finite_combination": True},
    )
    plan.last_batch_unique_yield_ratio = 0.02

    ask = _adaptive_batch_request(cfg, plan, remaining_global=10000)
    assert ask <= 50
    assert plan.adaptive_batch_mode == "structured_probe"


def test_monerodownloader_duplicate_probe_marks_exhausted(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_tiny_monerodownloader(root, labels_per_day=4, day_sensitive=False)

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=1200,
        batch_size=100,
        checkpoint_every=20,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    m = stats["distribution_by_algorithm"]["monerodownloader"]
    assert m["status"] == "exhausted"
    assert m["exhaustion_reason"] in {"near_capacity_exhaustion", "structured_space_consumed"}


def test_monerodownloader_not_misclassified_as_broken_when_useful(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_tiny_monerodownloader(root, labels_per_day=10)

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=600,
        batch_size=120,
        checkpoint_every=20,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    m = stats["distribution_by_algorithm"]["monerodownloader"]
    assert m["inserted_unique_total"] > 0
    assert m["status"] != "discarded"
    assert m["discard_reason"] in {None, ""}
