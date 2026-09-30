from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import GenerationError, run_pipeline
from dga_fraudulents_dataset.models import AlgorithmInspection
from dga_fraudulents_dataset.qsnatch_adapter import QsnatchAdapter
from dga_fraudulents_dataset.utils import read_json


def _real_qsnatch_inspection(*, strategy: dict | None = None) -> AlgorithmInspection:
    qpath = Path(__file__).resolve().parents[1] / "dga_algorithms" / "qsnatch"
    return AlgorithmInspection(
        algorithm_code="qsnatch",
        path=str(qpath),
        strategy="python_function",
        entrypoint=str(qpath / "dga_a.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
    )


def _write_fake_qsnatch(root: Path) -> Path:
    q = root / "qsnatch"
    q.mkdir(parents=True, exist_ok=True)
    (q / "dga_a.py").write_text(
        "def dga(date):\n"
        "    d = getattr(date, 'toordinal', lambda: 0)()\n"
        "    base = f'a{d % 17:x}'\n"
        "    yield f'{base}..com.bn'\n"
        "    yield f'{base}.net'\n"
        "    yield f'{base}.org'\n",
        encoding="utf-8",
    )
    (q / "dga_b.py").write_text(
        "def dga(date):\n"
        "    d = getattr(date, 'toordinal', lambda: 0)()\n"
        "    for i in range(8):\n"
        "        yield f'b{d % 97:x}{i}.com'\n",
        encoding="utf-8",
    )
    return q


def _fake_inspection(path: Path, *, strategy: dict | None = None) -> AlgorithmInspection:
    return AlgorithmInspection(
        algorithm_code="qsnatch",
        path=str(path),
        strategy="python_function",
        entrypoint=str(path / "dga_a.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
    )


def test_qsnatch_no_double_dot_domains_after_sanitization():
    adapter = QsnatchAdapter(_real_qsnatch_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    r = adapter.generate(500)
    assert r.generated > 0
    assert all(".." not in d for d in r.domains)


def test_qsnatch_suffix_join_handles_multilevel_tlds():
    adapter = QsnatchAdapter(_real_qsnatch_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    r = adapter.generate(800)
    assert all(".." not in d for d in r.domains)
    assert any(d.endswith(".com.bn") for d in r.domains)


def test_qsnatch_variant_a_b_profiled_separately():
    adapter = QsnatchAdapter(
        _real_qsnatch_inspection(strategy={"qsnatch_variant": "both"}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    p = adapter.profile()
    keys = [v["key"] for v in p["variants"]]
    assert set(keys) == {"a", "b"}


def test_qsnatch_invalid_rate_diagnostics_persisted(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_fake_qsnatch(root)

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=120,
        batch_size=60,
        checkpoint_every=20,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    q = stats["distribution_by_algorithm"]["qsnatch"]
    assert "invalid_reason_counts" in q
    assert q["invalid_reason_counts"].get("double_dot", 0) > 0
    assert q["profiling"]["invalid_rate"] >= 0.0


def test_qsnatch_correlated_traversal_reduced_via_day_scramble(tmp_path: Path):
    qpath = _write_fake_qsnatch(tmp_path / "algos")
    adapter = QsnatchAdapter(
        _fake_inspection(qpath, strategy={"day_window": 97, "qsnatch_variant": "both"}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r1 = adapter.generate(50)
    r2 = adapter.generate(50)
    d1 = r1.last_effective_params.get("day_offsets_used", [])
    d2 = r2.last_effective_params.get("day_offsets_used", [])
    assert len(set(d1 + d2)) > 4
    assert max(d1 + d2) - min(d1 + d2) >= 20


def test_qsnatch_quota_capped_by_variant_performance(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_fake_qsnatch(root)

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=5000,
        batch_size=300,
        checkpoint_every=100,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    q = stats["distribution_by_algorithm"]["qsnatch"]
    assert q["maximum_effective_quota"] <= q["profiling"]["estimated_max_unique_capacity"]
