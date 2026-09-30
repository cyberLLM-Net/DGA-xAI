from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.adapter_registry import get_adapter
from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.dmsniff_adapter import DmsniffAdapter
from dga_fraudulents_dataset.generator import GenerationError, run_pipeline
from dga_fraudulents_dataset.models import AlgorithmInspection
from dga_fraudulents_dataset.utils import read_json

DATE_ARGS = {
    "date_start": "2018-01-01",
    "date_end": "2030-12-31",
    "date_max_years_forward": 8,
    "date_max_years_backward": 8,
    "date_wrap_policy": "clamp",
}


def _real_dmsniff_inspection(*, strategy: dict | None = None, defaults: dict | None = None) -> AlgorithmInspection:
    p = Path(__file__).resolve().parents[1] / "dga_algorithms" / "dmsniff"
    return AlgorithmInspection(
        algorithm_code="dmsniff",
        path=str(p),
        strategy="python_function",
        entrypoint=str(p / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
        default_params=defaults or {},
    )


def _write_fake_dmsniff(root: Path) -> Path:
    d = root / "dmsniff"
    d.mkdir(parents=True, exist_ok=True)
    (d / "dga.py").write_text(
        "def dga(prefix):\n"
        "  if prefix == 'al':\n"
        "    primes=[1,3,5,7,11,13]\n"
        "  elif prefix == 'sn':\n"
        "    primes=[1,7,3,5,11,13]\n"
        "  else:\n"
        "    raise ValueError('unsupported prefix')\n"
        "  tlds=['.com','.org','.net','.ru','.in']\n"
        "  for nr in range(1,51):\n"
        "    core=''.join(chr(((p*nr)>>1)%24 + ord('a')) for p in primes)\n"
        "    yield prefix + core + tlds[(nr>>1)%5]\n",
        encoding="utf-8",
    )
    return d


def _fake_dmsniff_inspection(path: Path, *, strategy: dict | None = None, defaults: dict | None = None) -> AlgorithmInspection:
    return AlgorithmInspection(
        algorithm_code="dmsniff",
        path=str(path),
        strategy="python_function",
        entrypoint=str(path / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
        default_params=defaults or {},
    )


def test_dmsniff_detected_as_low_capacity_structured():
    adapter = DmsniffAdapter(_real_dmsniff_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    p = adapter.profile()
    assert p["low_capacity_structured"] is True
    assert p["low_structural_diversity"] is True
    assert set(p["detected_families"]) == {"al", "sn"}
    assert p["estimated_max_unique_capacity"] <= 100


def test_dmsniff_quota_is_capped_aggressively_in_plan(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_fake_dmsniff(root)
    cfg = AppConfig(algorithms_root=root, output_dir=out, target_count=10000, batch_size=500, checkpoint_every=100)
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass
    plan = read_json(out / "udcdga_dga_generation_plan.json")
    d = next(x for x in plan["algorithms"] if x["algorithm_code"] == "dmsniff")
    assert d["maximum_effective_quota"] <= 500
    assert d["profiling"]["low_capacity_structured"] is True


def test_dmsniff_early_exhaustion_under_duplicate_domination(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_fake_dmsniff(root)
    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=300,
        batch_size=120,
        checkpoint_every=40,
        exhausted_after_zero_unique_batches=2,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass
    stats = read_json(out / "udcdga_dga_domains_stats.json")
    d = stats["distribution_by_algorithm"]["dmsniff"]
    assert d["status"] in {"saturated", "exhausted"}
    if d["status"] == "exhausted":
        assert d["exhaustion_reason"] in {"finite_capacity_exhaustion", "near_capacity_exhaustion", "estimated_capacity_reached", "structured_space_consumed"}
    assert d["maximum_effective_quota"] <= 500
    assert d["inserted_unique_total"] <= d["maximum_effective_quota"]


def test_dmsniff_family_level_profile_is_persisted(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_fake_dmsniff(root)
    cfg = AppConfig(algorithms_root=root, output_dir=out, target_count=90, batch_size=60, checkpoint_every=30)
    run_pipeline(cfg)
    stats = read_json(out / "udcdga_dga_domains_stats.json")
    d = stats["distribution_by_algorithm"]["dmsniff"]
    prof = d["profiling"]
    assert set(prof["detected_families"]) == {"al", "sn"}
    assert "family_capacity_estimates" in prof
    assert "duplicate_growth_curve" in prof


def test_dmsniff_registry_routes_to_dedicated_adapter():
    adapter = get_adapter(
        _real_dmsniff_inspection(),
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=5,
        batch_timeout_seconds=5,
        max_cli_invocations_per_batch=5,
        **DATE_ARGS,
    )
    assert isinstance(adapter, DmsniffAdapter)
