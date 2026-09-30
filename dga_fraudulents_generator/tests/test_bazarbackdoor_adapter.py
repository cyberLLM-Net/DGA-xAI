from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.bazarbackdoor_adapter import BazarBackdoorAdapter
from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import GenerationError, run_pipeline
from dga_fraudulents_dataset.models import AlgorithmInspection
from dga_fraudulents_dataset.utils import read_json


def _real_bazar_inspection(*, strategy: dict | None = None) -> AlgorithmInspection:
    bpath = Path(__file__).resolve().parents[1] / "dga_algorithms" / "bazarbackdoor"
    return AlgorithmInspection(
        algorithm_code="bazarbackdoor",
        path=str(bpath),
        strategy="python_function",
        entrypoint=str(bpath / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
    )


def _write_fake_bazar(root: Path, *, month_sensitive: bool = True, pool: int = 24) -> Path:
    b = root / "bazarbackdoor"
    b.mkdir(parents=True, exist_ok=True)
    (b / "dga.py").write_text(
        "from datetime import datetime\n"
        "versions = {'v2': ('p', []), 'v5': ('p', [])}\n"
        "def dga(date, version):\n"
        "    if version not in versions:\n"
        "        raise KeyError(version)\n"
        "    seed = date.strftime('%m%Y')\n"
        "    mv = int(seed[:2]) if '" + ("1" if month_sensitive else "0") + "' == '1' else 1\n"
        "    base = 40 if version == 'v2' else 28\n"
        "    for i in range(base):\n"
        f"        idx = (i + mv) % {pool}\n"
        "        yield f'{version}{idx:04x}.bazar'\n",
        encoding="utf-8",
    )
    (b / "domain_to_seed.py").write_text(
        "def revert(domain):\n"
        "    return 'reverse-only'\n",
        encoding="utf-8",
    )
    return b


def _fake_inspection(path: Path, *, strategy: dict | None = None) -> AlgorithmInspection:
    return AlgorithmInspection(
        algorithm_code="bazarbackdoor",
        path=str(path),
        strategy="python_function",
        entrypoint=str(path / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
    )


def test_bazar_explicit_version_validation_rejects_unsupported():
    adapter = BazarBackdoorAdapter(
        _real_bazar_inspection(strategy={"bazar_versions": ["v1", "v2", "v5"]}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    p = adapter.profile()
    assert "v1" in p["rejected_versions"]
    assert p["rejected_versions"]["v1"] == "unsupported_version"
    assert any(v in p["accepted_versions"] for v in ("v2", "v5"))


def test_bazar_per_version_profiling_present():
    adapter = BazarBackdoorAdapter(
        _real_bazar_inspection(strategy={"bazar_versions": ["v2", "v3", "v7"], "profile_months": 6}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    p = adapter.profile()
    keys = {x["version"] for x in p["per_version"]}
    assert {"v2", "v3", "v7"}.issubset(keys)


def test_bazar_capacity_estimation_for_duplicate_heavy_space(tmp_path: Path):
    b = _write_fake_bazar(tmp_path / "algos", month_sensitive=False, pool=10)
    adapter = BazarBackdoorAdapter(
        _fake_inspection(b, strategy={"bazar_versions": ["v2", "v5"], "profile_months": 8, "month_window": 12}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    p = adapter.profile()
    assert p["sample_unique_yield"] < 0.7
    assert p["estimated_max_unique_capacity"] <= int(max(1, p["sample_unique"] * 1.4))


def test_bazar_unsupported_versions_not_attempted_at_runtime(tmp_path: Path):
    bpath = _write_fake_bazar(tmp_path / "algos", month_sensitive=True, pool=32)
    adapter = BazarBackdoorAdapter(
        _fake_inspection(bpath, strategy={"bazar_versions": ["v1", "v2"]}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r = adapter.generate(120)
    assert "v1" in r.last_effective_params["rejected_versions"]
    assert "v1" not in r.last_effective_params["version_usage_slots"]


def test_bazar_saturates_or_exhausts_under_duplicate_dominance(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_fake_bazar(root, month_sensitive=False, pool=8)

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=2000,
        batch_size=200,
        checkpoint_every=50,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    b = stats["distribution_by_algorithm"]["bazarbackdoor"]
    assert b["status"] in {"saturated", "exhausted"}
    assert b["status"] != "discarded"
    if b["status"] == "exhausted":
        assert b["exhaustion_reason"] in {"near_capacity_exhaustion", "structured_space_consumed", "estimated_capacity_reached"}
