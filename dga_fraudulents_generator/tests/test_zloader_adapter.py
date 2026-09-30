from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import GenerationError, _adaptive_batch_request, run_pipeline
from dga_fraudulents_dataset.models import AlgorithmInspection, AlgorithmPlan
from dga_fraudulents_dataset.utils import read_json
from dga_fraudulents_dataset.zloader_adapter import ZloaderAdapter


def _real_zloader_inspection() -> AlgorithmInspection:
    zpath = Path(__file__).resolve().parents[1] / "dga_algorithms" / "zloader"
    return AlgorithmInspection(
        algorithm_code="zloader",
        path=str(zpath),
        strategy="python_function",
        entrypoint=str(zpath / "dga.py"),
        callable_name="dga",
        parameter_strategy={"rc4_keys": ["q23Cud3xsNf3"], "date_mode": "daily_forward"},
    )


def _write_collision_zloader(root: Path, *, pool: int = 10) -> Path:
    z = root / "zloader"
    z.mkdir(parents=True, exist_ok=True)
    (z / "dga.py").write_text(
        "from datetime import datetime\n"
        "def seeding(d, key):\n"
        "    return 7\n"
        "def dga(seed, nr_of_domains):\n"
        "    for i in range(nr_of_domains):\n"
        f"        print(f'z{{(seed + i) % {pool}}}.com')\n",
        encoding="utf-8",
    )
    return z


def _collision_inspection(path: Path) -> AlgorithmInspection:
    return AlgorithmInspection(
        algorithm_code="zloader",
        path=str(path),
        strategy="python_function",
        entrypoint=str(path / "dga.py"),
        callable_name="dga",
        parameter_strategy={"rc4_keys": ["k1"], "date_mode": "daily_forward", "profile_offsets": 128, "profile_domains_per_offset": 64},
    )


def test_zloader_seed_change_affects_output():
    d1 = ZloaderAdapter._domains_from_seed(12345, 20)
    d2 = ZloaderAdapter._domains_from_seed(54321, 20)
    assert d1 != d2


def test_zloader_date_change_affects_output_when_supported():
    adapter = ZloaderAdapter(_real_zloader_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    r1 = adapter.generate(32)
    r2 = adapter.generate(32)
    assert r1.last_effective_params["date"] != r2.last_effective_params["date"]
    assert r1.domains != r2.domains


def test_zloader_native_batch_generation_used():
    adapter = ZloaderAdapter(_real_zloader_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    r = adapter.generate(64)
    assert r.generated == 64
    assert r.attempts == 1
    assert r.last_effective_params["native_batch_generation"] is True


def test_zloader_batch_size_shrinks_under_declining_yield():
    cfg = AppConfig(algorithms_root=Path("."), output_dir=Path("."), batch_size=5000)
    plan = AlgorithmPlan(
        algorithm_code="zloader",
        path="/tmp/zloader",
        category="seed_based",
        strategy="seed_date_collision_prone",
        entrypoint="/tmp/zloader/dga.py",
        callable_name="dga",
        required_params=["seed", "nr_of_domains"],
        default_params={},
        requires_seed=True,
        requires_date=False,
        target_count=10000,
        planned_quota=10000,
        effective_quota=10000,
        profiling={"collision_prone": True},
    )
    plan.last_batch_unique_yield_ratio = 0.01

    ask = _adaptive_batch_request(cfg, plan, remaining_global=10000)
    assert ask <= 25
    assert plan.adaptive_batch_mode in {"collision_probe", "collision_small"}


def test_zloader_saturates_or_exhausts_under_duplicate_dominance(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_collision_zloader(root, pool=6)

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=500,
        batch_size=100,
        checkpoint_every=20,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    z = stats["distribution_by_algorithm"]["zloader"]
    assert z["status"] in {"saturated", "exhausted"}
    assert z["profiling"]["collision_prone"] is True


def test_zloader_effective_quota_capped_by_observed_capacity(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_collision_zloader(root, pool=8)

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=10000,
        batch_size=500,
        checkpoint_every=100,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    z = stats["distribution_by_algorithm"]["zloader"]
    assert z["maximum_effective_quota"] <= z["profiling"]["estimated_max_unique_capacity"]
