from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import GenerationError, _redistribute_from_algorithm, run_pipeline
from dga_fraudulents_dataset.invocation import _structural_variability_metrics
from dga_fraudulents_dataset.models import AlgorithmPlan, AlgorithmInspection
from dga_fraudulents_dataset.planner import build_generation_plan
from dga_fraudulents_dataset.utils import read_json


def _write_constant_suffix_algo(root: Path) -> Path:
    b = root / "banjori"
    b.mkdir(parents=True, exist_ok=True)
    (b / "dga.py").write_text(
        "def map_to_lowercase_letter(s):\n"
        "  return ord('a') + ((s - ord('a')) % 26)\n"
        "def next_domain(domain):\n"
        "  dl = [ord(x) for x in list(domain)]\n"
        "  dl[0] = map_to_lowercase_letter(dl[0] + dl[3])\n"
        "  dl[1] = map_to_lowercase_letter(dl[0] + 2*dl[1])\n"
        "  dl[2] = map_to_lowercase_letter(dl[0] + dl[2] - 1)\n"
        "  dl[3] = map_to_lowercase_letter(dl[1] + dl[2] + dl[3])\n"
        "  return ''.join([chr(x) for x in dl])\n"
        "seed = 'earnestnessbiophysicalohax.com'\n",
        encoding="utf-8",
    )
    return b


def test_detects_constant_suffix_low_diversity_structure():
    domains = [f"{chr(ord('a') + i)}aaaestnessbiophysicalohax.com" for i in range(16)]
    m = _structural_variability_metrics(domains)
    assert m["low_structural_diversity"] is True
    assert m["suffix_constancy"] > 0.6
    assert m["constant_suffix"].endswith("estnessbiophysicalohax.com")


def test_banjori_quota_is_capped_from_profile():
    ins = AlgorithmInspection(
        algorithm_code="banjori",
        path="/tmp/banjori",
        category="seed_based",
        profile={
            "initial_capacity_score": 0.3,
            "initial_health_score": 0.4,
            "expected_diversity_score": 0.1,
            "recommended_max_effective_quota_multiplier": 1.0,
            "low_structural_diversity": True,
            "theoretical_capacity_estimate": 456_976,
            "recommended_effective_quota_cap": 180_000,
            "estimated_max_unique_capacity": 456_976,
        },
    )
    plans = build_generation_plan([ins], target_count=2_000_000)
    p = plans["banjori"]
    assert p.maximum_effective_quota <= 180_000
    assert p.effective_quota <= 180_000
    assert p.redistribution_eligibility is False


def test_redistribution_skips_saturated_low_diversity_banjori():
    donor = AlgorithmPlan(
        algorithm_code="donor",
        path="/d",
        category="x",
        strategy="python_function",
        entrypoint=None,
        callable_name=None,
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=1000,
        planned_quota=1000,
        effective_quota=1000,
        unique_valid_count=0,
        status="discarded",
    )
    ban = AlgorithmPlan(
        algorithm_code="banjori",
        path="/b",
        category="x",
        strategy="python_function",
        entrypoint=None,
        callable_name=None,
        required_params=[],
        default_params={},
        requires_seed=True,
        requires_date=False,
        target_count=180000,
        planned_quota=180000,
        effective_quota=180000,
        maximum_effective_quota=180000,
        unique_valid_count=90000,
        status="saturated",
        redistribution_eligibility=False,
        capacity_score=0.2,
        profiling={"low_structural_diversity": True},
    )
    good = AlgorithmPlan(
        algorithm_code="good",
        path="/g",
        category="x",
        strategy="python_function",
        entrypoint=None,
        callable_name=None,
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=500000,
        planned_quota=500000,
        effective_quota=500000,
        maximum_effective_quota=900000,
        unique_valid_count=300000,
        status="usable",
        redistribution_eligibility=True,
        capacity_score=2.0,
    )
    moved = _redistribute_from_algorithm({"donor": donor, "banjori": ban, "good": good}, "donor", "banjori_sat")
    assert moved > 0
    assert ban.effective_quota == 180000
    assert good.effective_quota > 500000


def test_banjori_stats_include_structural_metrics_and_capacity_cap(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "out"
    _write_constant_suffix_algo(root)
    cfg = AppConfig(algorithms_root=root, output_dir=out, target_count=120000, batch_size=4000, checkpoint_every=20000)
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass
    stats = read_json(out / "udcdga_dga_domains_stats.json")
    b = stats["distribution_by_algorithm"]["banjori"]
    assert b["suffix_constancy"] > 0.5
    assert b["estimated_capacity"] > 100000
    assert b["maximum_effective_quota"] <= 200000
