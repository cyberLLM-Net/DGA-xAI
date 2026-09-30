from __future__ import annotations

import json
from pathlib import Path

from dga_fraudulents_dataset.adapter_registry import get_adapter
from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.fosniw_adapter import FosniwAdapter
from dga_fraudulents_dataset.generator import GenerationError, _redistribute_from_algorithm, run_pipeline
from dga_fraudulents_dataset.models import AlgorithmInspection, AlgorithmPlan
from dga_fraudulents_dataset.utils import read_json
from dga_fraudulents_dataset.validation import validate_domain

DATE_ARGS = {
    "date_start": "2018-01-01",
    "date_end": "2030-12-31",
    "date_max_years_forward": 8,
    "date_max_years_backward": 8,
    "date_wrap_policy": "clamp",
}


def _real_fosniw_inspection(*, strategy: dict | None = None, defaults: dict | None = None) -> AlgorithmInspection:
    p = Path(__file__).resolve().parents[1] / "dga_algorithms" / "fosniw"
    return AlgorithmInspection(
        algorithm_code="fosniw",
        path=str(p),
        strategy="python_function",
        entrypoint=str(p / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
        default_params=defaults or {},
    )


def _write_fosniw_algo(root: Path) -> Path:
    out = root / "fosniw"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dga.py").write_text(
        "PATTERNS = {\n"
        "  'koreasys': 'appx.koreasys{}.com',\n"
        "  'winsoft': 'app2.winsoft{}.com',\n"
        "}\n"
        "def dga(prefix):\n"
        "  pat = PATTERNS.get(prefix)\n"
        "  if not pat:\n"
        "    raise ValueError('unsupported pattern {}'.format(prefix))\n"
        "  for i in range(101):\n"
        "    yield pat.format(i)\n",
        encoding="utf-8",
    )
    return out


def _fake_fosniw_inspection(path: Path, *, strategy: dict | None = None, defaults: dict | None = None) -> AlgorithmInspection:
    return AlgorithmInspection(
        algorithm_code="fosniw",
        path=str(path),
        strategy="python_function",
        entrypoint=str(path / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
        default_params=defaults or {},
    )


def test_fosniw_known_patterns_are_accepted():
    adapter = FosniwAdapter(_real_fosniw_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    p = adapter.profile()
    assert "koreasys" in p["supported_patterns"]
    assert "winsoft" in p["supported_patterns"]
    assert "koreasys" in p["accepted_patterns"]
    assert "winsoft" in p["accepted_patterns"]


def test_fosniw_unsupported_patterns_are_rejected_during_profiling():
    adapter = FosniwAdapter(
        _real_fosniw_inspection(defaults={"fosniw_patterns": ["al", "sn", "koreasys"]}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    p = adapter.profile()
    assert p["rejected_patterns"]["al"] == "unsupported_pattern"
    assert p["rejected_patterns"]["sn"] == "unsupported_pattern"
    assert "koreasys" in p["accepted_patterns"]


def test_fosniw_example_style_domains_generate_and_validate():
    adapter_ko = FosniwAdapter(
        _real_fosniw_inspection(defaults={"fosniw_patterns": ["koreasys"]}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    adapter_ws = FosniwAdapter(
        _real_fosniw_inspection(defaults={"fosniw_patterns": ["winsoft"]}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r1 = adapter_ko.generate(3)
    r2 = adapter_ws.generate(3)
    assert "appx.koreasys0.com" in r1.domains
    assert "app2.winsoft0.com" in r2.domains
    for d in (r1.domains + r2.domains):
        assert validate_domain(d).is_valid is True


def test_fosniw_is_finite_capacity_generator():
    adapter = FosniwAdapter(_real_fosniw_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    p = adapter.profile()
    assert p["structured_finite_combination"] is True
    assert p["estimated_max_unique_capacity"] <= 202
    r1 = adapter.generate(500)
    r2 = adapter.generate(10)
    assert r1.generated == p["estimated_max_unique_capacity"]
    assert r2.generated == 0


def test_fosniw_large_redistribution_not_assigned_when_capacity_capped():
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
    fosniw = AlgorithmPlan(
        algorithm_code="fosniw",
        path="/f",
        category="x",
        strategy="pattern_counter_structured",
        entrypoint=None,
        callable_name=None,
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=202,
        planned_quota=202,
        effective_quota=202,
        maximum_effective_quota=202,
        unique_valid_count=120,
        status="usable",
        redistribution_eligibility=True,
        capacity_score=1.0,
        profiling={"structured_finite_combination": True, "estimated_max_unique_capacity": 202},
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
        target_count=500,
        planned_quota=500,
        effective_quota=500,
        maximum_effective_quota=2000,
        unique_valid_count=300,
        status="usable",
        redistribution_eligibility=True,
        capacity_score=2.0,
    )

    moved = _redistribute_from_algorithm({"donor": donor, "fosniw": fosniw, "good": good}, "donor", "donor_down")
    assert moved > 0
    assert good.effective_quota > 500
    assert fosniw.effective_quota == 202


def test_fosniw_pipeline_caps_quota_and_reports_patterns(tmp_path: Path):
    root = tmp_path / "algos"
    _write_fosniw_algo(root)
    out = tmp_path / "results"
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(
        json.dumps(
            {
                "algorithm_overrides": {
                    "fosniw": {
                        "default_params": {"fosniw_patterns": ["koreasys", "winsoft"]},
                        "parameter_strategy": {"date_strategy": "daily_forward"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=500,
        batch_size=120,
        checkpoint_every=20,
        config_file=cfg_file,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    fs = stats["distribution_by_algorithm"]["fosniw"]
    assert fs["profiling"]["supported_patterns"] == ["koreasys", "winsoft"]
    assert fs["maximum_effective_quota"] <= 202
    assert fs["status"] in {"exhausted", "saturated", "partial", "usable"}


def test_fosniw_rejects_invalid_pattern_schedule_when_no_supported_patterns(tmp_path: Path):
    p = _write_fosniw_algo(tmp_path / "algos")
    adapter = FosniwAdapter(
        _fake_fosniw_inspection(p, defaults={"fosniw_patterns": ["al", "sn"]}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    profile = adapter.profile()
    assert profile["accepted_patterns"] == []
    assert profile["rejected_patterns"]["al"] == "unsupported_pattern"
    assert profile["rejected_patterns"]["sn"] == "unsupported_pattern"
    batch = adapter.generate(20)
    assert batch.generated == 0
    assert "invalid_pattern_schedule" in batch.errors


def test_fosniw_registry_routes_to_dedicated_adapter():
    adapter = get_adapter(
        _real_fosniw_inspection(),
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=5,
        batch_timeout_seconds=5,
        max_cli_invocations_per_batch=5,
        **DATE_ARGS,
    )
    assert isinstance(adapter, FosniwAdapter)
