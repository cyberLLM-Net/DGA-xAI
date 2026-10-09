from __future__ import annotations

import json
from pathlib import Path

from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import (
    _adaptive_batch_request,
    _evaluate_algorithm_health,
    _redistribute_from_algorithm,
    GenerationError,
    run_pipeline,
)
from dga_fraudulents_dataset.invocation import CliSubprocessAdapter, ParameterExplorer, PythonFunctionAdapter
from dga_fraudulents_dataset.models import AlgorithmInspection, AlgorithmPlan
from dga_fraudulents_dataset.utils import read_json

DATE_ARGS = {
    "date_start": "2018-01-01",
    "date_end": "2030-12-31",
    "date_max_years_forward": 8,
    "date_max_years_backward": 8,
    "date_wrap_policy": "clamp",
}


def _inspection_for(path: Path, code: str = "algo") -> AlgorithmInspection:
    return AlgorithmInspection(
        algorithm_code=code,
        path=str(path.parent),
        strategy="python_function",
        entrypoint=str(path),
        callable_name="dga",
        required_params=["seed", "nr"],
        requires_seed=True,
        requires_date=False,
    )


def test_scalar_generator_expands_to_batch_attempts(tmp_path: Path):
    f = tmp_path / "dga.py"
    f.write_text("def dga(seed, nr):\n    return f'{seed}-{nr}.com'\n", encoding="utf-8")
    ins = _inspection_for(f)
    adapter = PythonFunctionAdapter(ins, seed_strategy="sequential", date_strategy="daily_forward", **DATE_ARGS)

    result = adapter.generate(25)
    assert result.generated == 25
    assert len(result.domains) == 25
    assert len(set(result.domains)) == 25
    assert result.attempts >= 25


def test_batch_generator_preserves_batch_semantics(tmp_path: Path):
    f = tmp_path / "dga.py"
    f.write_text(
        "def dga(seed, nr):\n"
        "    return [f'b{seed}-{nr}-{i}.net' for i in range(10)]\n",
        encoding="utf-8",
    )
    ins = _inspection_for(f, code="batchalgo")
    adapter = PythonFunctionAdapter(ins, seed_strategy="sequential", date_strategy="daily_forward", **DATE_ARGS)

    result = adapter.generate(30)
    assert result.generated == 30
    assert len(result.domains) == 30
    assert result.attempts <= 6


def test_adaptive_degradation_after_empty_batches():
    plan = AlgorithmPlan(
        algorithm_code="x",
        path="/tmp",
        category=None,
        strategy="python_function",
        entrypoint="/tmp/dga.py",
        callable_name="dga",
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=100,
        planned_quota=100,
        effective_quota=100,
    )
    cfg = AppConfig(algorithms_root=Path("."), output_dir=Path("."), discard_after_consecutive_empty=2)

    _evaluate_algorithm_health(cfg, plan, inserted=0, generated=0, timed_out=False)
    assert plan.status in {"usable", "degraded", "partial"}
    _evaluate_algorithm_health(cfg, plan, inserted=0, generated=0, timed_out=False)
    assert plan.status == "discarded"


def test_quota_redistribution_toward_productive_algorithms():
    a = AlgorithmPlan(
        algorithm_code="a",
        path="/a",
        category="c1",
        strategy="x",
        entrypoint=None,
        callable_name=None,
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=100,
        planned_quota=100,
        effective_quota=100,
        unique_valid_count=5,
        status="discarded",
    )
    b = AlgorithmPlan(
        algorithm_code="b",
        path="/b",
        category="c1",
        strategy="x",
        entrypoint=None,
        callable_name=None,
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=100,
        planned_quota=100,
        effective_quota=100,
        maximum_effective_quota=200,
        unique_valid_count=60,
        capacity_score=2.0,
        health_score=1.0,
        status="usable",
    )
    moved = _redistribute_from_algorithm({"a": a, "b": b}, "a", "test")
    assert moved > 0
    assert b.effective_quota > 100


def test_timeout_handling_for_cli_adapter(tmp_path: Path):
    script = tmp_path / "slow.py"
    script.write_text(
        "import time\n"
        "time.sleep(2)\n"
        "print('never.com')\n",
        encoding="utf-8",
    )
    ins = AlgorithmInspection(
        algorithm_code="slow",
        path=str(tmp_path),
        strategy="cli_subprocess",
        entrypoint=str(script),
        callable_name=None,
    )
    adapter = CliSubprocessAdapter(
        ins,
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=1,
        batch_timeout_seconds=3,
        max_cli_invocations_per_batch=10,
        **DATE_ARGS,
    )
    result = adapter.generate(5)
    assert result.timed_out is True
    assert result.timeout_events >= 1


def test_cli_batch_timeout_handling(tmp_path: Path):
    script = tmp_path / "slow_batch.py"
    script.write_text(
        "import time\n"
        "time.sleep(0.25)\n"
        "print('x1.com')\n",
        encoding="utf-8",
    )
    ins = AlgorithmInspection(
        algorithm_code="slowbatch",
        path=str(tmp_path),
        strategy="cli_subprocess",
        entrypoint=str(script),
    )
    adapter = CliSubprocessAdapter(
        ins,
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=2,
        batch_timeout_seconds=1,
        max_cli_invocations_per_batch=200,
        **DATE_ARGS,
    )
    result = adapter.generate(2000)
    assert result.batch_aborted is True
    assert result.abort_reason == "batch_timeout"
    assert result.subprocess_calls < 200


def test_cli_stdout_stderr_capture(tmp_path: Path):
    script = tmp_path / "mixed.py"
    script.write_text(
        "import sys\n"
        "print('abc.com')\n"
        "print('noise', file=sys.stderr)\n"
        "print('def.net', file=sys.stderr)\n",
        encoding="utf-8",
    )
    ins = AlgorithmInspection(
        algorithm_code="mixed",
        path=str(tmp_path),
        strategy="cli_subprocess",
        entrypoint=str(script),
    )
    adapter = CliSubprocessAdapter(
        ins,
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=2,
        batch_timeout_seconds=5,
        max_cli_invocations_per_batch=5,
        **DATE_ARGS,
    )
    result = adapter.generate(5)
    assert "abc.com" in result.domains
    assert "def.net" not in result.domains
    assert result.stdout_summary is not None
    assert result.stderr_summary is not None


def test_cli_scalar_mode_capped_invocations(tmp_path: Path):
    script = tmp_path / "scalar.py"
    script.write_text("print('same.com')\n", encoding="utf-8")
    ins = AlgorithmInspection(
        algorithm_code="scalarcli",
        path=str(tmp_path),
        strategy="cli_subprocess",
        entrypoint=str(script),
    )
    adapter = CliSubprocessAdapter(
        ins,
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=2,
        batch_timeout_seconds=20,
        max_cli_invocations_per_batch=5,
        **DATE_ARGS,
    )
    result = adapter.generate(5000)
    assert result.subprocess_calls <= 5
    assert result.generated <= 5


def test_parameter_schedule_reproducibility(tmp_path: Path):
    ins = AlgorithmInspection(algorithm_code="sched", path=str(tmp_path), required_params=["seed", "date"])
    ex1 = ParameterExplorer(ins, seed_strategy="hashed_round_robin", date_strategy="daily_window", **DATE_ARGS)
    ex2 = ParameterExplorer(ins, seed_strategy="hashed_round_robin", date_strategy="daily_window", **DATE_ARGS)

    p1, _ = ex1.build_values(["seed", "date", "nr"], 10)
    p2, _ = ex2.build_values(["seed", "date", "nr"], 10)
    assert p1 == p2


def test_manual_override_and_status_metrics_persisted(tmp_path: Path):
    alg_root = tmp_path / "alg"
    out = tmp_path / "out"
    a = alg_root / "a1"
    a.mkdir(parents=True)
    (a / "dga.py").write_text("def dga(seed, nr):\n    return f'{seed}-{nr}.com'\n", encoding="utf-8")

    config_file = tmp_path / "cfg.json"
    config_file.write_text(
        json.dumps(
            {
                "algorithm_overrides": {
                    "a1": {
                        "parameter_strategy": {"seed_strategy": "sequential"},
                        "priority_weight": 2.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig(
        algorithms_root=alg_root,
        output_dir=out,
        target_count=20,
        batch_size=10,
        config_file=config_file,
        heartbeat_seconds=0,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    state = read_json(out / "udcdga_generation_state.json")
    plan = read_json(out / "udcdga_dga_generation_plan.json")
    stats = read_json(out / "udcdga_dga_domains_stats.json")

    assert "a1" in state["plans"]
    assert "status_history" in state["plans"]["a1"]
    algo_plan = next(x for x in plan["algorithms"] if x["algorithm_code"] == "a1")
    assert "supported_parameter_axes" in algo_plan
    assert "profiling" in algo_plan
    assert "distribution_by_algorithm" in stats


def test_profiling_stage_affects_initial_capacity(tmp_path: Path):
    alg_root = tmp_path / "alg"
    out = tmp_path / "out"

    high = alg_root / "high"
    low = alg_root / "low"
    high.mkdir(parents=True)
    low.mkdir(parents=True)

    (high / "dga.py").write_text(
        "def dga(seed, nr):\n    return [f'h{seed}-{nr}-{i}.com' for i in range(5)]\n",
        encoding="utf-8",
    )
    (low / "dga.py").write_text(
        "def dga(seed, nr):\n    return 'same.com'\n",
        encoding="utf-8",
    )

    cfg = AppConfig(algorithms_root=alg_root, output_dir=out, target_count=20, batch_size=20)
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    plan = read_json(out / "udcdga_dga_generation_plan.json")
    rec = {x["algorithm_code"]: x for x in plan["algorithms"]}
    assert rec["high"]["capacity_score"] >= rec["low"]["capacity_score"]


def test_heartbeat_logging_exists(tmp_path: Path, caplog):
    alg_root = tmp_path / "alg"
    out = tmp_path / "out"
    a = alg_root / "a1"
    a.mkdir(parents=True)
    (a / "dga.py").write_text("def dga(seed, nr):\n    return f'{seed}-{nr}.net'\n", encoding="utf-8")

    cfg = AppConfig(
        algorithms_root=alg_root,
        output_dir=out,
        target_count=30,
        batch_size=10,
        heartbeat_seconds=0,
    )

    caplog.set_level("INFO")
    run_pipeline(cfg)
    assert any("Heartbeat" in r.message for r in caplog.records)


def test_timeout_degrades_then_discards():
    plan = AlgorithmPlan(
        algorithm_code="x",
        path="/tmp",
        category=None,
        strategy="cli_subprocess",
        entrypoint="/tmp/dga.py",
        callable_name="dga",
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=100,
        planned_quota=100,
        effective_quota=100,
    )
    cfg = AppConfig(algorithms_root=Path("."), output_dir=Path("."))
    _evaluate_algorithm_health(cfg, plan, inserted=0, generated=0, timed_out=True, timeout_events=1)
    _evaluate_algorithm_health(cfg, plan, inserted=0, generated=0, timed_out=True, timeout_events=1)
    assert plan.status in {"degraded", "partial", "discarded"}
    _evaluate_algorithm_health(cfg, plan, inserted=0, generated=0, timed_out=True, timeout_events=2)
    assert plan.status == "discarded"


def test_generator_continues_after_cli_failure(tmp_path: Path):
    alg_root = tmp_path / "alg"
    out = tmp_path / "out"
    bad = alg_root / "badcli"
    good = alg_root / "goodpy"
    bad.mkdir(parents=True)
    good.mkdir(parents=True)

    (bad / "dga.py").write_text(
        "import time\n"
        "time.sleep(2)\n"
        "print('never.com')\n",
        encoding="utf-8",
    )
    (good / "dga.py").write_text(
        "def dga(seed, nr):\n"
        "    return f'{seed}-{nr}.org'\n",
        encoding="utf-8",
    )

    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(
        json.dumps(
            {
                "algorithm_overrides": {
                    "badcli": {"adapter_type": "cli", "force_scalar_mode": True, "max_cli_invocations_per_batch": 2}
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig(
        algorithms_root=alg_root,
        output_dir=out,
        target_count=20,
        batch_size=10,
        algorithm_timeout_seconds=1,
        algorithm_batch_timeout_seconds=2,
        config_file=cfg_file,
    )
    try:
        payload = run_pipeline(cfg)
    except GenerationError:
        payload = read_json(out / "udcdga_dga_domains_stats.json")
    assert payload["unique_domains"] >= 1


def test_saturated_then_exhausted_transition():
    plan = AlgorithmPlan(
        algorithm_code="sat",
        path="/tmp",
        category=None,
        strategy="python_function",
        entrypoint="/tmp/dga.py",
        callable_name="dga",
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=500,
        planned_quota=500,
        effective_quota=500,
    )
    cfg = AppConfig(
        algorithms_root=Path("."),
        output_dir=Path("."),
        saturation_window=3,
        saturation_min_yield=0.05,
        exhausted_after_zero_unique_batches=3,
    )

    _evaluate_algorithm_health(cfg, plan, inserted=0, generated=100, timed_out=False)
    _evaluate_algorithm_health(cfg, plan, inserted=0, generated=100, timed_out=False)
    assert plan.status in {"saturated", "degraded", "partial"}
    _evaluate_algorithm_health(cfg, plan, inserted=0, generated=100, timed_out=False)
    assert plan.status == "exhausted"


def test_batch_shrinking_modes():
    cfg = AppConfig(algorithms_root=Path("."), output_dir=Path("."), batch_size=5000)
    plan = AlgorithmPlan(
        algorithm_code="b",
        path="/b",
        category=None,
        strategy="x",
        entrypoint=None,
        callable_name=None,
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=10000,
        planned_quota=10000,
        effective_quota=10000,
        unique_valid_count=0,
    )
    plan.last_batch_unique_yield_ratio = 0.9
    assert _adaptive_batch_request(cfg, plan, 20000) == 5000
    plan.last_batch_unique_yield_ratio = 0.4
    assert _adaptive_batch_request(cfg, plan, 20000) <= 2500
    plan.last_batch_unique_yield_ratio = 0.1
    assert _adaptive_batch_request(cfg, plan, 20000) <= 500
    plan.last_batch_unique_yield_ratio = 0.0
    assert _adaptive_batch_request(cfg, plan, 20000) <= 50
    plan.unique_valid_count = plan.effective_quota
    assert _adaptive_batch_request(cfg, plan, 20000) == 0


def test_date_clamping_behavior():
    ins = AlgorithmInspection(algorithm_code="d", path="/tmp", required_params=["date"])
    ex = ParameterExplorer(
        ins,
        seed_strategy="sequential",
        date_strategy="daily_forward",
        date_start="2020-01-01",
        date_end="2020-01-05",
        date_max_years_forward=0,
        date_max_years_backward=0,
        date_wrap_policy="clamp",
    )
    vals, _ = ex.build_values(["date"], 1000)
    assert str(vals["date"].date()) == "2020-01-05"


def test_date_wrap_behavior():
    ins = AlgorithmInspection(algorithm_code="d2", path="/tmp", required_params=["date"])
    ex = ParameterExplorer(
        ins,
        seed_strategy="sequential",
        date_strategy="daily_forward",
        date_start="2020-01-01",
        date_end="2020-01-05",
        date_max_years_forward=0,
        date_max_years_backward=0,
        date_wrap_policy="wrap",
    )
    vals, _ = ex.build_values(["date"], 10)
    assert str(vals["date"].date()) in {"2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"}


def test_json_persists_saturation_fields(tmp_path: Path):
    alg_root = tmp_path / "alg"
    out = tmp_path / "out"
    a = alg_root / "a1"
    a.mkdir(parents=True)
    (a / "dga.py").write_text("def dga(seed, nr):\n    return f'{seed}-{nr}.com'\n", encoding="utf-8")

    cfg = AppConfig(algorithms_root=alg_root, output_dir=out, target_count=20, batch_size=10)
    run_pipeline(cfg)
    stats = read_json(out / "udcdga_dga_domains_stats.json")
    algo = stats["distribution_by_algorithm"]["a1"]
    assert "maximum_effective_quota" in algo
    assert "oversupply" in algo
    assert "rolling_yield_summary" in algo
    assert "adaptive_batch_mode" in algo


def test_near_quota_exhaustion_redistribution():
    donor = AlgorithmPlan(
        algorithm_code="donor",
        path="/d",
        category="c",
        strategy="x",
        entrypoint=None,
        callable_name=None,
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=100,
        planned_quota=100,
        effective_quota=100,
        unique_valid_count=99,
        status="exhausted",
    )
    recv = AlgorithmPlan(
        algorithm_code="recv",
        path="/r",
        category="c",
        strategy="x",
        entrypoint=None,
        callable_name=None,
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=100,
        planned_quota=100,
        effective_quota=100,
        maximum_effective_quota=200,
        capacity_score=2.0,
        redistribution_eligibility=True,
        status="usable",
    )
    moved = _redistribute_from_algorithm({"donor": donor, "recv": recv}, "donor", "test")
    assert moved >= 1
    assert recv.effective_quota > 100


def test_cap_on_over_quota_without_redistribution():
    cfg = AppConfig(algorithms_root=Path("."), output_dir=Path("."))
    plan = AlgorithmPlan(
        algorithm_code="cap",
        path="/tmp",
        category=None,
        strategy="x",
        entrypoint=None,
        callable_name=None,
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=100,
        planned_quota=100,
        effective_quota=100,
        unique_valid_count=100,
        redistributed_in=0,
    )
    assert plan.remaining_quota() == 0
    assert _adaptive_batch_request(cfg, plan, 1000) == 0
