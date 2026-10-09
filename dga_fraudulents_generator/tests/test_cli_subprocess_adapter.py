from __future__ import annotations

import csv
from pathlib import Path

from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import run_pipeline
from dga_fraudulents_dataset.invocation import CliSubprocessAdapter
from dga_fraudulents_dataset.models import AlgorithmInspection


DATE_ARGS = {
    "date_start": "2020-01-01",
    "date_end": "2020-12-31",
    "date_max_years_forward": 1,
    "date_max_years_backward": 1,
    "date_wrap_policy": "clamp",
}


def make_adapter(
    algorithm_dir: Path,
    entrypoint: str | Path,
    *,
    timeout_seconds: int = 2,
    max_invocations: int = 1,
    parameter_strategy: dict | None = None,
) -> CliSubprocessAdapter:
    inspection = AlgorithmInspection(
        algorithm_code="cli_fixture",
        path=str(algorithm_dir),
        strategy="cli_subprocess",
        entrypoint=str(entrypoint),
        parameter_strategy=parameter_strategy or {},
    )
    return CliSubprocessAdapter(
        inspection,
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=timeout_seconds,
        batch_timeout_seconds=3,
        max_cli_invocations_per_batch=max_invocations,
        **DATE_ARGS,
    )


def write_script(algorithm_dir: Path, source: str) -> Path:
    algorithm_dir.mkdir(parents=True)
    script = algorithm_dir / "dga.py"
    script.write_text(source, encoding="utf-8")
    return script


def test_successful_subprocess_parses_valid_stdout(tmp_path: Path):
    algorithm_dir = tmp_path / "algorithms" / "success"
    script = write_script(algorithm_dir, "print('alpha.example')\nprint('beta.test')\n")

    result = make_adapter(algorithm_dir, script).generate(10)

    assert result.domains == ["alpha.example", "beta.test"]
    assert result.errors == []


def test_successful_subprocess_never_parses_noisy_stderr(tmp_path: Path):
    algorithm_dir = tmp_path / "algorithms" / "stderr_noise"
    script = write_script(
        algorithm_dir,
        "import sys\nprint('stdout.example')\nprint('stderr.example', file=sys.stderr)\n",
    )

    result = make_adapter(algorithm_dir, script).generate(10)

    assert result.domains == ["stdout.example"]
    assert "stderr.example" in result.stderr_summary


def test_nonzero_exit_discards_domain_like_stderr(tmp_path: Path):
    algorithm_dir = tmp_path / "algorithms" / "failed_stderr"
    script = write_script(
        algorithm_dir,
        "import sys\nprint('error.example', file=sys.stderr)\nraise SystemExit(7)\n",
    )

    result = make_adapter(algorithm_dir, script).generate(10)

    assert result.domains == []
    assert "subprocess_exit:7" in result.errors
    assert "error.example" in result.stderr_summary


def test_nonzero_exit_discards_domain_like_stdout(tmp_path: Path):
    algorithm_dir = tmp_path / "algorithms" / "failed_stdout"
    script = write_script(algorithm_dir, "print('not-eligible.example')\nraise SystemExit(4)\n")

    result = make_adapter(algorithm_dir, script).generate(10)

    assert result.domains == []
    assert "subprocess_exit:4" in result.errors
    assert "not-eligible.example" in result.stdout_summary


def test_missing_script_is_an_explicit_invocation_failure(tmp_path: Path):
    algorithm_dir = tmp_path / "algorithms" / "missing"
    algorithm_dir.mkdir(parents=True)

    result = make_adapter(algorithm_dir, algorithm_dir / "absent.py").generate(10)

    assert result.domains == []
    assert result.subprocess_calls == 1
    assert any(error.startswith("missing_entrypoint:") for error in result.errors)


def test_root_relative_entrypoint_resolves_while_algorithm_dir_is_cwd(
    tmp_path: Path, monkeypatch
):
    project = tmp_path / "project"
    algorithm_dir = project / "dga_algorithms" / "resolved"
    write_script(
        algorithm_dir,
        "from pathlib import Path\n"
        "assert Path.cwd() == Path(__file__).resolve().parent\n"
        "print('resolved.example')\n",
    )
    monkeypatch.chdir(project)

    result = make_adapter(
        algorithm_dir,
        Path("dga_algorithms") / "resolved" / "dga.py",
    ).generate(10)

    assert result.domains == ["resolved.example"]
    assert result.errors == []


def test_timeout_behavior_remains_unchanged(tmp_path: Path):
    algorithm_dir = tmp_path / "algorithms" / "timeout"
    script = write_script(
        algorithm_dir,
        "import time\ntime.sleep(2)\nprint('too-late.example')\n",
    )

    result = make_adapter(algorithm_dir, script, timeout_seconds=1).generate(10)

    assert result.domains == []
    assert result.timed_out is True
    assert result.timeout_events == 1
    assert "timeout" in result.errors


def sequence_script(count: int) -> str:
    return f"for i in range({count}):\n    print(f'domain{{i:03d}}.example')\n"


def test_deterministic_batch_sequence_continues_without_repeated_prefix(tmp_path: Path):
    algorithm_dir = tmp_path / "algorithms" / "continuation"
    script = write_script(algorithm_dir, sequence_script(12))
    adapter = make_adapter(algorithm_dir, script)

    first = adapter.generate(4)
    second = adapter.generate(4)

    assert first.domains == [f"domain{i:03d}.example" for i in range(4)]
    assert second.domains == [f"domain{i:03d}.example" for i in range(4, 8)]
    assert set(first.domains).isdisjoint(second.domains)


def test_three_batches_equal_one_continuous_sequence(tmp_path: Path):
    algorithm_dir = tmp_path / "algorithms" / "three_batches"
    script = write_script(algorithm_dir, sequence_script(12))
    adapter = make_adapter(algorithm_dir, script)

    combined = sum((adapter.generate(3).domains for _ in range(3)), [])

    assert combined == [f"domain{i:03d}.example" for i in range(9)]


def test_checkpoint_resume_continues_same_logical_sequence(tmp_path: Path):
    algorithm_dir = tmp_path / "algorithms" / "resume"
    script = write_script(algorithm_dir, sequence_script(12))
    interrupted = make_adapter(algorithm_dir, script)
    first = interrupted.generate(5)
    checkpoint = first.last_effective_params

    resumed = make_adapter(
        algorithm_dir,
        script,
        parameter_strategy={
            "resume_cli_invocation_count": checkpoint["cli_invocation_count"],
            "resume_cli_sequence_key": checkpoint["cli_sequence_key"],
            "resume_cli_sequence_offset": checkpoint["cli_sequence_offset"],
        },
    )
    resumed_sequence = first.domains + resumed.generate(4).domains
    continuous = make_adapter(algorithm_dir, script).generate(9).domains

    assert resumed_sequence == continuous


def test_pipeline_checkpoint_resume_matches_continuous_run(tmp_path: Path):
    algorithms_root = tmp_path / "algorithms"
    algorithm_dir = algorithms_root / "resumecli"
    write_script(algorithm_dir, sequence_script(100))
    resumed_output = tmp_path / "resumed"
    common = {
        "algorithms_root": algorithms_root,
        "batch_size": 5,
        "checkpoint_every": 1,
        "heartbeat_seconds": 0,
    }

    run_pipeline(AppConfig(output_dir=resumed_output, target_count=5, **common))
    run_pipeline(AppConfig(output_dir=resumed_output, target_count=9, resume=True, **common))
    continuous_output = tmp_path / "continuous"
    run_pipeline(AppConfig(output_dir=continuous_output, target_count=9, **common))

    def domains(output_dir: Path) -> list[str]:
        with (output_dir / "udcdga_dga_domains.csv").open(newline="") as handle:
            return [row["domain"] for row in csv.DictReader(handle)]

    assert domains(resumed_output) == domains(continuous_output)


def test_finite_batch_generator_reports_sequence_exhaustion(tmp_path: Path):
    algorithm_dir = tmp_path / "algorithms" / "finite"
    script = write_script(algorithm_dir, sequence_script(3))
    adapter = make_adapter(algorithm_dir, script)

    assert adapter.generate(2).domains == ["domain000.example", "domain001.example"]
    assert adapter.generate(2).domains == ["domain002.example"]
    exhausted = adapter.generate(2)

    assert exhausted.domains == []
    assert exhausted.last_effective_params["sequence_exhausted"] is True


def test_repeated_runs_with_same_configuration_are_reproducible(tmp_path: Path):
    algorithm_dir = tmp_path / "algorithms" / "reproducible"
    script = write_script(algorithm_dir, sequence_script(12))

    left = make_adapter(algorithm_dir, script)
    right = make_adapter(algorithm_dir, script)
    left_domains = left.generate(3).domains + left.generate(4).domains
    right_domains = right.generate(3).domains + right.generate(4).domains

    assert left_domains == right_domains


def test_continuation_prevents_false_saturation_in_production_pipeline(tmp_path: Path):
    algorithms_root = tmp_path / "algorithms"
    algorithm_dir = algorithms_root / "batchcli"
    write_script(algorithm_dir, sequence_script(100))
    output_dir = tmp_path / "output"
    cfg = AppConfig(
        algorithms_root=algorithms_root,
        output_dir=output_dir,
        target_count=9,
        batch_size=3,
        checkpoint_every=3,
        heartbeat_seconds=0,
    )

    stats = run_pipeline(cfg)
    plan = stats["distribution_by_algorithm"]["batchcli"]

    assert plan["inserted_unique_total"] == 9
    assert plan["duplicates_total"] == 0
    assert plan["status"] == "usable"
