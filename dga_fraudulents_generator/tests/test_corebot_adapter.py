from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.corebot_adapter import CorebotAdapter
from dga_fraudulents_dataset.generator import GenerationError, run_pipeline
from dga_fraudulents_dataset.models import AlgorithmInspection
from dga_fraudulents_dataset.utils import read_json
from dga_fraudulents_dataset.validation import validate_domain


def _real_corebot_inspection(*, strategy: dict | None = None) -> AlgorithmInspection:
    cpath = Path(__file__).resolve().parents[1] / "dga_algorithms" / "corebot"
    return AlgorithmInspection(
        algorithm_code="corebot",
        path=str(cpath),
        strategy="python_function",
        entrypoint=str(cpath / "dga.py"),
        callable_name="init_rand_and_chars",
        parameter_strategy=strategy or {},
    )


def _write_fake_corebot(root: Path, *, invalid_mode: bool = False) -> Path:
    c = root / "corebot"
    c.mkdir(parents=True, exist_ok=True)
    if not invalid_mode:
        script = (
            "def init_rand_and_chars(year, month, day, nr_b, r):\n"
            "    charset='abcdefghijklmnopqrstuvwxyz0123456789'\n"
            "    return charset, (r + year + month + day + nr_b) & 0xffffffff\n"
            "def generate_domain(charset, r):\n"
            "    out=[]\n"
            "    for i in range(12):\n"
            "        r = (1664525*r + 1013904223) & 0xffffffff\n"
            "        out.append(charset[r % len(charset)])\n"
            "    print(''.join(out)+'.ddns.net')\n"
            "    return r\n"
        )
    else:
        script = (
            "def init_rand_and_chars(year, month, day, nr_b, r):\n"
            "    return 'abc', r\n"
            "def generate_domain(charset, r):\n"
            "    if (r % 2) == 0:\n"
            "        print('bad..ddns.net')\n"
            "    else:\n"
            "        print('bad_label!.ddns.net')\n"
            "    return (r + 1) & 0xffffffff\n"
        )
    (c / "dga.py").write_text(script, encoding="utf-8")
    return c


def _fake_inspection(path: Path, *, strategy: dict | None = None) -> AlgorithmInspection:
    return AlgorithmInspection(
        algorithm_code="corebot",
        path=str(path),
        strategy="python_function",
        entrypoint=str(path / "dga.py"),
        callable_name="init_rand_and_chars",
        parameter_strategy=strategy or {},
    )


def test_corebot_example_style_outputs_pass_validation():
    ok = validate_domain("lkhylm0mhyfuhg.ddns.net")
    assert ok.is_valid is True
    assert ok.normalized == "lkhylm0mhyfuhg.ddns.net"


def test_ddns_multilevel_suffix_is_accepted():
    assert validate_domain("abc.ddns.net").is_valid is True
    assert validate_domain("x.y.ddns.net").is_valid is True


def test_corebot_profiling_and_generation_use_same_adapter_logic():
    adapter = CorebotAdapter(_real_corebot_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    p = adapter.profile()
    r = adapter.generate(80)
    assert p["adapter_type"] == "corebot_dedicated"
    assert r.adapter_type == "corebot_dedicated"
    assert r.last_effective_params["generation_mode"] == "date_nr_b_r_schedule"


def test_corebot_parameter_sweeps_include_nr_b_and_r():
    adapter = CorebotAdapter(
        _real_corebot_inspection(strategy={"nr_b_values": [1, 2, 5], "r_seeds": [1, 2, 3], "profile_slots": 16}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    p = adapter.profile()
    assert p["tested_nr_b_values"] == [1, 2, 5]
    assert p["tested_r_seeds"] == [1, 2, 3]


def test_corebot_invalid_reason_reporting(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_fake_corebot(root, invalid_mode=True)

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
    c = stats["distribution_by_algorithm"]["corebot"]
    reasons = c["invalid_reason_counts"]
    assert reasons.get("double_dot", 0) > 0
    assert reasons.get("invalid_character", 0) > 0


def test_corebot_corrected_integration_produces_valid_output(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_fake_corebot(root, invalid_mode=False)

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=150,
        batch_size=80,
        checkpoint_every=20,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    c = stats["distribution_by_algorithm"]["corebot"]
    assert c["valid_total"] > 0
    assert c["inserted_unique_total"] > 0
