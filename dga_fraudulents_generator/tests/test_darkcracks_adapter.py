from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.darkcracks_adapter import DarkcracksAdapter
from dga_fraudulents_dataset.generator import GenerationError, run_pipeline
from dga_fraudulents_dataset.models import AlgorithmInspection
from dga_fraudulents_dataset.utils import read_json
from dga_fraudulents_dataset.validation import validate_domain


def _real_dark_inspection(*, strategy: dict | None = None) -> AlgorithmInspection:
    dpath = Path(__file__).resolve().parents[1] / "dga_algorithms" / "darkcracks"
    return AlgorithmInspection(
        algorithm_code="darkcracks",
        path=str(dpath),
        strategy="python_function",
        entrypoint=str(dpath / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
    )


def _write_fake_dark(root: Path, *, repetitive: bool = False) -> Path:
    d = root / "darkcracks"
    d.mkdir(parents=True, exist_ok=True)
    if not repetitive:
        script = (
            "import base64\n"
            "def dga(seed, date):\n"
            "    # fails if seed is not string (simulate real behavior)\n"
            "    s = seed.encode('ascii')\n"
            "    token = base64.urlsafe_b64encode((date.strftime('%Y%m%d') + ':' + s.decode('ascii')).encode('ascii')).decode('ascii').rstrip('=')\n"
            "    return token.lower()[:12] + '.com'\n"
        )
    else:
        script = (
            "def dga(seed, date):\n"
            "    _ = seed.encode('ascii')\n"
            "    return 'repeat-fixed.com'\n"
        )
    (d / "dga.py").write_text(script, encoding="utf-8")
    return d


def _fake_inspection(path: Path, *, strategy: dict | None = None) -> AlgorithmInspection:
    return AlgorithmInspection(
        algorithm_code="darkcracks",
        path=str(path),
        strategy="python_function",
        entrypoint=str(path / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
    )


def test_darkcracks_int_seed_inputs_normalized_to_str():
    adapter = DarkcracksAdapter(
        _real_dark_inspection(strategy={"seed_texts": [1, 2, 3], "profile_slots": 10}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r = adapter.generate(20)
    assert r.generated > 0
    for item in r.last_effective_params["schedules_used"]:
        assert isinstance(item["seed"], str)


def test_darkcracks_no_encode_type_errors_after_adapter_fix(tmp_path: Path):
    dpath = _write_fake_dark(tmp_path / "algos", repetitive=False)
    adapter = DarkcracksAdapter(_fake_inspection(dpath), seed_strategy="sequential", date_strategy="daily_forward")
    r = adapter.generate(30)
    assert r.generated > 0
    assert not any("type_mismatch_encode_expected_str" in e for e in r.errors)


def test_darkcracks_example_style_outputs_pass_validation():
    assert validate_domain("sTDFUgOAgjL.com").is_valid is True


def test_darkcracks_profiling_and_generation_share_adapter_logic():
    adapter = DarkcracksAdapter(_real_dark_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    p = adapter.profile()
    r = adapter.generate(25)
    assert p["adapter_type"] == "darkcracks_dedicated"
    assert r.adapter_type == "darkcracks_dedicated"
    assert r.last_effective_params["generation_mode"] == "type_normalized_seed_date"


def test_darkcracks_low_diversity_saturates_or_exhausts(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_fake_dark(root, repetitive=True)
    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=300,
        batch_size=100,
        checkpoint_every=20,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    d = stats["distribution_by_algorithm"]["darkcracks"]
    assert d["status"] in {"saturated", "exhausted", "partial"}
