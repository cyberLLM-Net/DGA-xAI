from __future__ import annotations

from pathlib import Path

import pytest

from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import GenerationError, run_pipeline
from dga_fraudulents_dataset.gozi_adapter import GoziAdapter
from dga_fraudulents_dataset.models import AlgorithmInspection
from dga_fraudulents_dataset.utils import read_json
from dga_fraudulents_dataset.validation import validate_domain


def _real_gozi_inspection(*, strategy: dict | None = None) -> AlgorithmInspection:
    gpath = Path(__file__).resolve().parents[1] / "dga_algorithms" / "gozi"
    return AlgorithmInspection(
        algorithm_code="gozi",
        path=str(gpath),
        strategy="python_function",
        entrypoint=str(gpath / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
    )


def _write_fake_gozi(root: Path, *, missing_luther: bool = False) -> Path:
    g = root / "gozi"
    g.mkdir(parents=True, exist_ok=True)
    (g / "dga.py").write_text(
        "from datetime import datetime\n"
        "seeds = {\n"
        "  'luther': {'div': 4, 'tld': '.com', 'nr': 12},\n"
        "  'gpl': {'div': 5, 'tld': '.ru', 'nr': 10},\n"
        "}\n"
        "def dga(date, wordlist):\n"
        "  # intentionally left simple; adapter should not rely on this path\n"
        "  return []\n",
        encoding="utf-8",
    )
    if not missing_luther:
        (g / "luther").write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    (g / "gpl").write_text("odin\nthor\nfreya\n", encoding="utf-8")
    return g


def _fake_inspection(path: Path, *, strategy: dict | None = None) -> AlgorithmInspection:
    return AlgorithmInspection(
        algorithm_code="gozi",
        path=str(path),
        strategy="python_function",
        entrypoint=str(path / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
    )


def test_gozi_resources_resolve_relative_to_algorithm_directory():
    adapter = GoziAdapter(
        _real_gozi_inspection(strategy={"gozi_wordlists": ["luther", "gpl"], "profile_slots": 16}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    p = adapter.profile()
    deps = p["resource_dependencies"]
    assert any(d["resource_name"] == "luther" and d["resolved"] for d in deps)
    assert any(d["resource_name"] == "gpl" and d["resolved"] for d in deps)
    for d in deps:
        assert str(Path(d["resource_path"]).name) == d["resource_name"]


def test_gozi_passes_wordlist_as_str_not_list():
    adapter = GoziAdapter(
        _real_gozi_inspection(strategy={"gozi_wordlists": ["luther"], "profile_slots": 8}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r = adapter.generate(24)
    schedule = r.last_effective_params["schedule_used"]
    assert schedule
    assert all(item["wordlist_type"] == "str" for item in schedule)


def test_gozi_example_style_domains_pass_validation():
    ok = validate_domain("quodpresidentemaxsagit.com")
    assert ok.is_valid is True


def test_gozi_missing_resource_file_has_clear_diagnostics(tmp_path: Path):
    g = _write_fake_gozi(tmp_path / "algos", missing_luther=True)
    with pytest.raises(RuntimeError) as exc:
        GoziAdapter(
            _fake_inspection(g, strategy={"gozi_wordlists": ["luther"]}),
            seed_strategy="sequential",
            date_strategy="daily_forward",
        )
    assert "missing_resource_file" in str(exc.value)


def test_gozi_profiling_and_generation_share_adapter_behavior():
    adapter = GoziAdapter(
        _real_gozi_inspection(strategy={"gozi_wordlists": ["luther"], "profile_slots": 16}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    p = adapter.profile()
    r = adapter.generate(36)
    assert p["adapter_type"] == "gozi_dedicated"
    assert r.adapter_type == "gozi_dedicated"
    assert r.last_effective_params["generation_mode"] == "resource_wordlist_date"


def test_gozi_pipeline_runs_and_emits_valid_domains(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_fake_gozi(root, missing_luther=False)

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
    g = stats["distribution_by_algorithm"]["gozi"]
    assert g["valid_total"] > 0
    assert g["inserted_unique_total"] > 0
    assert "resource_dependencies" in g["profiling"]
