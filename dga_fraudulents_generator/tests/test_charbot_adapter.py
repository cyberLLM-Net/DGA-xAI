from __future__ import annotations

import socket
from pathlib import Path

from dga_fraudulents_dataset.adapter_registry import get_adapter
from dga_fraudulents_dataset.charbot_adapter import CharbotAdapter
from dga_fraudulents_dataset.models import AlgorithmInspection

DATE_ARGS = {
    "date_start": "2018-01-01",
    "date_end": "2030-12-31",
    "date_max_years_forward": 8,
    "date_max_years_backward": 8,
    "date_wrap_policy": "clamp",
}


def _load_charbot_module():
    from dga_fraudulents_dataset import charbot_generator

    return charbot_generator


def _write_base_domains(tmp_path: Path) -> Path:
    p = tmp_path / "base_domains.csv"
    p.write_text(
        "1,google.com\n"
        "2,example.org\n"
        "3,cloudflare.com\n"
        "4,microsoft.com\n"
        "5,wikipedia.org\n",
        encoding="utf-8",
    )
    return p


def _charbot_inspection(base_domains_path: Path) -> AlgorithmInspection:
    algo_path = Path(__file__).resolve().parents[1] / "dga_algorithms" / "charbot"
    return AlgorithmInspection(
        algorithm_code="charbot",
        path=str(algo_path),
        strategy="python_function",
        default_params={
            "base_domains_path": str(base_domains_path),
            "num_mutated_characters": 2,
            "allowed_tlds": ["com", "net", "org"],
            "min_base_domain_length": 6,
            "random_sampling": True,
        },
    )


def test_charbot_generate_batch_differs_by_seed(tmp_path: Path):
    charbot = _load_charbot_module()
    base = charbot.load_base_domains(_write_base_domains(tmp_path), min_base_domain_length=6)

    first = charbot.generate_batch(base_domains=base, seed=111, counter=0, batch_size=20)
    second = charbot.generate_batch(base_domains=base, seed=222, counter=0, batch_size=20)

    assert len(first) == 20
    assert len(second) == 20
    assert first != second


def test_charbot_generate_batch_differs_by_counter(tmp_path: Path):
    charbot = _load_charbot_module()
    base = charbot.load_base_domains(_write_base_domains(tmp_path), min_base_domain_length=6)

    first = charbot.generate_batch(base_domains=base, seed=111, counter=0, batch_size=20)
    second = charbot.generate_batch(base_domains=base, seed=111, counter=1, batch_size=20)

    assert first != second


def test_charbot_generate_batch_requires_no_network(tmp_path: Path, monkeypatch):
    charbot = _load_charbot_module()
    base = charbot.load_base_domains(_write_base_domains(tmp_path), min_base_domain_length=6)

    def _blocked(*args, **kwargs):
        raise AssertionError("network access is not allowed")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    out = charbot.generate_batch(base_domains=base, seed=7, counter=3, batch_size=10)

    assert len(out) == 10


def test_charbot_adapter_batch_generation_and_counter_progression(tmp_path: Path):
    ins = _charbot_inspection(_write_base_domains(tmp_path))
    adapter = CharbotAdapter(ins, seed_strategy="sequential")

    b1 = adapter.generate(25)
    b2 = adapter.generate(25)

    assert b1.generated == 25
    assert len(b1.domains) == 25
    assert b2.generated == 25
    assert b1.domains != b2.domains


def test_charbot_is_routed_to_dedicated_adapter(tmp_path: Path):
    ins = _charbot_inspection(_write_base_domains(tmp_path))

    adapter = get_adapter(
        ins,
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=5,
        batch_timeout_seconds=5,
        max_cli_invocations_per_batch=10,
        **DATE_ARGS,
    )

    assert isinstance(adapter, CharbotAdapter)
