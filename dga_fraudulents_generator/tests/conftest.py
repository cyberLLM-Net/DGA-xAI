from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


EXTERNAL_ADAPTER_MODULES = {
    "test_bazarbackdoor_adapter": "bazarbackdoor",
    "test_charbot_adapter": "charbot",
    "test_chinad_adapter": "chinad",
    "test_corebot_adapter": "corebot",
    "test_darkcracks_adapter": "darkcracks",
    "test_dmsniff_adapter": "dmsniff",
    "test_fobber_adapter": "fobber",
    "test_fosniw_adapter": "fosniw",
    "test_gozi_adapter": "gozi",
    "test_locky_adapter": "locky",
    "test_monerodownloader_adapter": "monerodownloader",
    "test_mydoom_adapter": "mydoom",
    "test_newgoz_adapter": "newgoz",
    "test_ngioweb_adapter": "ngioweb",
    "test_nymaim_adapter": "nymaim",
    "test_qsnatch_adapter": "qsnatch",
    "test_zloader_adapter": "zloader",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    algorithms_root = ROOT / "dga_algorithms"
    for item in items:
        module_name = item.module.__name__.rsplit(".", 1)[-1]
        algorithm_code = EXTERNAL_ADAPTER_MODULES.get(module_name)
        if algorithm_code is None:
            continue
        item.add_marker(pytest.mark.integration)
        implementation_dir = algorithms_root / algorithm_code
        if not implementation_dir.exists() or not any(implementation_dir.glob("*.py")):
            item.add_marker(
                pytest.mark.skip(
                    reason=f"external DGA implementation unavailable: {algorithm_code}"
                )
            )
