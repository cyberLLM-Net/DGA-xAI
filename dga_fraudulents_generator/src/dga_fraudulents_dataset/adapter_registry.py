from __future__ import annotations

from .banjori_adapter import BanjoriAdapter
from .bazarbackdoor_adapter import BazarBackdoorAdapter
from .charbot_adapter import CharbotAdapter
from .chinad_adapter import ChinadAdapter
from .corebot_adapter import CorebotAdapter
from .darkcracks_adapter import DarkcracksAdapter
from .dmsniff_adapter import DmsniffAdapter
from .fobber_adapter import FobberAdapter
from .fosniw_adapter import FosniwAdapter
from .gozi_adapter import GoziAdapter
from .invocation import build_adapter
from .locky_adapter import LockyAdapter
from .m0yv_adapter import M0yvAdapter
from .monerodownloader_adapter import MoneroDownloaderAdapter
from .mydoom_adapter import MydoomAdapter
from .models import AlgorithmInspection
from .newgoz_adapter import NewgozAdapter
from .ngioweb_adapter import NgiowebAdapter
from .nymaim_adapter import NymaimAdapter
from .orchard_adapter import OrchardAdapter
from .qsnatch_adapter import QsnatchAdapter
from .zloader_adapter import ZloaderAdapter


def get_adapter(
    inspection: AlgorithmInspection,
    seed_strategy: str,
    date_strategy: str,
    timeout_seconds: int,
    batch_timeout_seconds: int,
    max_cli_invocations_per_batch: int,
    date_start: str,
    date_end: str,
    date_max_years_forward: int,
    date_max_years_backward: int,
    date_wrap_policy: str,
):
    if inspection.algorithm_code.lower() == "charbot":
        return CharbotAdapter(inspection, seed_strategy)
    if inspection.algorithm_code.lower() == "chinad":
        return ChinadAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "banjori":
        return BanjoriAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "corebot":
        return CorebotAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "darkcracks":
        return DarkcracksAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "dmsniff":
        return DmsniffAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "gozi":
        return GoziAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "locky":
        return LockyAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "fobber":
        return FobberAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "fosniw":
        return FosniwAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "ngioweb":
        return NgiowebAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "bazarbackdoor":
        return BazarBackdoorAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "m0yv":
        return M0yvAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "monerodownloader":
        return MoneroDownloaderAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "mydoom":
        return MydoomAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "newgoz":
        return NewgozAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "nymaim":
        return NymaimAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "orchard":
        return OrchardAdapter(inspection, seed_strategy)
    if inspection.algorithm_code.lower() == "qsnatch":
        return QsnatchAdapter(inspection, seed_strategy, date_strategy)
    if inspection.algorithm_code.lower() == "zloader":
        return ZloaderAdapter(inspection, seed_strategy, date_strategy)

    return build_adapter(
        inspection,
        seed_strategy,
        date_strategy,
        timeout_seconds,
        batch_timeout_seconds,
        max_cli_invocations_per_batch,
        date_start,
        date_end,
        date_max_years_forward,
        date_max_years_backward,
        date_wrap_policy,
    )
