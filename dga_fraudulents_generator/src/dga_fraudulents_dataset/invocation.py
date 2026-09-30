from __future__ import annotations

import hashlib
import importlib.util
import inspect
import io
import logging
import math
import re
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .utils import stable_int_seed

logger = logging.getLogger(__name__)

DOMAIN_PRINT_RE = re.compile(r"[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+")


def _hash32(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _common_suffix(values: list[str]) -> str:
    if not values:
        return ""
    rev = [v[::-1] for v in values if v]
    if not rev:
        return ""
    min_len = min(len(v) for v in rev)
    out: list[str] = []
    for i in range(min_len):
        c = rev[0][i]
        if all(v[i] == c for v in rev[1:]):
            out.append(c)
        else:
            break
    return "".join(out)[::-1]


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(max(p, 1e-12))
    return h


def _structural_variability_metrics(domains: list[str]) -> dict[str, Any]:
    clean = [str(d).strip().lower() for d in domains if str(d).strip()]
    if not clean:
        return {
            "prefix_entropy": 0.0,
            "suffix_constancy": 0.0,
            "constant_suffix": "",
            "effective_variable_positions": 0,
            "theoretical_capacity_estimate": 0,
            "low_structural_diversity": False,
            "recommended_effective_quota_cap": 0,
        }

    common_suffix = _common_suffix(clean)
    avg_len = sum(len(d) for d in clean) / max(len(clean), 1)
    suffix_constancy = (len(common_suffix) / avg_len) if avg_len > 0 else 0.0

    prefixes = [d[: len(d) - len(common_suffix)] if common_suffix else d for d in clean]
    pooled_prefix = "".join(prefixes)
    prefix_entropy = _shannon_entropy(pooled_prefix)

    max_len = max((len(p) for p in prefixes), default=0)
    position_sets: list[set[str]] = []
    for i in range(max_len):
        chars = {p[i] for p in prefixes if len(p) > i}
        if chars:
            position_sets.append(chars)
    effective_variable_positions = sum(1 for s in position_sets if len(s) > 1)

    empirical_capacity = 1
    for s in position_sets:
        empirical_capacity *= max(1, len(s))
        if empirical_capacity > 10_000_000_000:
            break

    theoretical_capacity = empirical_capacity
    prefix_lengths = [len(p) for p in prefixes]
    fixed_prefix_len = len(set(prefix_lengths)) == 1 and bool(prefix_lengths)
    alpha_only = all(all(("a" <= ch <= "z") for ch in s) for s in position_sets if len(s) > 1)
    if alpha_only and effective_variable_positions > 0 and effective_variable_positions <= 8:
        if fixed_prefix_len and 1 <= prefix_lengths[0] <= 8:
            theoretical_capacity = 26 ** prefix_lengths[0]
        else:
            theoretical_capacity = 26 ** effective_variable_positions

    low_structural_diversity = (
        len(clean) >= 16
        and suffix_constancy >= 0.55
        and effective_variable_positions <= 8
        and theoretical_capacity <= 2_000_000
    )

    recommended_cap = 0
    if low_structural_diversity and theoretical_capacity > 0:
        recommended_cap = min(
            theoretical_capacity,
            max(50_000, min(200_000, int(theoretical_capacity * 0.4))),
        )

    return {
        "prefix_entropy": prefix_entropy,
        "suffix_constancy": suffix_constancy,
        "constant_suffix": common_suffix,
        "effective_variable_positions": effective_variable_positions,
        "theoretical_capacity_estimate": int(theoretical_capacity),
        "low_structural_diversity": low_structural_diversity,
        "recommended_effective_quota_cap": int(recommended_cap),
    }


@dataclass
class InvocationContext:
    seed_base: int
    base_date: datetime
    round_id: int = 0


class ParameterExplorer:
    def __init__(
        self,
        inspection: AlgorithmInspection,
        seed_strategy: str,
        date_strategy: str,
        date_start: str,
        date_end: str,
        date_max_years_forward: int,
        date_max_years_backward: int,
        date_wrap_policy: str,
    ) -> None:
        self.inspection = inspection
        self.seed_strategy = inspection.parameter_strategy.get("seed_strategy", seed_strategy)
        self.date_strategy = inspection.parameter_strategy.get("date_strategy", date_strategy)
        self.seed_base = stable_int_seed(inspection.algorithm_code)
        self.base_date = datetime(2020, 1, 1)
        try:
            self.base_date = datetime.fromisoformat(inspection.parameter_strategy.get("base_date", "2020-01-01"))
        except Exception:
            self.base_date = datetime(2020, 1, 1)
        self.seed_param_name = inspection.seed_parameter_name
        self.date_param_name = inspection.date_parameter_name
        self.counter_param_name = inspection.counter_parameter_name
        self.wrap_policy = inspection.parameter_strategy.get("date_wrap_policy", inspection.date_wrap_policy or date_wrap_policy)

        self.date_start = datetime.fromisoformat(date_start)
        self.date_end = datetime.fromisoformat(date_end)
        self.date_min = self.date_start
        self.date_max = self.date_end
        if date_max_years_backward > 0:
            self.date_min = max(self.date_min, self.base_date - timedelta(days=365 * date_max_years_backward))
        if date_max_years_forward > 0:
            self.date_max = min(self.date_max, self.base_date + timedelta(days=365 * date_max_years_forward))
        if self.date_min > self.date_max:
            self.date_min, self.date_max = self.date_max, self.date_min
        self.last_requested_date: datetime | None = None
        self.last_effective_date: datetime | None = None

    def _seed_value(self, rid: int) -> int:
        if self.seed_strategy in {"fixed", "incremental"}:
            return self.seed_base + (rid if self.seed_strategy != "fixed" else 0)
        if self.seed_strategy == "sequential":
            return self.seed_base + rid
        if self.seed_strategy == "hashed_round_robin":
            return _hash32(f"{self.inspection.algorithm_code}:{rid}")
        return self.seed_base + rid

    def _normalize_date(self, requested: datetime) -> datetime:
        self.last_requested_date = requested
        out = requested
        if self.date_min <= requested <= self.date_max:
            self.last_effective_date = out
            return out

        if self.wrap_policy == "clamp":
            out = min(max(requested, self.date_min), self.date_max)
        elif self.wrap_policy == "wrap":
            span = max((self.date_max - self.date_min).days + 1, 1)
            delta = (requested - self.date_min).days % span
            out = self.date_min + timedelta(days=delta)
        else:  # reset-cycle
            out = self.date_min
        self.last_effective_date = out
        return out

    def _date_value(self, rid: int) -> datetime:
        if self.date_strategy in {"fixed"}:
            return self._normalize_date(self.base_date)
        if self.date_strategy in {"daily_roll", "daily_forward"}:
            return self._normalize_date(self.base_date + timedelta(days=rid))
        if self.date_strategy in {"daily_window"}:
            return self._normalize_date(self.base_date + timedelta(days=rid % 31))
        if self.date_strategy in {"monthly_window"}:
            month_step = rid % 24
            year = self.base_date.year + (self.base_date.month - 1 + month_step) // 12
            month = (self.base_date.month - 1 + month_step) % 12 + 1
            return self._normalize_date(datetime(year, month, 1))
        if self.date_strategy in {"hourly_roll"}:
            return self._normalize_date(self.base_date + timedelta(hours=rid))
        return self._normalize_date(self.base_date + timedelta(days=rid))

    def build_values(self, param_names: list[str], rid: int) -> tuple[dict[str, Any], list[str]]:
        values: dict[str, Any] = {}
        axes: list[str] = []
        dval = self._date_value(rid)
        sval = self._seed_value(rid)

        for name in param_names:
            n = name.lower()
            if self.seed_param_name and n == self.seed_param_name.lower():
                values[name] = sval
                axes.append("seed")
                continue
            if self.date_param_name and n == self.date_param_name.lower():
                values[name] = dval
                axes.append("date")
                continue
            if self.counter_param_name and n == self.counter_param_name.lower():
                values[name] = rid
                axes.append("counter")
                continue

            if n in {"seed"}:
                values[name] = sval
                axes.append("seed")
            elif n in {"magic"}:
                values[name] = (sval & 0xFFFFFFFF) or 1
                axes.append("seed")
            elif n in {"date", "d", "dt", "when", "time"}:
                values[name] = dval
                axes.append("date")
            elif n in {"year"}:
                values[name] = dval.year
                axes.append("date")
            elif n in {"month"}:
                values[name] = dval.month
                axes.append("date")
            elif n in {"day"}:
                values[name] = dval.day
                axes.append("date")
            elif n in {"nr", "n", "num", "count", "domain_nr", "sequence_nr", "day_index", "tld_index", "back", "number"}:
                values[name] = rid
                axes.append("counter")
            elif n in {"config_nr"}:
                values[name] = (rid % 3) + 1
                axes.append("counter")
            elif n in {"version"}:
                values[name] = ["v1", "v2", "v3", "v4", "v5", "v6", "v7"][rid % 7]
                axes.append("counter")
            elif n in {"prefix"}:
                values[name] = "sn" if rid % 2 == 0 else "al"
                axes.append("counter")
            elif n in {"tlds"}:
                values[name] = ["com", "net", "org", "biz", "info", "ru"]
            elif n in {"wordlist"}:
                values[name] = ["alpha", "beta", "gamma", "delta"]
            elif n in {"domain"}:
                values[name] = f"example{rid % 1000}.com"
            elif n in {"md5"}:
                values[name] = hashlib.md5(str(rid).encode("ascii")).hexdigest()
                axes.append("counter")
            elif n in {"length", "loops"}:
                values[name] = 16 + (rid % 8)
                axes.append("counter")
            elif n in {"config"}:
                values[name] = ["a", "b", "c"][rid % 3]
                axes.append("counter")
            else:
                values[name] = 1

        return values, sorted(set(axes))


class PythonFunctionAdapter(AlgorithmAdapter):
    def __init__(
        self,
        inspection: AlgorithmInspection,
        seed_strategy: str,
        date_strategy: str,
        date_start: str,
        date_end: str,
        date_max_years_forward: int,
        date_max_years_backward: int,
        date_wrap_policy: str,
    ) -> None:
        self.inspection = inspection
        self.module = self._load_module(Path(inspection.entrypoint)) if inspection.entrypoint else None
        self.fn = getattr(self.module, inspection.callable_name, None) if self.module and inspection.callable_name else None
        if self.fn is None:
            raise RuntimeError("callable not found")
        self.sig = inspect.signature(self.fn)
        self.explorer = ParameterExplorer(
            inspection,
            seed_strategy,
            date_strategy,
            date_start,
            date_end,
            date_max_years_forward,
            date_max_years_backward,
            date_wrap_policy,
        )
        self.round_id = 0
        self.force_scalar = inspection.force_scalar_mode
        self.force_batch = inspection.force_batch_mode
        self.mode_hint = "unknown"
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(f"dga_fraudulents_dataset_algo_{path.parent.name}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot create spec for {path}")
        module = importlib.util.module_from_spec(spec)
        prev = list(sys.path)
        try:
            sys.path.insert(0, str(path.parent))
            with redirect_stdout(io.StringIO()):
                spec.loader.exec_module(module)
        finally:
            sys.path[:] = prev
        return module

    def _extract_domains(self, result: Any, captured_stdout: str) -> tuple[list[str], str]:
        out: list[str] = []
        mode = "unknown"
        if isinstance(result, str):
            out = [result]
            mode = "scalar"
        elif isinstance(result, (list, tuple, set)):
            out = [str(x) for x in result]
            mode = "batch"
        elif result is None:
            out = []
        elif isinstance(result, Iterator) or hasattr(result, "__iter__"):
            tmp = []
            for i, x in enumerate(result):
                tmp.append(str(x))
                if i >= 4095:
                    break
            out = tmp
            mode = "batch"
        else:
            mode = "scalar"
            out = [str(result)]

        # Some algorithms only print domains.
        printed = DOMAIN_PRINT_RE.findall(captured_stdout)
        if printed:
            out.extend(printed)
            if len(printed) > 1 and mode == "unknown":
                mode = "batch"
            elif mode == "unknown":
                mode = "scalar"

        return out, mode

    def _invoke_once(self, rid: int) -> tuple[list[str], dict[str, Any], list[str], str]:
        param_names = list(self.sig.parameters.keys())
        values, axes = self.explorer.build_values(param_names, rid)
        if (
            self.explorer.last_requested_date is not None
            and self.explorer.last_effective_date is not None
            and self.explorer.last_requested_date != self.explorer.last_effective_date
        ):
            logger.info(
                "algorithm=%s date_clamped requested=%s effective=%s policy=%s",
                self.inspection.algorithm_code,
                self.explorer.last_requested_date.date().isoformat(),
                self.explorer.last_effective_date.date().isoformat(),
                self.explorer.wrap_policy,
            )
        args = []
        kwargs = {}
        for pname, p in self.sig.parameters.items():
            value = values[pname]
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                args.append(value)
            elif p.kind == inspect.Parameter.KEYWORD_ONLY:
                kwargs[pname] = value

        sink = io.StringIO()
        with redirect_stdout(sink):
            result = self.fn(*args, **kwargs)
        domains, mode = self._extract_domains(result, sink.getvalue())
        return domains, values, axes, mode

    def generate(self, batch_size: int) -> BatchGenerationResult:
        out: list[str] = []
        attempts = 0
        last_params: dict[str, Any] = {}
        axes_seen: set[str] = set()
        max_attempts = max(batch_size * 12, 256)
        errors: list[str] = []

        while len(out) < batch_size and attempts < max_attempts:
            attempts += 1
            rid = self.round_id
            self.round_id += 1
            try:
                domains, params, axes, mode = self._invoke_once(rid)
            except Exception as exc:
                errors.append(str(exc))
                continue
            last_params = params
            axes_seen.update(axes)

            if self.force_scalar:
                selected = domains[:1]
                self.mode_hint = "scalar"
            elif self.force_batch:
                selected = domains
                self.mode_hint = "batch"
            else:
                if mode != "unknown":
                    self.mode_hint = mode
                if self.mode_hint == "scalar":
                    selected = domains[:1]
                else:
                    selected = domains

            out.extend(selected)

        out = out[:batch_size]
        return BatchGenerationResult(
            domains=out,
            attempts=attempts,
            generated=len(out),
            errors=errors[-5:],
            adapter_type="python_function",
            last_effective_params=last_params,
            supported_parameter_axes=sorted(axes_seen),
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache

        r1 = self.generate(min(sample_size, 32))
        r2 = self.generate(min(sample_size, 32))
        s1 = set(r1.domains)
        s2 = set(r2.domains)
        combined = r1.domains + r2.domains
        unique = len(set(combined))
        unique_ratio = unique / max(len(combined), 1)
        structural = _structural_variability_metrics(combined)

        profile = {
            "generates_any": len(combined) > 0,
            "mode": self.mode_hint,
            "seed_sensitive": "seed" in set(r1.supported_parameter_axes) and s1 != s2,
            "date_sensitive": "date" in set(r1.supported_parameter_axes) and s1 != s2,
            "supported_parameter_axes": sorted(set(r1.supported_parameter_axes + r2.supported_parameter_axes)),
            "sample_generated": len(combined),
            "sample_unique": unique,
            "sample_unique_yield": unique_ratio,
            "initial_health_score": min(1.0, max(0.05, unique_ratio)),
            "initial_capacity_score": min(2.0, max(0.05, unique_ratio * 2)),
            "expected_diversity_score": min(1.0, max(0.02, unique_ratio)),
            "recommended_max_effective_quota_multiplier": 1.0 + min(0.5, unique_ratio * 0.6),
            "recommended_saturation_sensitivity": 1.0 - min(0.8, unique_ratio),
            "recommended_date_window": {
                "start": self.explorer.date_min.date().isoformat(),
                "end": self.explorer.date_max.date().isoformat(),
            },
            "apparent_finite_space": unique_ratio < 0.2,
            "prefix_entropy": structural["prefix_entropy"],
            "suffix_constancy": structural["suffix_constancy"],
            "constant_suffix": structural["constant_suffix"],
            "effective_variable_positions": structural["effective_variable_positions"],
            "theoretical_capacity_estimate": structural["theoretical_capacity_estimate"],
            "low_structural_diversity": structural["low_structural_diversity"],
            "recommended_effective_quota_cap": structural["recommended_effective_quota_cap"],
            "estimated_max_unique_capacity": (
                int(structural["theoretical_capacity_estimate"])
                if structural["low_structural_diversity"]
                else 0
            ),
            "adapter_type": "python_function",
        }
        self._profile_cache = profile
        return profile


class CliSubprocessAdapter(AlgorithmAdapter):
    def __init__(
        self,
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
    ) -> None:
        self.inspection = inspection
        self.timeout_seconds = timeout_seconds
        self.batch_timeout_seconds = inspection.algorithm_batch_timeout_seconds or batch_timeout_seconds
        self.max_cli_invocations_per_batch = (
            inspection.max_cli_invocations_per_batch or max_cli_invocations_per_batch
        )
        self.round_id = 0
        self.explorer = ParameterExplorer(
            inspection,
            seed_strategy,
            date_strategy,
            date_start,
            date_end,
            date_max_years_forward,
            date_max_years_backward,
            date_wrap_policy,
        )
        self.force_scalar = inspection.force_scalar_mode
        self.force_batch = inspection.force_batch_mode
        self.cli_mode = "scalar" if self.force_scalar else ("batch" if self.force_batch else "unknown")
        self._profile_cache: dict[str, Any] | None = None

    def _build_cmd(self, rid: int) -> tuple[list[str], dict[str, Any], list[str]]:
        if not self.inspection.entrypoint:
            return [], {}, []
        entry = Path(self.inspection.entrypoint)
        cmd = [sys.executable, str(entry)]
        values, axes = self.explorer.build_values(["seed", "date", "n"], rid)
        if (
            self.explorer.last_requested_date is not None
            and self.explorer.last_effective_date is not None
            and self.explorer.last_requested_date != self.explorer.last_effective_date
        ):
            logger.info(
                "algorithm=%s date_clamped requested=%s effective=%s policy=%s",
                self.inspection.algorithm_code,
                self.explorer.last_requested_date.date().isoformat(),
                self.explorer.last_effective_date.date().isoformat(),
                self.explorer.wrap_policy,
            )

        if self.inspection.requires_date:
            d = values.get("date")
            if isinstance(d, datetime):
                cmd.extend(["--date", d.strftime("%Y-%m-%d")])
        if self.inspection.requires_seed:
            cmd.extend(["--seed", str(values.get("seed", 1))])

        return cmd, values, axes

    def _run_once(
        self, rid: int
    ) -> tuple[list[str], bool, str, str, int, float, dict[str, Any], list[str], list[str]]:
        cmd, params, axes = self._build_cmd(rid)
        if not cmd:
            return [], False, "", "", -1, 0.0, params, axes, []

        call_errors: list[str] = []
        start = time.monotonic()
        logger.debug(
            "CLI call start algorithm=%s mode=%s cmd=%s params=%s",
            self.inspection.algorithm_code,
            self.cli_mode,
            cmd,
            params,
        )
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=Path(self.inspection.path),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate(timeout=2)
                elapsed = time.monotonic() - start
                logger.warning(
                    "CLI timeout algorithm=%s cmd=%s elapsed=%.3fs stdout_len=%s stderr_len=%s",
                    self.inspection.algorithm_code,
                    cmd,
                    elapsed,
                    len(stdout or ""),
                    len(stderr or ""),
                )
                return (
                    [],
                    True,
                    (stdout or "")[:400],
                    (stderr or "")[:400],
                    proc.returncode if proc.returncode is not None else -9,
                    elapsed,
                    params,
                    axes,
                    ["timeout"],
                )

            elapsed = time.monotonic() - start
            lines = (stdout or "").splitlines() + (stderr or "").splitlines()
            domains: list[str] = []
            for line in lines:
                domains.extend(DOMAIN_PRINT_RE.findall(line))
            logger.debug(
                "CLI call end algorithm=%s returncode=%s elapsed=%.3fs parsed_domains=%s stdout_len=%s stderr_len=%s",
                self.inspection.algorithm_code,
                proc.returncode,
                elapsed,
                len(domains),
                len(stdout or ""),
                len(stderr or ""),
            )
            return domains, False, (stdout or "")[:400], (stderr or "")[:400], proc.returncode or 0, elapsed, params, axes, call_errors
        except Exception as exc:
            elapsed = time.monotonic() - start
            call_errors.append(str(exc))
            logger.warning(
                "CLI invocation exception algorithm=%s cmd=%s elapsed=%.3fs error=%s",
                self.inspection.algorithm_code,
                cmd,
                elapsed,
                exc,
            )
            return [], False, "", "", -1, elapsed, params, axes, call_errors

    def generate(self, batch_size: int) -> BatchGenerationResult:
        out: list[str] = []
        attempts = 0
        timed_out = False
        timeout_events = 0
        errors: list[str] = []
        last_params: dict[str, Any] = {}
        axes_seen: set[str] = set()
        stdout_summary = ""
        stderr_summary = ""
        batch_aborted = False
        abort_reason = None

        max_attempts = self.max_cli_invocations_per_batch
        if self.force_batch:
            max_attempts = min(max_attempts, max(batch_size // 3, 1))
        batch_start = time.monotonic()

        while len(out) < batch_size and attempts < max_attempts:
            if (time.monotonic() - batch_start) > self.batch_timeout_seconds:
                batch_aborted = True
                abort_reason = "batch_timeout"
                logger.warning(
                    "CLI batch timeout algorithm=%s elapsed=%.3fs attempts=%s generated=%s",
                    self.inspection.algorithm_code,
                    time.monotonic() - batch_start,
                    attempts,
                    len(out),
                )
                break

            attempts += 1
            rid = self.round_id
            self.round_id += 1
            domains, to, out_s, err_s, returncode, elapsed, params, axes, call_errors = self._run_once(rid)
            last_params = params
            axes_seen.update(axes)
            stdout_summary = out_s
            stderr_summary = err_s
            errors.extend(call_errors)
            if to:
                timed_out = True
                timeout_events += 1
                errors.append("timeout")
                continue

            if self.cli_mode == "unknown":
                if len(domains) <= 1:
                    self.cli_mode = "scalar"
                else:
                    self.cli_mode = "batch"

            if self.force_scalar:
                out.extend(domains[:1])
            else:
                out.extend(domains)
                if self.cli_mode == "scalar":
                    # Scalar CLI: capped attempts per batch avoid runaway loops.
                    continue

        out = out[:batch_size]
        return BatchGenerationResult(
            domains=out,
            attempts=attempts,
            generated=len(out),
            timed_out=timed_out,
            errors=errors[-5:],
            adapter_type="cli_subprocess",
            last_effective_params=last_params,
            supported_parameter_axes=sorted(axes_seen),
            stdout_summary=stdout_summary,
            stderr_summary=stderr_summary,
            batch_aborted=batch_aborted,
            abort_reason=abort_reason,
            timeout_events=timeout_events,
            subprocess_calls=attempts,
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache
        r1 = self.generate(min(sample_size, 32))
        r2 = self.generate(min(sample_size, 32))
        combined = r1.domains + r2.domains
        unique = len(set(combined))
        unique_ratio = unique / max(len(combined), 1)
        structural = _structural_variability_metrics(combined)
        profile = {
            "generates_any": len(combined) > 0,
            "mode": "batch" if len(r1.domains) > 1 else "scalar",
            "seed_sensitive": "seed" in set(r1.supported_parameter_axes),
            "date_sensitive": "date" in set(r1.supported_parameter_axes),
            "supported_parameter_axes": sorted(set(r1.supported_parameter_axes + r2.supported_parameter_axes)),
            "sample_generated": len(combined),
            "sample_unique": unique,
            "sample_unique_yield": unique_ratio,
            "initial_health_score": min(1.0, max(0.05, unique_ratio)),
            "initial_capacity_score": min(2.0, max(0.05, unique_ratio * 2)),
            "expected_diversity_score": min(1.0, max(0.02, unique_ratio)),
            "recommended_max_effective_quota_multiplier": 1.0 + min(0.5, unique_ratio * 0.6),
            "recommended_saturation_sensitivity": 1.0 - min(0.8, unique_ratio),
            "recommended_date_window": {
                "start": self.explorer.date_min.date().isoformat(),
                "end": self.explorer.date_max.date().isoformat(),
            },
            "apparent_finite_space": unique_ratio < 0.2,
            "prefix_entropy": structural["prefix_entropy"],
            "suffix_constancy": structural["suffix_constancy"],
            "constant_suffix": structural["constant_suffix"],
            "effective_variable_positions": structural["effective_variable_positions"],
            "theoretical_capacity_estimate": structural["theoretical_capacity_estimate"],
            "low_structural_diversity": structural["low_structural_diversity"],
            "recommended_effective_quota_cap": structural["recommended_effective_quota_cap"],
            "estimated_max_unique_capacity": (
                int(structural["theoretical_capacity_estimate"])
                if structural["low_structural_diversity"]
                else 0
            ),
            "adapter_type": "cli_subprocess",
        }
        self._profile_cache = profile
        return profile


def build_adapter(
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
) -> AlgorithmAdapter:
    if inspection.discard:
        raise RuntimeError("discard override")

    preferred = inspection.adapter_type
    if preferred in {"cli_subprocess", "cli"}:
        return CliSubprocessAdapter(
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

    if inspection.strategy == "python_function" and inspection.entrypoint and inspection.callable_name:
        try:
            return PythonFunctionAdapter(
                inspection,
                seed_strategy,
                date_strategy,
                date_start,
                date_end,
                date_max_years_forward,
                date_max_years_backward,
                date_wrap_policy,
            )
        except Exception as exc:
            logger.warning("Falling back to CLI adapter for %s: %s", inspection.algorithm_code, exc)
            return CliSubprocessAdapter(
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
    return CliSubprocessAdapter(
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


def smoke_test_adapter(adapter: AlgorithmAdapter, sample_size: int = 10) -> tuple[bool, int]:
    try:
        items = adapter.generate(sample_size)
    except Exception:
        return False, 0
    return len(items.domains) > 0, len(items.domains)
