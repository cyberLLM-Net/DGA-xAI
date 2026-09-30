from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DomainValidationResult:
    is_valid: bool
    normalized: str | None
    reason: str | None = None


def normalize_domain(value: str) -> str:
    """Strip whitespace, lowercase the value, and remove one trailing dot."""
    out = value.strip().lower()
    if out.endswith("."):
        out = out[:-1]
    return out


def validate_domain(value: str) -> DomainValidationResult:
    """Normalize and validate a domain, converting Unicode labels with IDNA.

    The ASCII result is limited to 253 characters and must contain at least
    two non-empty labels of at most 63 characters. Labels may contain letters,
    digits, and interior hyphens; the final label must contain 2--63 letters.
    """
    if value is None:
        return DomainValidationResult(False, None, "none")
    normalized = normalize_domain(value)
    if not normalized:
        return DomainValidationResult(False, None, "empty")

    try:
        # Normalize IDN to punycode. If conversion fails, keep original and validate later.
        normalized = normalized.encode("idna").decode("ascii")
    except Exception:
        pass

    if len(normalized) > 253:
        return DomainValidationResult(False, None, "too_long")

    if ".." in normalized:
        return DomainValidationResult(False, None, "double_dot")

    labels = normalized.split(".")
    if len(labels) < 2:
        return DomainValidationResult(False, None, "invalid_suffix")
    for label in labels:
        if label == "":
            return DomainValidationResult(False, None, "empty_label")
        if len(label) > 63:
            return DomainValidationResult(False, None, "label_too_long")
        if label.startswith("-") or label.endswith("-"):
            return DomainValidationResult(False, None, "invalid_character")
        for ch in label:
            if not (("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "-"):
                return DomainValidationResult(False, None, "invalid_character")

    suffix = labels[-1]
    if not (2 <= len(suffix) <= 63 and suffix.isalpha()):
        return DomainValidationResult(False, None, "invalid_suffix")

    return DomainValidationResult(True, normalized, None)


def extract_domain_like_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        candidate = line.strip().split()[0] if line.strip() else ""
        res = validate_domain(candidate)
        if res.is_valid and res.normalized:
            out.append(res.normalized)
    return out
