from __future__ import annotations

import re

_DOMAIN_RE_LENIENT = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,63}$")
_DOMAIN_RE_BALANCED = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
_DOMAIN_RE_STRICT = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}$")


def normalize_domain(value: str) -> str:
    """Remove a BOM, surrounding whitespace, case, and a final root dot.

    Unicode labels are left unchanged; this function does not apply IDNA or
    Punycode conversion.
    """
    normalized = value.replace("\ufeff", "").strip().lower().rstrip(".")
    return normalized


def is_valid_domain(domain: str, strictness: str = "balanced") -> bool:
    """Validate an ASCII domain using the selected TLD policy.

    ``lenient`` permits a 2--63 character alphanumeric/hyphen TLD,
    ``balanced`` requires 2--63 ASCII letters, and ``strict`` requires 2--24
    ASCII letters. All modes require at least one dot and a total length no
    greater than 253 characters.
    """
    if not domain or " " in domain or ".." in domain:
        return False
    if strictness == "lenient":
        return bool(_DOMAIN_RE_LENIENT.match(domain))
    if strictness == "strict":
        return bool(_DOMAIN_RE_STRICT.match(domain))
    return bool(_DOMAIN_RE_BALANCED.match(domain))
