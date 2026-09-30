from __future__ import annotations

import csv
import hashlib
import random
import re
from pathlib import Path

DEFAULT_TLDS: tuple[str, ...] = (
    "com",
    "at",
    "uk",
    "pl",
    "be",
    "biz",
    "co",
    "jp",
    "cz",
    "de",
    "eu",
    "fr",
    "info",
    "it",
    "ru",
    "lv",
    "me",
    "name",
    "net",
    "nz",
    "org",
    "us",
)
DEFAULT_DNS_CHARSET: tuple[str, ...] = tuple([chr(x) for x in range(0x61, 0x61 + 26)] + [chr(x) for x in range(0x30, 0x30 + 10)] + ["-"])
DOMAIN_RE = re.compile(r"^[a-z0-9-]+\.[a-z0-9-]+$")


def _normalize_domain(raw: str) -> str | None:
    domain = raw.strip().lower().rstrip(".")
    if not domain:
        return None
    if domain.startswith("http://") or domain.startswith("https://"):
        domain = domain.split("://", 1)[1]
    domain = domain.split("/", 1)[0]
    if "," in domain:
        parts = [p.strip() for p in domain.split(",") if p.strip()]
        domain = parts[-1] if parts else ""
    if DOMAIN_RE.match(domain) is None:
        return None
    return domain


def load_base_domains(path: str | Path, min_base_domain_length: int = 6) -> list[str]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Base domains file not found: {src}")

    out: list[str] = []
    seen: set[str] = set()
    with src.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            for cell in row:
                norm = _normalize_domain(cell)
                if not norm:
                    continue
                sld = norm.split(".", 1)[0]
                if len(sld) < min_base_domain_length:
                    continue
                if norm not in seen:
                    seen.add(norm)
                    out.append(norm)
    return out


def _derived_seed(seed: int, counter: int, salt: int = 0) -> int:
    payload = f"{int(seed)}:{int(counter)}:{int(salt)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def _mutate_label(label: str, rng: random.Random, num_mutated_characters: int, dnscharset: tuple[str, ...]) -> str:
    chars = list(label)
    if not chars:
        return label
    mutations = max(1, min(num_mutated_characters, len(chars)))
    for idx in rng.sample(range(len(chars)), mutations):
        chars[idx] = rng.choice(dnscharset)
    return "".join(chars)


def _pick_base_domain(
    base_domains: list[str],
    rng: random.Random,
    batch_index: int,
    counter: int,
    random_sampling: bool,
) -> str:
    if not base_domains:
        raise ValueError("base_domains cannot be empty")
    if random_sampling:
        return base_domains[rng.randrange(len(base_domains))]
    return base_domains[(counter + batch_index) % len(base_domains)]


def generate_batch(
    *,
    base_domains: list[str],
    seed: int,
    counter: int,
    batch_size: int,
    num_mutated_characters: int = 2,
    allowed_tlds: list[str] | tuple[str, ...] = DEFAULT_TLDS,
    min_base_domain_length: int = 6,
    random_sampling: bool = True,
    dnscharset: list[str] | tuple[str, ...] = DEFAULT_DNS_CHARSET,
) -> list[str]:
    if batch_size <= 0:
        return []
    if not allowed_tlds:
        raise ValueError("allowed_tlds cannot be empty")

    filtered = [d for d in base_domains if len(d.split(".", 1)[0]) >= min_base_domain_length]
    if not filtered:
        raise ValueError("No eligible base domains after applying minimum length filter")

    tlds = [str(t).lstrip(".").lower() for t in allowed_tlds if str(t).strip()]
    chars = tuple(str(c) for c in dnscharset if str(c))
    if not chars:
        raise ValueError("dnscharset cannot be empty")

    rng = random.Random(_derived_seed(seed, counter))
    out: list[str] = []
    for i in range(batch_size):
        base = _pick_base_domain(filtered, rng, i, counter, random_sampling)
        label = base.split(".", 1)[0]
        mutated = _mutate_label(label, rng, num_mutated_characters, chars)
        tld = tlds[rng.randrange(len(tlds))]
        out.append(f"{mutated}.{tld}")
    return out
