from __future__ import annotations

import math
from typing import Iterable, List

from .schema import CLASS_COLUMN, DOMAIN_COLUMN, LABEL_COLUMN


def generate_arff_header(
    relation_name: str,
    feature_names: List[str],
    domain_attribute_name: str = DOMAIN_COLUMN,
    class_attribute_name: str = CLASS_COLUMN,
    label_attribute_name: str = LABEL_COLUMN,
) -> str:
    lines: List[str] = [
        f"@RELATION {relation_name}",
        "",
        f"@ATTRIBUTE {domain_attribute_name} STRING",
    ]
    lines.extend([f"@ATTRIBUTE {name} NUMERIC" for name in feature_names])
    lines.append(f"@ATTRIBUTE {class_attribute_name} STRING")
    lines.append(f"@ATTRIBUTE {label_attribute_name} {{0,1}}")
    lines.append("")
    lines.append("@DATA")
    return "\n".join(lines) + "\n"


def escape_arff_string(value: str) -> str:
    sanitized = value.replace("\\", "\\\\").replace("'", "\\'")
    sanitized = sanitized.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return f"'{sanitized}'"


def format_arff_numeric(value: float | int | None) -> str:
    if value is None:
        return "?"
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return "?"
    if math.isnan(fval) or math.isinf(fval):
        return "?"
    return repr(fval)


def format_arff_row(domain: str, feature_values: Iterable[float], class_value: str, label: int) -> str:
    parts = [escape_arff_string(domain)]
    parts.extend(format_arff_numeric(v) for v in feature_values)
    parts.append(escape_arff_string(class_value))
    parts.append(str(label))
    return ",".join(parts) + "\n"


def validate_domain_line(raw_line: str) -> str | None:
    domain = raw_line.strip()
    if not domain:
        return None
    if "," in domain:
        return None
    if any(ch.isspace() for ch in domain):
        return None
    return domain


def count_data_rows(lines: Iterable[str]) -> int:
    return sum(1 for line in lines if line.strip())
