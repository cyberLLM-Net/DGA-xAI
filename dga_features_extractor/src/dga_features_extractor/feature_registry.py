from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Sequence


FEATURE_SCHEMA_VERSION = "1.0"
SUPPORTED_FEATURE_DTYPES = frozenset({"float64"})
RESERVED_FEATURE_NAMES = frozenset({"DOMAIN", "CLASS", "LABEL"})
_FEATURE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class FeatureContext:
    """Immutable input supplied to registered feature extractors."""

    normalized_domain: str
    default_feature_values: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    extractor: Callable[[FeatureContext], float]
    dtype: str = "float64"
    description: str = ""
    definition_version: str = "1"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _FEATURE_NAME_RE.fullmatch(self.name):
            raise ValueError(f"Invalid feature name: {self.name!r}")
        if self.name.upper() in RESERVED_FEATURE_NAMES:
            raise ValueError(f"Reserved feature name: {self.name}")
        if not callable(self.extractor):
            raise TypeError(f"Feature extractor for {self.name!r} must be callable")
        if self.dtype not in SUPPORTED_FEATURE_DTYPES:
            raise ValueError(f"Unsupported feature dtype {self.dtype!r}; supported: float64")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(f"Feature {self.name!r} requires a non-empty description")
        if not isinstance(self.definition_version, str) or not self.definition_version.strip():
            raise ValueError(f"Feature {self.name!r} requires a definition_version")

    def metadata(self) -> dict[str, str]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "description": self.description,
            "definition_version": self.definition_version,
        }


@dataclass(frozen=True)
class FeatureRegistry:
    _definitions: tuple[FeatureDefinition, ...] = ()
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for definition in self._definitions:
            if not isinstance(definition, FeatureDefinition):
                raise TypeError("FeatureRegistry entries must be FeatureDefinition instances")
            if definition.name in seen:
                raise ValueError(f"Duplicate feature name: {definition.name}")
            seen.add(definition.name)
        if not isinstance(self.feature_schema_version, str) or not self.feature_schema_version.strip():
            raise ValueError("feature_schema_version must be a non-empty string")

    @property
    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return self._definitions

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self._definitions)

    def with_feature(self, definition: FeatureDefinition) -> "FeatureRegistry":
        return FeatureRegistry((*self._definitions, definition), self.feature_schema_version)

    def without(self, name: str) -> "FeatureRegistry":
        return FeatureRegistry(tuple(item for item in self._definitions if item.name != name), self.feature_schema_version)

    def reordered(self, names: Sequence[str]) -> "FeatureRegistry":
        if len(names) != len(self._definitions) or set(names) != set(self.names):
            raise ValueError("Reordered names must contain every registered feature exactly once")
        by_name = {item.name: item for item in self._definitions}
        return FeatureRegistry(tuple(by_name[name] for name in names), self.feature_schema_version)


# Historical production order. This tuple is the single source of truth for the
# backward-compatible default 54-feature schema.
DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "1G_75P", "1G_DIST", "1G_DST_CA", "1G_DST_CH", "1G_DST_EM", "1G_DST_EU",
    "1G_DST_KL", "1G_DST_MA", "1G_KEN", "1G_KUR", "1G_NORM", "1G_PEA",
    "1G_PRO", "1G_REP", "1G_SKE", "2G_DIST", "2G_DST_EM", "2G_DST_EU",
    "2G_DST_KL", "2G_KEN", "2G_KUR", "2G_NORM", "2G_PEA", "2G_REP",
    "2G_SKE", "2G_TKUR", "3G_25P", "3G_DST_EM", "3G_DST_KL", "3G_KEN",
    "3G_KUR", "3G_NORM", "3G_PRO", "3G_REP", "3G_SKE",
    "L_CONSONANT_RATIO_LETTERS", "L_ENTROPY", "L_HAS_HYPHEN", "L_LC_C",
    "L_LC_D", "L_LC_V", "L_LEN_LABEL_MAX", "L_LEN_LABEL_MIN",
    "L_MAX_CHAR_RUN", "L_MAX_CONSONANT_RUN_LETTERS", "L_MAX_VOWEL_RUN_LETTERS",
    "L_NUM_DOTS", "L_UNIQUE_RATIO", "L_VC_ALTERNATIONS", "N_CON_2LD",
    "N_CON_OLD", "N_LEN_OLD", "N_LET_2LD", "N_VOW_2LD",
)


def _default_extractor(name: str) -> Callable[[FeatureContext], float]:
    def extract(context: FeatureContext) -> float:
        return float(context.default_feature_values[name])

    return extract


def _default_description(name: str) -> str:
    lexical = {
        "L_CONSONANT_RATIO_LETTERS": "Ratio of consonants to alphabetic characters in the domain labels.",
        "L_ENTROPY": "Shannon entropy of characters in the concatenated domain labels.",
        "L_HAS_HYPHEN": "Indicator that the concatenated domain labels contain a hyphen.",
        "L_LC_C": "Longest consecutive consonant run in the concatenated domain labels.",
        "L_LC_D": "Longest consecutive digit run in the concatenated domain labels.",
        "L_LC_V": "Longest consecutive vowel run in the concatenated domain labels.",
        "L_LEN_LABEL_MAX": "Maximum DNS-label length in the normalized domain.",
        "L_LEN_LABEL_MIN": "Minimum DNS-label length in the normalized domain.",
        "L_MAX_CHAR_RUN": "Longest run of one repeated character in the concatenated domain labels.",
        "L_MAX_CONSONANT_RUN_LETTERS": "Longest consonant run after retaining alphabetic characters.",
        "L_MAX_VOWEL_RUN_LETTERS": "Longest vowel run after retaining alphabetic characters.",
        "L_NUM_DOTS": "Number of separators between labels in the normalized domain.",
        "L_UNIQUE_RATIO": "Ratio of distinct characters to length in the concatenated domain labels.",
        "L_VC_ALTERNATIONS": "Number of vowel/consonant transitions among alphabetic characters.",
        "N_CON_2LD": "Consonant ratio in the second-level domain label.",
        "N_CON_OLD": "Consonant ratio in labels preceding the second-level domain.",
        "N_LEN_OLD": "Combined length of labels preceding the second-level domain.",
        "N_LET_2LD": "Alphabetic-character ratio in the second-level domain label.",
        "N_VOW_2LD": "Vowel ratio in the second-level domain label.",
    }
    if name in lexical:
        return lexical[name]
    n = name[0]
    metric = name.split("_", 1)[1]
    metrics = {
        "75P": "75th percentile of observed n-gram probabilities",
        "25P": "25th percentile of observed n-gram probabilities",
        "DIST": "number of distinct observed n-grams",
        "REP": "number of observed n-grams occurring more than once",
        "SKE": "skewness of observed n-gram probabilities",
        "KUR": "kurtosis of observed n-gram probabilities",
        "NORM": "Jarque-Bera statistic of observed n-gram probabilities",
        "PRO": "pronounceability score relative to the reference n-gram distribution",
        "PEA": "Pearson correlation with the reference n-gram distribution",
        "KEN": "Kendall rank correlation with the reference n-gram distribution",
        "TKUR": "kurtosis of the aligned reference n-gram probabilities",
        "DST_KL": "Kullback-Leibler divergence from the reference n-gram distribution",
        "DST_CA": "Canberra distance from the reference n-gram distribution",
        "DST_CH": "Chebyshev distance from the reference n-gram distribution",
        "DST_EU": "Euclidean distance from the reference n-gram distribution",
        "DST_MA": "Manhattan distance from the reference n-gram distribution",
        "DST_EM": "one-dimensional earth mover distance from the reference n-gram distribution",
    }
    return f"{metrics[metric]} for {n}-grams over alphabetic domain characters."


_DEFAULT_REGISTRY = FeatureRegistry(
    tuple(
        FeatureDefinition(
            name=name,
            extractor=_default_extractor(name),
            dtype="float64",
            description=_default_description(name),
            definition_version="1",
        )
        for name in DEFAULT_FEATURE_NAMES
    )
)


def default_feature_registry() -> FeatureRegistry:
    """Return the immutable default registry."""

    return _DEFAULT_REGISTRY


def feature_name_order_checksum(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def canonical_definition_payload(definitions: Sequence[FeatureDefinition]) -> bytes:
    payload = [
        {
            "definition_version": item.definition_version,
            "dtype": item.dtype,
            "name": item.name,
        }
        for item in definitions
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def feature_definition_checksum(definitions: Sequence[FeatureDefinition]) -> str:
    """Fingerprint ordered definition metadata, not callable implementation code."""

    return hashlib.sha256(canonical_definition_payload(definitions)).hexdigest()


def registry_metadata(registry: FeatureRegistry) -> dict[str, object]:
    return {
        "feature_schema_version": registry.feature_schema_version,
        "feature_count": len(registry.definitions),
        "feature_names": list(registry.names),
        "feature_definitions": [item.metadata() for item in registry.definitions],
        "feature_name_order_checksum": feature_name_order_checksum(registry.names),
        "feature_definition_checksum": feature_definition_checksum(registry.definitions),
    }
