from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

FIELD_DESCRIPTION_FILE_NAME = "udcdga_ARFF_dataset_schema.json"
SAMPLE_FILE_NAME = "udcdga_ARFF_dataset_samples.arff"
DOMAIN_COLUMN = "DOMAIN"
CLASS_COLUMN = "CLASS"
LABEL_COLUMN = "LABEL"

# Default 54-feature subset used by the dataset build flow.
SELECTED_FEATURE_NAMES: List[str] = [
    "1G_75P",
    "1G_DIST",
    "1G_DST_CA",
    "1G_DST_CH",
    "1G_DST_EM",
    "1G_DST_EU",
    "1G_DST_KL",
    "1G_DST_MA",
    "1G_KEN",
    "1G_KUR",
    "1G_NORM",
    "1G_PEA",
    "1G_PRO",
    "1G_REP",
    "1G_SKE",
    "2G_DIST",
    "2G_DST_EM",
    "2G_DST_EU",
    "2G_DST_KL",
    "2G_KEN",
    "2G_KUR",
    "2G_NORM",
    "2G_PEA",
    "2G_REP",
    "2G_SKE",
    "2G_TKUR",
    "3G_25P",
    "3G_DST_EM",
    "3G_DST_KL",
    "3G_KEN",
    "3G_KUR",
    "3G_NORM",
    "3G_PRO",
    "3G_REP",
    "3G_SKE",
    "L_CONSONANT_RATIO_LETTERS",
    "L_ENTROPY",
    "L_HAS_HYPHEN",
    "L_LC_C",
    "L_LC_D",
    "L_LC_V",
    "L_LEN_LABEL_MAX",
    "L_LEN_LABEL_MIN",
    "L_MAX_CHAR_RUN",
    "L_MAX_CONSONANT_RUN_LETTERS",
    "L_MAX_VOWEL_RUN_LETTERS",
    "L_NUM_DOTS",
    "L_UNIQUE_RATIO",
    "L_VC_ALTERNATIONS",
    "N_CON_2LD",
    "N_CON_OLD",
    "N_LEN_OLD",
    "N_LET_2LD",
    "N_VOW_2LD",
]


def selected_feature_names() -> List[str]:
    return list(SELECTED_FEATURE_NAMES)


def parse_arff_attribute_names(path: Path) -> List[str]:
    attrs: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.upper().startswith("@DATA"):
                break
            if line.upper().startswith("@ATTRIBUTE"):
                parts = line.split()
                if len(parts) >= 2:
                    attrs.append(parts[1])
    return attrs


def resolve_feature_names(reference_schema_path: Path | None) -> List[str]:
    if reference_schema_path is not None and reference_schema_path.exists():
        attrs = parse_arff_attribute_names(reference_schema_path)
        filtered = [
            a
            for a in attrs
            if a.upper() not in {DOMAIN_COLUMN, CLASS_COLUMN, LABEL_COLUMN}
            and a.lower() not in {"domain", "class", "label"}
        ]
        if filtered:
            return filtered
    return selected_feature_names()


def full_dataset_columns(feature_names: List[str]) -> List[str]:
    return [DOMAIN_COLUMN, *feature_names, CLASS_COLUMN, LABEL_COLUMN]


def feature_schema_checksum(feature_names: List[str]) -> str:
    payload = "\n".join(feature_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_field_descriptions(feature_names: List[str]) -> Dict[str, object]:
    fields: List[Dict[str, object]] = [
        {
            "field_name": DOMAIN_COLUMN,
            "logical_type": "domain_name",
            "storage_type": "string",
            "nullable": False,
            "description": "Normalized domain name used as input for feature extraction.",
            "how_it_is_computed": "Read from CSV domain column or first CSV column, lower-cased, IDNA normalized, trailing dot removed.",
            "source_columns_or_dependencies": ["input_csv.domain", "input_csv.col_0"],
            "normalization_or_transformation_rules": [
                "Trim whitespace",
                "Convert to lower-case",
                "Strip trailing dot",
                "Reject rows containing spaces or invalid DNS label characters",
                "Convert IDNs to ASCII punycode",
            ],
            "notes_on_edge_cases": "Rows with empty domain, malformed labels, label length >63, or domain length >253 are rejected.",
        }
    ]

    for feature_name in feature_names:
        fields.append(
            {
                "field_name": feature_name,
                "logical_type": "engineered_numeric_feature",
                "storage_type": "float64",
                "nullable": False,
                "description": f"Engineered DNS feature `{feature_name}` computed from the normalized domain string.",
                "how_it_is_computed": "Produced by compute_domain_features_subset(domain, selected_feature_names).",
                "source_columns_or_dependencies": [DOMAIN_COLUMN, "compute_domain_features_subset"],
                "normalization_or_transformation_rules": [
                    "Missing/invalid numeric values are stored as 0.0 in Parquet export",
                    "ARFF sample uses '?' only for invalid numeric conversions",
                ],
                "notes_on_edge_cases": "If a specific feature key is missing from extraction output, 0.0 is used.",
            }
        )

    fields.append(
        {
            "field_name": CLASS_COLUMN,
            "logical_type": "source_class_descriptor",
            "storage_type": "string",
            "nullable": False,
            "description": "Descriptive class/category value for each domain.",
            "how_it_is_computed": "For benign rows CLASS=TLD extracted from DOMAIN. For DGA rows CLASS=algorithm_code from the DGA CSV.",
            "source_columns_or_dependencies": [DOMAIN_COLUMN, "dga_input.algorithm_code"],
            "normalization_or_transformation_rules": [
                "Trim whitespace",
                "Lower-case output value",
            ],
            "notes_on_edge_cases": "For domains without dot separator, benign CLASS is empty string.",
        }
    )
    fields.append(
        {
            "field_name": LABEL_COLUMN,
            "logical_type": "binary_class_label",
            "storage_type": "int8",
            "nullable": False,
            "description": "Binary target label for supervised learning.",
            "how_it_is_computed": "Set from data source provenance: benign input=0, DGA input=1.",
            "source_columns_or_dependencies": ["benign_input", "dga_input"],
            "normalization_or_transformation_rules": [
                "Label constrained to {0,1}",
            ],
            "notes_on_edge_cases": "Final dataset is enforced to be balanced across LABEL values.",
        }
    )

    return {
        "schema_name": "udc_dataset_schema",
        "schema_version": "1.0.0",
        "generated_by": "dga_features_extractor",
        "field_count": len(fields),
        "fields": fields,
    }


def write_field_descriptions(path: Path, feature_names: List[str]) -> Dict[str, object]:
    payload = build_field_descriptions(feature_names)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    return payload
