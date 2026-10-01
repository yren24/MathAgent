from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

from mint_scout.invariants.manifest import stable_hash
from mint_scout.openai_responses import request_structured_output


ONBOARDING_SCHEMA = "mint-agent.dataset-onboarding.v1"
MAPPING_FIELDS = (
    "sample_id",
    "target",
    "split",
    "protein",
    "ligand",
    "molecule",
    "pdb_id",
    "smiles",
    "source_filename",
)
REQUIRED_BY_SYSTEM_TYPE = {
    "protein_ligand": ("sample_id", "target", "protein", "ligand"),
    "small_molecule": ("sample_id", "target", "molecule"),
}
VALID_SPLITS = {"train", "validation", "test", "inference"}
PATH_SUFFIXES = {
    ".cif",
    ".ent",
    ".mae",
    ".mmcif",
    ".mol",
    ".mol2",
    ".pdb",
    ".pdbqt",
    ".sdf",
}


def profile_manifest(path: str | Path, *, max_rows: int = 200) -> dict[str, Any]:
    """Return a bounded, value-redacted profile suitable for LLM column mapping."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        columns, rows, row_count, split_values = _profile_csv(
            manifest_path, max_rows=max_rows
        )
        manifest_format = "csv"
    elif suffix in {".jsonl", ".ndjson"}:
        columns, rows, row_count, split_values = _profile_jsonl(
            manifest_path, max_rows=max_rows
        )
        manifest_format = "jsonl"
    else:
        raise ValueError(
            f"Unsupported dataset manifest format {suffix!r}; use .csv, .jsonl, or .ndjson"
        )
    if not columns:
        raise ValueError("Dataset manifest has no columns")
    if row_count == 0:
        raise ValueError("Dataset manifest is empty")
    return {
        "report_schema": ONBOARDING_SCHEMA,
        "format": manifest_format,
        "row_count": row_count,
        "rows_profiled": len(rows),
        "profile_limit": max_rows,
        "columns": [
            _column_profile(
                name,
                rows,
                full_manifest_split_values=split_values.get(name, set()),
            )
            for name in columns
        ],
        "privacy": {
            "raw_values_included": False,
            "sample_ids_included": False,
            "filesystem_paths_included": False,
        },
    }


def request_manifest_column_mapping(
    profile: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    model: str,
    api_key: str,
    endpoint: str = "https://api.openai.com/v1/responses",
    timeout: float = 60.0,
    opener: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    columns = tuple(
        str(item["name"])
        for item in profile.get("columns", ())
        if isinstance(item, Mapping) and item.get("name")
    )
    if not columns:
        raise ValueError("manifest profile does not contain columns")
    dataset = request.get("dataset", {})
    current_columns = dataset.get("columns", {}) if isinstance(dataset, Mapping) else {}
    input_payload = {
        "system_type": request.get("system_type"),
        "task_type": request.get("task_type"),
        "label_name": dataset.get("label_name") if isinstance(dataset, Mapping) else None,
        "current_columns": current_columns,
        "manifest_profile": profile,
    }
    parsed, provenance, _body = request_structured_output(
        json.dumps(input_payload, sort_keys=True),
        instructions=(
            "Map the profiled manifest columns to the requested molecular-learning schema. "
            "Choose only exact column names exposed by the profile. Never infer a unit "
            "conversion, transform a target, invent a split, or invent a filesystem path. "
            "For protein-ligand regression, sample_id, numeric target, protein path, and "
            "ligand path must be high-confidence to proceed. For small-molecule "
            "regression, sample_id, numeric target, and molecule path must be "
            "high-confidence to proceed. Use null for absent optional split or "
            "identifiers. Add every uncertain selection to requires_confirmation. "
            "The summary and evidence must be concise English."
        ),
        schema=_mapping_schema(columns),
        schema_name="mint_agent_manifest_column_mapping",
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        timeout=timeout,
        opener=opener,
    )
    provenance["input_hash"] = stable_hash(input_payload)
    provenance["used_for_numeric_decisions"] = False
    return parsed, provenance


def apply_column_mapping(
    request: Mapping[str, Any],
    mapping_result: Mapping[str, Any],
    *,
    available_columns: tuple[str, ...],
) -> tuple[dict[str, Any], list[str]]:
    updated = copy.deepcopy(dict(request))
    dataset = updated.setdefault("dataset", {})
    columns = dataset.setdefault("columns", {})
    roles = columns.setdefault("roles", {})
    identifiers = columns.setdefault("identifiers", {})
    mapping = mapping_result.get("mapping", {})
    confidence = mapping_result.get("confidence", {})
    confirmations = {
        str(value) for value in mapping_result.get("requires_confirmation", ())
    }
    errors: list[str] = []
    if not isinstance(mapping, Mapping) or not isinstance(confidence, Mapping):
        return updated, ["Column mapper returned an invalid mapping object."]
    available = set(available_columns)
    for field in MAPPING_FIELDS:
        selected = mapping.get(field)
        if selected is not None and selected not in available:
            errors.append(
                f"Column mapper selected unavailable column {selected!r} for {field}."
            )
    required_fields = _required_fields(updated)
    for field in MAPPING_FIELDS:
        if _configured_column(updated, field) is not None:
            continue
        selected = mapping.get(field)
        if selected is None and field in required_fields:
            errors.append(f"No manifest column was identified for required field {field}.")
        elif selected is not None and (
            confidence.get(field) != "high" or field in confirmations
        ):
            errors.append(
                f"Manifest column mapping for {field} requires user confirmation."
            )
    if confirmations:
        errors.extend(
            f"Manifest column mapping for {field} requires user confirmation."
            for field in sorted(confirmations)
            if field not in required_fields
        )
    if errors:
        return updated, list(dict.fromkeys(errors))

    _fill_missing(columns, "sample_id", mapping.get("sample_id"))
    _fill_missing(columns, "target", mapping.get("target"))
    _fill_missing(columns, "split", mapping.get("split"))
    _fill_missing(roles, "protein", mapping.get("protein"))
    _fill_missing(roles, "ligand", mapping.get("ligand"))
    _fill_missing(roles, "molecule", mapping.get("molecule"))
    _fill_missing(identifiers, "pdb_id", mapping.get("pdb_id"))
    _fill_missing(identifiers, "smiles", mapping.get("smiles"))
    _fill_missing(identifiers, "source_filename", mapping.get("source_filename"))

    core = [
        _configured_column(updated, field)
        for field in required_fields
    ]
    if len(core) != len(set(core)):
        errors.append("Required manifest roles must map to distinct columns.")
    return updated, errors


def configured_column_errors(
    request: Mapping[str, Any], *, available_columns: tuple[str, ...]
) -> list[str]:
    available = set(available_columns)
    errors = []
    for field in MAPPING_FIELDS:
        selected = _configured_column(request, field)
        if selected is not None and selected not in available:
            errors.append(
                f"Configured manifest column {selected!r} for {field} is not present."
            )
    return errors


def missing_required_columns(request: Mapping[str, Any]) -> list[str]:
    return [
        field
        for field in _required_fields(request)
        if _configured_column(request, field) is None
    ]


def available_column_names(profile: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(item["name"])
        for item in profile.get("columns", ())
        if isinstance(item, Mapping) and item.get("name")
    )


def _profile_csv(
    path: Path, *, max_rows: int
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...], int, dict[str, set[str]]]:
    profiled: list[dict[str, Any]] = []
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV manifest has no header")
        columns = tuple(str(value) for value in reader.fieldnames)
        if len(columns) != len(set(columns)):
            raise ValueError("CSV manifest contains duplicate column names")
        split_values = {name: set() for name in columns}
        for row in reader:
            row_count += 1
            _record_split_values(row, split_values)
            if len(profiled) < max_rows:
                profiled.append(dict(row))
    return columns, tuple(profiled), row_count, split_values


def _profile_jsonl(
    path: Path, *, max_rows: int
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...], int, dict[str, set[str]]]:
    profiled: list[dict[str, Any]] = []
    columns: list[str] = []
    split_values: dict[str, set[str]] = {}
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on manifest line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"Manifest line {line_number} must be a JSON object")
            row_count += 1
            normalized = {str(key): value for key, value in row.items()}
            for key in normalized:
                if key not in columns:
                    columns.append(key)
                    split_values[key] = set()
            _record_split_values(normalized, split_values)
            if len(profiled) < max_rows:
                profiled.append(normalized)
    return tuple(columns), tuple(profiled), row_count, split_values


def _column_profile(
    name: str,
    rows: tuple[dict[str, Any], ...],
    *,
    full_manifest_split_values: set[str],
) -> dict[str, Any]:
    non_empty = 0
    numeric = 0
    path_like = 0
    suffixes: Counter[str] = Counter()
    value_hashes: set[str] = set()
    for row in rows:
        normalized = _normalized_value(row.get(name))
        if normalized is None:
            continue
        non_empty += 1
        value_hashes.add(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
        if _is_finite_number(normalized):
            numeric += 1
        suffix = Path(normalized).suffix.lower()
        if suffix in PATH_SUFFIXES or "/" in normalized or "\\" in normalized:
            path_like += 1
        if suffix in PATH_SUFFIXES:
            suffixes[suffix] += 1
    denominator = non_empty or 1
    return {
        "name": name,
        "profiled_count": len(rows),
        "non_empty_count": non_empty,
        "missing_count": len(rows) - non_empty,
        "numeric_fraction": round(numeric / denominator, 6),
        "path_like_fraction": round(path_like / denominator, 6),
        "unique_fraction": round(len(value_hashes) / denominator, 6),
        "file_suffixes": sorted(suffixes),
        "recognized_split_values": sorted(full_manifest_split_values),
        "recognized_split_values_scope": "full_manifest",
    }


def _record_split_values(
    row: Mapping[str, object], split_values: Mapping[str, set[str]]
) -> None:
    for name, value in row.items():
        normalized = _normalized_value(value)
        if normalized is None:
            continue
        lowered = normalized.lower()
        if lowered in VALID_SPLITS and name in split_values:
            split_values[name].add(lowered)


def _mapping_schema(columns: tuple[str, ...]) -> dict[str, Any]:
    selection = {"type": ["string", "null"], "enum": [None, *columns]}
    confidence = {"type": "string", "enum": ["high", "medium", "low"]}
    mapping_properties = {field: selection for field in MAPPING_FIELDS}
    confidence_properties = {field: confidence for field in MAPPING_FIELDS}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "mapping",
            "confidence",
            "requires_confirmation",
            "evidence",
            "summary",
        ],
        "properties": {
            "mapping": {
                "type": "object",
                "additionalProperties": False,
                "required": list(MAPPING_FIELDS),
                "properties": mapping_properties,
            },
            "confidence": {
                "type": "object",
                "additionalProperties": False,
                "required": list(MAPPING_FIELDS),
                "properties": confidence_properties,
            },
            "requires_confirmation": {
                "type": "array",
                "items": {"type": "string", "enum": list(MAPPING_FIELDS)},
            },
            "evidence": {
                "type": "object",
                "additionalProperties": False,
                "required": list(MAPPING_FIELDS),
                "properties": {
                    field: {"type": "string"} for field in MAPPING_FIELDS
                },
            },
            "summary": {"type": "string"},
        },
    }


def _configured_column(request: Mapping[str, Any], field: str) -> str | None:
    dataset = request.get("dataset")
    if not isinstance(dataset, Mapping):
        return None
    columns = dataset.get("columns")
    if not isinstance(columns, Mapping):
        return None
    if field in {"sample_id", "target", "split"}:
        return _optional_text(columns.get(field))
    if field in {"protein", "ligand", "molecule"}:
        roles = columns.get("roles")
        return _optional_text(roles.get(field)) if isinstance(roles, Mapping) else None
    identifiers = columns.get("identifiers")
    if not isinstance(identifiers, Mapping):
        return None
    return _optional_text(identifiers.get(field))


def _required_fields(request: Mapping[str, Any]) -> tuple[str, ...]:
    system_type = str(request.get("system_type") or "")
    return REQUIRED_BY_SYSTEM_TYPE.get(system_type, REQUIRED_BY_SYSTEM_TYPE["protein_ligand"])


def _fill_missing(owner: dict[str, Any], key: str, value: object) -> None:
    if _optional_text(owner.get(key)) is None and _optional_text(value) is not None:
        owner[key] = str(value)


def _normalized_value(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_text(value: object) -> str | None:
    return _normalized_value(value)


def _is_finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
