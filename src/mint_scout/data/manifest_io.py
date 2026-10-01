from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, cast

from mint_scout.data.element_pairs import SystemType
from mint_scout.schemas import (
    DatasetManifest,
    ManifestValidationError,
    SampleRecord,
    SampleSplit,
    TaskCard,
    TaskType,
    route_evaluation_mode,
)


DEFAULT_ROLE_COLUMNS = {
    "protein_ligand": {"protein": "protein_path", "ligand": "ligand_path"},
    "small_molecule": {"molecule": "molecule_path"},
}


def load_dataset_manifest(
    path: str | Path,
    *,
    dataset_id: str,
    system_type: SystemType,
    columns: Mapping[str, Any] | None = None,
    label_name: str | None = None,
    source: str | None = None,
    validate_paths: bool = True,
) -> DatasetManifest:
    manifest_path = Path(path).expanduser().resolve()
    rows = _read_rows(manifest_path)
    column_config = dict(columns or {})
    sample_id_column = str(column_config.get("sample_id", "sample_id"))
    target_column = str(column_config.get("target", "target"))
    split_column = str(column_config.get("split", "split"))
    role_columns = _mapping_of_strings(
        column_config.get("roles", DEFAULT_ROLE_COLUMNS[system_type]),
        name="dataset_manifest.columns.roles",
    )
    identifier_columns = _mapping_of_strings(
        column_config.get("identifiers", {}),
        name="dataset_manifest.columns.identifiers",
    )

    samples = tuple(
        _sample_from_row(
            row,
            row_number=index,
            manifest_path=manifest_path,
            sample_id_column=sample_id_column,
            target_column=target_column,
            split_column=split_column,
            role_columns=role_columns,
            identifier_columns=identifier_columns,
            validate_paths=validate_paths,
        )
        for index, row in enumerate(rows, start=2 if manifest_path.suffix.lower() == ".csv" else 1)
    )
    if not samples:
        raise ManifestValidationError(f"Dataset manifest is empty: {manifest_path}")
    return DatasetManifest(
        dataset_id=dataset_id,
        system_type=system_type,
        samples=samples,
        label_name=label_name,
        source=source or str(manifest_path),
    )


def task_card_from_config(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    user_target: float | None = None,
) -> TaskCard | None:
    raw_manifest = config.get("dataset_manifest")
    if raw_manifest is None:
        return None
    if not isinstance(raw_manifest, Mapping):
        raise ManifestValidationError("dataset_manifest must be a mapping")
    raw_system_type = str(config.get("system_type") or "")
    if raw_system_type not in DEFAULT_ROLE_COLUMNS:
        raise ManifestValidationError(
            f"Unsupported dataset_manifest system_type: {raw_system_type!r}"
        )
    raw_task_type = str(config.get("task_type") or "")
    try:
        task_type = TaskType(raw_task_type)
    except ValueError as exc:
        raise ManifestValidationError(f"Unsupported task_type: {raw_task_type!r}") from exc

    path_value = raw_manifest.get("path")
    if not path_value:
        raise ManifestValidationError("dataset_manifest.path is required")
    manifest_path = Path(str(path_value)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = Path(config_path).expanduser().resolve().parent / manifest_path
    validate_paths = raw_manifest.get("validate_paths", True)
    if not isinstance(validate_paths, bool):
        raise ManifestValidationError("dataset_manifest.validate_paths must be true or false")
    dataset_id = str(config.get("task_id") or "")
    system_type = cast(SystemType, raw_system_type)
    manifest = load_dataset_manifest(
        manifest_path,
        dataset_id=dataset_id,
        system_type=system_type,
        columns=_optional_mapping(raw_manifest.get("columns"), "dataset_manifest.columns"),
        label_name=_optional_string(raw_manifest.get("label_name")),
        source=_optional_string(raw_manifest.get("source")),
        validate_paths=validate_paths,
    )
    retrieval = config.get("label_retrieval", {})
    if retrieval is None:
        retrieval = {}
    if not isinstance(retrieval, Mapping):
        raise ManifestValidationError("label_retrieval must be a mapping")
    return TaskCard(
        task_id=dataset_id,
        dataset=manifest,
        task_type=task_type,
        target_metric=str(config.get("primary_metric", "PCC")).upper(),
        user_target=user_target,
        label_retrieval_allowed=bool(retrieval.get("allowed", False)),
        required_roles=tuple(DEFAULT_ROLE_COLUMNS[raw_system_type]),
    )


def load_configured_protein_ligand_records(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    split: str,
) -> tuple["CasfRecord", ...]:
    from mint_scout.data.casf_index import CasfRecord

    card = task_card_from_config(config, config_path=config_path)
    if card is None:
        raise ManifestValidationError("dataset_manifest is required")
    if card.dataset.system_type != "protein_ligand":
        raise ManifestValidationError(
            "PLBind feature tools require system_type=protein_ligand"
        )
    plan = route_evaluation_mode(card)
    cv_config = config.get("cv", {})
    if not isinstance(cv_config, Mapping):
        raise ManifestValidationError("cv must be a mapping")
    group_identifier = _optional_string(cv_config.get("group_identifier"))
    if split == "train":
        explicit_train = card.dataset.samples_in_split(SampleSplit.TRAIN)
        selected_ids = (
            {sample.sample_id for sample in explicit_train}
            if explicit_train
            else set(plan.modeling_sample_ids)
        )
    elif split == "validation":
        explicit_validation = card.dataset.samples_in_split(SampleSplit.VALIDATION)
        selected_ids = (
            {sample.sample_id for sample in explicit_validation}
            if explicit_validation
            else set(plan.validation_sample_ids)
        )
    elif split == "test":
        explicit_test = (
            card.dataset.samples_in_split(SampleSplit.TEST)
            + card.dataset.samples_in_split(SampleSplit.INFERENCE)
        )
        selected_ids = (
            {sample.sample_id for sample in explicit_test}
            if explicit_test
            else set(plan.evaluation_sample_ids + plan.inference_sample_ids)
        )
    elif split == "all":
        selected_ids = {sample.sample_id for sample in card.dataset.samples}
    else:
        raise ValueError(f"Unsupported record split: {split!r}")
    records = []
    for sample in card.dataset.samples:
        if sample.sample_id not in selected_ids:
            continue
        group_id = (
            sample.identifiers.get(group_identifier)
            if group_identifier is not None
            else None
        )
        if group_identifier is not None and group_id is None:
            raise ManifestValidationError(
                f"Sample {sample.sample_id!r} is missing configured CV group "
                f"identifier {group_identifier!r}"
            )
        records.append(
            CasfRecord(
                pdb_id=sample.sample_id,
                label=sample.target,
                split=sample.split.value if sample.split is not None else None,
                protein_path=sample.role_paths.get("protein"),
                ligand_path=sample.role_paths.get("ligand"),
                group_id=group_id,
            )
        )
    if not records and split not in {"test", "validation"}:
        raise ManifestValidationError(
            f"DatasetManifest produced no samples for {split!r}; evaluation mode is {plan.mode.value}"
        )
    return tuple(records)


def _read_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ManifestValidationError(f"CSV manifest has no header: {path}")
            return tuple(dict(row) for row in reader)
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestValidationError(
                    f"Invalid JSON on manifest line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ManifestValidationError(
                    f"Manifest line {line_number} must contain a JSON object"
                )
            rows.append(dict(row))
        return tuple(rows)
    raise ManifestValidationError(
        f"Unsupported dataset manifest format {suffix!r}; use .csv, .jsonl, or .ndjson"
    )


def _sample_from_row(
    row: Mapping[str, Any],
    *,
    row_number: int,
    manifest_path: Path,
    sample_id_column: str,
    target_column: str,
    split_column: str,
    role_columns: Mapping[str, str],
    identifier_columns: Mapping[str, str],
    validate_paths: bool,
) -> SampleRecord:
    sample_id = _required_text(row.get(sample_id_column), sample_id_column, row_number)
    target = _optional_target(row.get(target_column), row_number=row_number)
    split = _optional_split(row.get(split_column), row_number=row_number)
    role_paths = {
        role: path
        for role, column in role_columns.items()
        if (path := _optional_path(row.get(column), manifest_path.parent)) is not None
    }
    if validate_paths:
        missing = [path for path in role_paths.values() if not path.is_file()]
        if missing:
            raise ManifestValidationError(
                f"Manifest row {row_number} sample {sample_id!r} has missing structure path: {missing[0]}"
            )
    identifiers = {
        name: value
        for name, column in identifier_columns.items()
        if (value := _optional_string(row.get(column))) is not None
    }
    return SampleRecord(
        sample_id=sample_id,
        target=target,
        split=split,
        role_paths=role_paths,
        identifiers=identifiers,
    )


def _required_text(value: object, column: str, row_number: int) -> str:
    normalized = _optional_string(value)
    if normalized is None:
        raise ManifestValidationError(
            f"Manifest row {row_number} is missing required {column!r}"
        )
    return normalized


def _optional_target(value: object, *, row_number: int) -> float | None:
    normalized = _optional_string(value)
    if normalized is None:
        return None
    try:
        target = float(normalized)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(
            f"Manifest row {row_number} target is not numeric: {normalized!r}"
        ) from exc
    if not math.isfinite(target):
        raise ManifestValidationError(
            f"Manifest row {row_number} target must be finite"
        )
    return target


def _optional_split(value: object, *, row_number: int) -> SampleSplit | None:
    normalized = _optional_string(value)
    if normalized is None:
        return None
    try:
        return SampleSplit(normalized.lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in SampleSplit)
        raise ManifestValidationError(
            f"Manifest row {row_number} has invalid split {normalized!r}; expected one of {allowed}"
        ) from exc


def _optional_path(value: object, base_dir: Path) -> Path | None:
    normalized = _optional_string(value)
    if normalized is None:
        return None
    path = Path(normalized).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _mapping_of_strings(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{name} must be a mapping")
    result = {str(key): str(item) for key, item in value.items()}
    if any(not key or not item for key, item in result.items()):
        raise ManifestValidationError(f"{name} keys and values must be nonempty")
    return result


def _optional_mapping(value: object, name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{name} must be a mapping")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
