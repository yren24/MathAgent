from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from mint_scout.config import load_yaml
from mint_scout.invariants.manifest import stable_hash
from mint_scout.models.gbt import GBTConfig


EVIDENCE_SCOPES = {
    "design",
    "smoke",
    "probe",
    "full_train",
    "validation",
    "external_test",
    "acceptance_test",
    "inference",
    "unspecified",
}


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    artifact_kind: str
    path: str
    content_sha256: str
    dataset_id: str | None
    split: str | None
    evidence_scope: str
    invariants: tuple[str, ...]
    representation_hash: str | None
    selection_hash: str | None
    sample_count: int | None
    sample_order_hash: str | None
    status: str | None
    report_schema: str | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.evidence_scope not in EVIDENCE_SCOPES:
            raise ValueError(
                f"Unknown evidence scope {self.evidence_scope!r}; expected one of {sorted(EVIDENCE_SCOPES)}"
            )
        if self.sample_count is not None and self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["invariants"] = list(self.invariants)
        payload["metadata"] = dict(self.metadata)
        return payload


class ArtifactRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_kind TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    content_sha256 TEXT NOT NULL,
                    dataset_id TEXT,
                    split TEXT,
                    evidence_scope TEXT NOT NULL,
                    invariants_json TEXT NOT NULL,
                    representation_hash TEXT,
                    selection_hash TEXT,
                    sample_count INTEGER,
                    sample_order_hash TEXT,
                    status TEXT,
                    report_schema TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS artifacts_lookup
                ON artifacts (
                    artifact_kind,
                    dataset_id,
                    split,
                    evidence_scope,
                    representation_hash,
                    status
                );
                """
            )

    def register(self, record: ArtifactRecord) -> ArtifactRecord:
        self.initialize()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, artifact_kind, path, content_sha256, dataset_id, split,
                    evidence_scope, invariants_json, representation_hash, selection_hash,
                    sample_count, sample_order_hash, status, report_schema, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    artifact_id=excluded.artifact_id,
                    artifact_kind=excluded.artifact_kind,
                    content_sha256=excluded.content_sha256,
                    dataset_id=excluded.dataset_id,
                    split=excluded.split,
                    evidence_scope=excluded.evidence_scope,
                    invariants_json=excluded.invariants_json,
                    representation_hash=excluded.representation_hash,
                    selection_hash=excluded.selection_hash,
                    sample_count=excluded.sample_count,
                    sample_order_hash=excluded.sample_order_hash,
                    status=excluded.status,
                    report_schema=excluded.report_schema,
                    metadata_json=excluded.metadata_json
                """,
                _record_values(record),
            )
        return record

    def query(
        self,
        *,
        artifact_kind: str | None = None,
        dataset_id: str | None = None,
        split: str | None = None,
        evidence_scope: str | None = None,
        invariant: str | None = None,
        representation_hash: str | None = None,
        status: str | None = None,
    ) -> tuple[ArtifactRecord, ...]:
        if evidence_scope is not None and evidence_scope not in EVIDENCE_SCOPES:
            raise ValueError(f"Unknown evidence scope {evidence_scope!r}")
        if not self.path.exists():
            return ()
        clauses: list[str] = []
        values: list[Any] = []
        for column, value in (
            ("artifact_kind", artifact_kind),
            ("dataset_id", dataset_id),
            ("split", split),
            ("evidence_scope", evidence_scope),
            ("representation_hash", representation_hash),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                values.append(value)
        sql = "SELECT * FROM artifacts"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY artifact_kind, path"
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            records = tuple(_row_to_record(row) for row in connection.execute(sql, values))
        if invariant is None:
            return records
        normalized = invariant.upper()
        return tuple(record for record in records if normalized in record.invariants)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register and query deterministic pipeline artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--registry", type=Path, required=True)
    register_parser.add_argument("--artifact", type=Path, required=True)
    register_parser.add_argument("--kind", default=None)
    register_parser.add_argument("--dataset-id", default=None)
    register_parser.add_argument("--split", default=None)
    register_parser.add_argument("--evidence-scope", choices=sorted(EVIDENCE_SCOPES), default=None)
    register_parser.add_argument("--invariant", action="append", default=None)
    register_parser.add_argument("--representation-hash", default=None)
    register_parser.add_argument("--selection-hash", default=None)
    register_parser.add_argument("--status", default=None)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--registry", type=Path, required=True)
    query_parser.add_argument("--kind", default=None)
    query_parser.add_argument("--dataset-id", default=None)
    query_parser.add_argument("--split", default=None)
    query_parser.add_argument("--evidence-scope", choices=sorted(EVIDENCE_SCOPES), default=None)
    query_parser.add_argument("--invariant", default=None)
    query_parser.add_argument("--representation-hash", default=None)
    query_parser.add_argument("--status", default=None)

    batch_parser = subparsers.add_parser("register-batch")
    batch_parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "register-batch":
        registered = register_batch(args.spec)
        print(json.dumps([record.to_dict() for record in registered], indent=2, sort_keys=True))
        return 0

    registry = ArtifactRegistry(args.registry)
    if args.command == "register":
        record = inspect_artifact(args.artifact)
        record = replace(
            record,
            artifact_kind=args.kind or record.artifact_kind,
            dataset_id=args.dataset_id or record.dataset_id,
            split=args.split or record.split,
            evidence_scope=args.evidence_scope or record.evidence_scope,
            invariants=(
                tuple(dict.fromkeys(value.upper() for value in args.invariant))
                if args.invariant
                else record.invariants
            ),
            representation_hash=args.representation_hash or record.representation_hash,
            selection_hash=args.selection_hash or record.selection_hash,
            status=args.status or record.status,
        )
        registry.register(record)
        print(json.dumps(record.to_dict(), sort_keys=True))
        return 0

    records = registry.query(
        artifact_kind=args.kind,
        dataset_id=args.dataset_id,
        split=args.split,
        evidence_scope=args.evidence_scope,
        invariant=args.invariant,
        representation_hash=args.representation_hash,
        status=args.status,
    )
    print(json.dumps([record.to_dict() for record in records], indent=2, sort_keys=True))
    return 0


def register_batch(path: str | Path) -> tuple[ArtifactRecord, ...]:
    spec = load_yaml(path)
    if "registry" not in spec:
        raise ValueError("artifact batch spec requires registry")
    entries = spec.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("artifact batch spec requires a nonempty artifacts list")
    registry = ArtifactRegistry(Path(str(spec["registry"])).expanduser())
    records = []
    allowed = {
        "path",
        "kind",
        "dataset_id",
        "split",
        "evidence_scope",
        "invariants",
        "representation_hash",
        "selection_hash",
        "status",
    }
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"artifacts[{index}] must be a mapping")
        unknown = sorted(set(entry) - allowed)
        if unknown:
            raise ValueError(f"Unknown artifacts[{index}] fields: {unknown}")
        if "path" not in entry:
            raise ValueError(f"artifacts[{index}].path is required")
        record = inspect_artifact(str(entry["path"]))
        invariants = entry.get("invariants")
        if invariants is not None:
            if not isinstance(invariants, list):
                raise ValueError(f"artifacts[{index}].invariants must be a list")
            normalized_invariants = tuple(
                dict.fromkeys(str(value).upper() for value in invariants)
            )
        else:
            normalized_invariants = record.invariants
        record = replace(
            record,
            artifact_kind=str(entry.get("kind", record.artifact_kind)),
            dataset_id=_optional_string(entry.get("dataset_id", record.dataset_id)),
            split=_optional_string(entry.get("split", record.split)),
            evidence_scope=str(entry.get("evidence_scope", record.evidence_scope)),
            invariants=normalized_invariants,
            representation_hash=_optional_string(
                entry.get("representation_hash", record.representation_hash)
            ),
            selection_hash=_optional_string(entry.get("selection_hash", record.selection_hash)),
            status=_optional_string(entry.get("status", record.status)),
        )
        registry.register(record)
        records.append(record)
    return tuple(records)


def inspect_artifact(path: str | Path) -> ArtifactRecord:
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    content_sha256 = _file_sha256(artifact_path)
    if artifact_path.suffix == ".jsonl":
        summary = _inspect_jsonl(artifact_path)
    else:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {artifact_path}")
        summary = _inspect_json(payload)
    artifact_id = stable_hash(
        {
            "artifact_kind": summary["artifact_kind"],
            "path": str(artifact_path),
            "content_sha256": content_sha256,
        }
    )
    return ArtifactRecord(
        artifact_id=artifact_id,
        artifact_kind=summary["artifact_kind"],
        path=str(artifact_path),
        content_sha256=content_sha256,
        dataset_id=summary.get("dataset_id"),
        split=summary.get("split"),
        evidence_scope=summary.get("evidence_scope", "unspecified"),
        invariants=tuple(summary.get("invariants", ())),
        representation_hash=summary.get("representation_hash"),
        selection_hash=summary.get("selection_hash"),
        sample_count=summary.get("sample_count"),
        sample_order_hash=summary.get("sample_order_hash"),
        status=summary.get("status"),
        report_schema=summary.get("report_schema"),
        metadata=summary.get("metadata", {}),
    )


def _inspect_json(payload: Mapping[str, Any]) -> dict[str, Any]:
    report_schema = payload.get("report_schema")
    artifact_version = payload.get("artifact_version")
    schema = str(report_schema or artifact_version or "")
    kind_by_schema = {
        "mint-agent.dataset-audit.v1": "dataset_audit",
        "mint-agent.user-preflight.v1": "user_preflight",
        "mint-agent.dataset-preparation.v1": "dataset_preparation",
        "mint-agent.bdb2020plus-preparation.v1": "dataset_preparation",
        "mint-agent.hiqbind-preparation.v1": "dataset_preparation",
        "mint-agent.representation-design.v1": "representation_design",
        "mint-agent.filtration-repair.v1": "filtration_repair",
        "mint-agent.controlled-experiment-matrix.v1": "controlled_experiment_plan",
        "mint-agent.controlled-experiment-matrix.v2": "controlled_experiment_plan",
        "mint-agent.controlled-probe-selection.v1": "controlled_probe_selection",
        "mint-agent.controlled-representation-selection.v1": "controlled_representation_selection",
        "mint-agent.controlled-representation-selection.v2": "controlled_representation_selection",
        "mint-agent.casf-probe-selection.v1": "probe_selection",
        "mint-agent.probe-representativeness-audit.v1": "probe_audit",
        "mint-agent.feature-qc.v1": "feature_qc",
        "mint-agent.filtration-axis-audit.v1": "filtration_audit",
        "mint-agent.feature-outlier-diagnostic.v1": "feature_outlier_diagnostic",
        "mint-agent.casf-scout.v1": "scout_oof",
        "mint-agent.combined-scout.v1": "scout_combined",
        "mint-agent.legacy-pl-evaluation.v1": "model_evaluation",
        "mint-agent.representation-comparison.v1": "representation_comparison",
        "mint-agent.filtration-comparison.v1": "filtration_comparison",
        "mint-agent.graph-run.v1": "agent_run",
        "mint-agent.slurm-job-ledger.v1": "job_ledger",
        "mint-agent.phase4.casf.v1": "model_evaluation",
        "mint-agent.external-gbt-evaluation.v1": "model_evaluation",
        "mint-agent.split-gbt-evaluation.v1": "model_evaluation",
        "mint-agent.validation-gbt-evaluation.v1": "model_evaluation",
        "mint-agent.probe-rank-selection.v1": "model_evaluation",
        "mint-agent.frozen-test-gbt-evaluation.v1": "model_evaluation",
        "mint-agent.acceptance-gbt-evaluation.v1": "model_evaluation",
        "mint-agent.progressive-acceptance-gbt-evaluation.v1": "model_evaluation",
        "mint-agent.scout-execution.v1": "scout_execution_plan",
    }
    artifact_kind = kind_by_schema.get(schema, "json_artifact")
    sample_ids = _sample_ids_from_payload(payload)
    invariants = _invariants_from_payload(payload)
    scope = _evidence_scope_from_payload(payload, artifact_kind)
    status = payload.get("status")
    if status is None and isinstance(payload.get("result"), Mapping):
        status = payload["result"].get("status")
    metadata = {
        key: payload[key]
        for key in (
            "probe_hash",
            "combined_probe_hash",
            "feature_qc_hash",
            "qc_hash",
            "diagnostic_hash",
            "target_metric",
            "target_direction",
            "target_value",
            "gbt_parameter_hash",
            "frozen_priority_order",
            "max_acquisitions",
            "selection_objective",
            "objective_complete",
            "scout_artifact",
            "fold_source",
            "priority_source",
            "run_kind",
            "evaluation_hash",
            "audit_input_hash",
            "design_input_hash",
            "request_hash",
            "comparison_hash",
            "candidate_rank",
            "candidate_count",
            "selected_score",
            "scout_probe_hash",
            "ranking_policy",
            "shared_probe_hash",
            "experiment_hash",
            "parent_representation_hash",
            "repair_round",
            "audit_representation_hash",
        )
        if key in payload
    }
    prediction_reuse = payload.get("prediction_reuse")
    if isinstance(prediction_reuse, Mapping):
        metadata["prediction_reuse"] = dict(prediction_reuse)
    source = payload.get("source")
    if isinstance(source, Mapping):
        for key in ("feature_qc_hash", "feature_manifest_sha256"):
            if key in source and key not in metadata:
                metadata[key] = source[key]
    scout = payload.get("scout")
    if isinstance(scout, Mapping):
        for key in ("probe_hash", "gbt_parameter_hash"):
            if key in scout and key not in metadata:
                metadata[key] = scout[key]
    if artifact_kind == "model_evaluation" and "gbt_parameter_hash" not in metadata:
        embedded_gbt = payload.get("gbt_config")
        if isinstance(embedded_gbt, Mapping):
            try:
                metadata["gbt_parameter_hash"] = GBTConfig(
                    **dict(embedded_gbt)
                ).parameter_hash
            except (TypeError, ValueError):
                pass
            else:
                metadata["gbt_parameter_hash_source"] = (
                    "derived_from_embedded_gbt_config"
                )
    fold_assignment = payload.get("full_fold_assignment")
    result = payload.get("result")
    if not isinstance(fold_assignment, Mapping) and isinstance(result, Mapping):
        fold_assignment = result.get("evaluation_fold_assignment")
    if isinstance(fold_assignment, Mapping):
        try:
            normalized_folds = {
                str(sample_id): int(fold_id)
                for sample_id, fold_id in fold_assignment.items()
            }
        except (TypeError, ValueError):
            pass
        else:
            metadata["fold_assignment_hash"] = stable_hash(normalized_folds)
    selection = payload.get("selection")
    if (
        isinstance(selection, Mapping)
        and isinstance(selection.get("selected_subset"), Sequence)
        and not isinstance(selection.get("selected_subset"), (str, bytes))
    ):
        metadata["selected_subset"] = [
            str(value).upper() for value in selection["selected_subset"]
        ]
    elif isinstance(payload.get("selected_subset"), Sequence) and not isinstance(
        payload.get("selected_subset"), (str, bytes)
    ):
        metadata["selected_subset"] = [
            str(value).upper() for value in payload["selected_subset"]
        ]
    elif isinstance(result, Mapping) and isinstance(
        result.get("selected_subset"), Sequence
    ):
        metadata["selected_subset"] = [
            str(value).upper() for value in result["selected_subset"]
        ]
    split_sample_counts = payload.get("sample_counts")
    split_order_hashes = payload.get("sample_order_hashes")
    formal_test_count = (
        int(split_sample_counts["test"])
        if schema in {
            "mint-agent.split-gbt-evaluation.v1",
            "mint-agent.frozen-test-gbt-evaluation.v1",
        }
        and isinstance(split_sample_counts, Mapping)
        and split_sample_counts.get("test") is not None
        else None
    )
    formal_test_order_hash = (
        str(split_order_hashes["test"])
        if schema in {
            "mint-agent.split-gbt-evaluation.v1",
            "mint-agent.frozen-test-gbt-evaluation.v1",
        }
        and isinstance(split_order_hashes, Mapping)
        and split_order_hashes.get("test") is not None
        else None
    )
    return {
        "artifact_kind": artifact_kind,
        "dataset_id": (
            payload.get("task_id")
            or payload.get("dataset_id")
            or payload.get("external_dataset_id")
        ),
        "split": payload.get("split") or (
            "test"
            if scope == "external_test"
            else "train" if payload.get("modeling_scope") == "train" else None
        ),
        "evidence_scope": scope,
        "invariants": invariants,
        "representation_hash": _representation_hash_from_payload(payload),
        "selection_hash": payload.get("selection_hash"),
        "sample_count": formal_test_count or _sample_count(payload, sample_ids),
        "sample_order_hash": (
            stable_hash(sample_ids)
            if sample_ids
            else payload.get("sample_order_hash")
            or payload.get("evaluation_sample_order_hash")
            or formal_test_order_hash
        ),
        "status": str(status) if status is not None else None,
        "report_schema": schema or None,
        "metadata": metadata,
    }


def _inspect_jsonl(path: Path) -> dict[str, Any]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"JSONL artifact is empty: {path}")
    sample_ids = tuple(str(row.get("sample_id")) for row in rows)
    if any(value == "None" for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"JSONL artifact sample IDs are missing or duplicated: {path}")
    invariants = tuple(sorted({str(row["invariant"]).upper() for row in rows if row.get("invariant")}))
    splits = {str(row["split"]) for row in rows if row.get("split") is not None}
    statuses = {str(row["status"]) for row in rows if row.get("status") is not None}
    structure_hashes = tuple(
        str(structure_inputs["combined_sha256"])
        for row in rows
        if isinstance((structure_inputs := row.get("structure_inputs")), Mapping)
        and structure_inputs.get("combined_sha256")
    )
    if statuses and statuses.issubset({"cached", "computed"}):
        status = "COMPLETE"
    elif "failed" in statuses:
        status = "FAILED"
    else:
        status = statuses.pop() if len(statuses) == 1 else "MIXED"
    metadata = {
        "failed_count": sum(row.get("status") == "failed" for row in rows),
        "cached_count": sum(row.get("status") == "cached" for row in rows),
        "computed_count": sum(row.get("status") == "computed" for row in rows),
    }
    if len(structure_hashes) == len(rows):
        metadata["structure_input_set_hash"] = stable_hash(structure_hashes)
    return {
        "artifact_kind": "feature_manifest",
        "dataset_id": None,
        "split": splits.pop() if len(splits) == 1 else None,
        "evidence_scope": "unspecified",
        "invariants": invariants,
        "representation_hash": None,
        "selection_hash": None,
        "sample_count": len(sample_ids),
        "sample_order_hash": stable_hash(sample_ids),
        "status": status,
        "report_schema": "mint-agent.feature-manifest-jsonl.v1",
        "metadata": metadata,
    }


def _sample_ids_from_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("sample_ids", "probe_sample_ids", "modeling_sample_ids"):
        values = payload.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            return tuple(str(value) for value in values)
    scout = payload.get("scout")
    if isinstance(scout, Mapping):
        values = scout.get("probe_sample_ids")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            return tuple(str(value) for value in values)
    return ()


def _representation_hash_from_payload(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("representation_hash")
    if value is not None and str(value):
        return str(value)
    nested = payload.get("representation_spec")
    if isinstance(nested, Mapping):
        value = nested.get("spec_hash")
        if value is not None and str(value):
            return str(value)
    return None


def _sample_count(payload: Mapping[str, Any], sample_ids: tuple[str, ...]) -> int | None:
    if sample_ids:
        return len(sample_ids)
    for key in (
        "sample_count",
        "probe_sample_count",
        "probe_final_size",
        "modeling_sample_count",
        "evaluation_sample_count",
    ):
        if payload.get(key) is not None:
            return int(payload[key])
    return None


def _invariants_from_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    names: set[str] = set()
    if payload.get("invariant") is not None:
        names.add(str(payload["invariant"]).upper())
    for key in ("feature_manifests", "invariants"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            nested_manifests = tuple(
                nested
                for nested in value.values()
                if isinstance(nested, Mapping)
            )
            if nested_manifests and len(nested_manifests) == len(value):
                for nested in nested_manifests:
                    names.update(str(name).upper() for name in nested)
            else:
                names.update(str(name).upper() for name in value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            names.update(str(name).upper() for name in value)
    result = payload.get("result")
    if isinstance(result, Mapping):
        for key in ("acquisition_order", "selected_subset"):
            value = result.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                names.update(str(name).upper() for name in value)
    priority_order = payload.get("frozen_priority_order")
    if isinstance(priority_order, Sequence) and not isinstance(priority_order, (str, bytes)):
        for subset in priority_order:
            if isinstance(subset, Sequence) and not isinstance(subset, (str, bytes)):
                names.update(str(name).upper() for name in subset)
    jobs = payload.get("jobs")
    if isinstance(jobs, Sequence) and not isinstance(jobs, (str, bytes)):
        for job in jobs:
            if not isinstance(job, Mapping):
                continue
            plan = job.get("plan")
            if isinstance(plan, Mapping):
                plan_invariants = plan.get("invariants")
                if isinstance(plan_invariants, Sequence) and not isinstance(
                    plan_invariants, (str, bytes)
                ):
                    names.update(str(value).upper() for value in plan_invariants)
            environment = plan.get("environment") if isinstance(plan, Mapping) else None
            if isinstance(environment, Mapping) and environment.get("INVARIANT"):
                names.add(str(environment["INVARIANT"]).upper())
    return tuple(sorted(names))


def _evidence_scope_from_payload(payload: Mapping[str, Any], artifact_kind: str) -> str:
    if payload.get("evidence_scope") in EVIDENCE_SCOPES:
        return str(payload["evidence_scope"])
    if artifact_kind in {
        "probe_selection",
        "probe_audit",
        "scout_oof",
        "scout_combined",
        "scout_execution_plan",
    }:
        return "probe"
    if artifact_kind in {
        "dataset_audit",
        "representation_design",
        "filtration_repair",
    }:
        return "design"
    if artifact_kind == "model_evaluation":
        if payload.get("mode") == "historical-fixed-test":
            return "external_test"
        if payload.get("mode") == "train-cv":
            return "full_train"
        if payload.get("run_kind") == "engineering_smoke":
            return "smoke"
        if payload.get("evaluation_mode") == "explicit_labeled_test":
            return "external_test"
        if payload.get("split") == "train":
            return "full_train"
    if artifact_kind == "feature_qc" and payload.get("selection_hash"):
        return "probe"
    return "unspecified"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _record_values(record: ArtifactRecord) -> tuple[Any, ...]:
    return (
        record.artifact_id,
        record.artifact_kind,
        record.path,
        record.content_sha256,
        record.dataset_id,
        record.split,
        record.evidence_scope,
        json.dumps(record.invariants),
        record.representation_hash,
        record.selection_hash,
        record.sample_count,
        record.sample_order_hash,
        record.status,
        record.report_schema,
        json.dumps(dict(record.metadata), sort_keys=True),
    )


def _row_to_record(row: sqlite3.Row) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=row["artifact_id"],
        artifact_kind=row["artifact_kind"],
        path=row["path"],
        content_sha256=row["content_sha256"],
        dataset_id=row["dataset_id"],
        split=row["split"],
        evidence_scope=row["evidence_scope"],
        invariants=tuple(json.loads(row["invariants_json"])),
        representation_hash=row["representation_hash"],
        selection_hash=row["selection_hash"],
        sample_count=row["sample_count"],
        sample_order_hash=row["sample_order_hash"],
        status=row["status"],
        report_schema=row["report_schema"],
        metadata=json.loads(row["metadata_json"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
