from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mint_scout.data.geometry import load_atom_cloud
from mint_scout.data.structure_staging import file_sha256
from mint_scout.feature_qc import (
    _parse_manifest_args,
    _resolve_sample_ids,
    assert_qc_report_compatible,
)
from mint_scout.invariants.manifest import stable_hash
from mint_scout.invariants.plbind_tools import expected_feature_shape
from mint_scout.representation import RepresentationSpec


REPORT_SCHEMA = "mint-agent.feature-outlier-diagnostic.v1"
FPRC_STATISTIC_NAMES = (
    "minimum",
    "maximum",
    "mean",
    "standard_deviation",
    "positive_sum",
    "absolute_deviation_sum",
    "second_moment_sum",
    "positive_second_moment_sum",
    "pseudo_quasi_wiener_index",
    "third_absolute_deviation_sum",
)
FPRC_COMPONENT_NAMES = tuple(
    f"{source}_{statistic}"
    for source in ("node_curvature", "edge_curvature")
    for statistic in FPRC_STATISTIC_NAMES
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Locate the axes responsible for population-level feature outliers."
    )
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument("--probe-selection", type=Path, default=None)
    parser.add_argument("--sample-id-file", type=Path, default=None)
    parser.add_argument(
        "--feature-manifest",
        action="append",
        required=True,
        metavar="INVARIANT=PATH",
    )
    parser.add_argument("--qc-report", type=Path, default=None)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--reason-code", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--evidence-scope", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    representation = RepresentationSpec.read(args.representation_spec)
    representation.assert_frozen()
    sample_ids, selection_hash = _resolve_sample_ids(
        args.probe_selection, args.sample_id_file
    )
    manifests = _parse_manifest_args(args.feature_manifest)
    if len(manifests) != 1:
        raise ValueError("Outlier diagnosis accepts exactly one feature manifest per run")
    invariant, manifest_path = next(iter(manifests.items()))

    qc_report = None
    flagged = list(str(sample_id) for sample_id in args.sample_id)
    if args.qc_report is not None:
        qc_report = assert_qc_report_compatible(
            path=args.qc_report,
            representation=representation,
            sample_ids=sample_ids,
            manifest_paths={invariant: manifest_path},
        )
        if qc_report.get("selection_hash") != selection_hash:
            raise ValueError("Feature QC report selection hash does not match")
        reason_codes = tuple(args.reason_code or ("robust_outlier",))
        flagged.extend(
            flagged_samples_from_qc(qc_report, invariant, reason_codes=reason_codes)
        )
    flagged_sample_ids = tuple(dict.fromkeys(flagged))
    if not flagged_sample_ids:
        raise ValueError("No samples selected for outlier diagnosis")

    report = run_feature_outlier_diagnostic(
        representation=representation,
        sample_ids=sample_ids,
        invariant=invariant,
        manifest_path=manifest_path,
        flagged_sample_ids=flagged_sample_ids,
        top_k=args.top_k,
        selection_hash=selection_hash,
        dataset_id=args.dataset_id or _optional_string(qc_report, "dataset_id"),
        evidence_scope=args.evidence_scope
        or _optional_string(qc_report, "evidence_scope"),
        qc_hash=_optional_string(qc_report, "qc_hash"),
        sample_metadata=_probe_sample_metadata(args.probe_selection),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"status={report['status']} invariant={invariant} "
        f"flagged={len(flagged_sample_ids)} output={args.output}"
    )
    return 0


def flagged_samples_from_qc(
    report: Mapping[str, Any],
    invariant: str,
    *,
    reason_codes: Sequence[str] = ("robust_outlier",),
) -> tuple[str, ...]:
    accepted = set(str(code) for code in reason_codes)
    result = []
    for issue in report.get("issues", ()):
        if not isinstance(issue, Mapping):
            continue
        sample_id = issue.get("sample_id")
        if (
            str(issue.get("invariant", "")).upper() == invariant.upper()
            and str(issue.get("issue", "")) in accepted
            and sample_id is not None
        ):
            result.append(str(sample_id))
    return tuple(dict.fromkeys(result))


def run_feature_outlier_diagnostic(
    *,
    representation: RepresentationSpec,
    sample_ids: tuple[str, ...],
    invariant: str,
    manifest_path: Path,
    flagged_sample_ids: tuple[str, ...],
    top_k: int = 10,
    selection_hash: str | None = None,
    dataset_id: str | None = None,
    evidence_scope: str | None = None,
    qc_hash: str | None = None,
    sample_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    representation.assert_frozen()
    name = invariant.upper()
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be nonempty and unique")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    unknown = sorted(set(flagged_sample_ids) - set(sample_ids))
    if unknown:
        raise ValueError(f"Flagged samples are outside the selected population: {unknown}")

    expected_shape = expected_feature_shape(name, representation)
    rows = _load_manifest(
        manifest_path,
        invariant=name,
        sample_ids=sample_ids,
        representation_hash=representation.spec_hash,
    )
    profile = representation.filtration_profiles[name]
    axis_values = {
        "filtration": np.empty((len(sample_ids), expected_shape[0]), dtype=np.float64),
        "element_pair": np.empty((len(sample_ids), expected_shape[1]), dtype=np.float64),
        "component": np.empty((len(sample_ids), expected_shape[2]), dtype=np.float64),
    }
    total_norms = np.empty(len(sample_ids), dtype=np.float64)
    flagged_arrays: dict[str, np.ndarray] = {}
    flagged_set = set(flagged_sample_ids)

    for sample_index, sample_id in enumerate(sample_ids):
        feature_path = Path(str(rows[sample_id]["output_path"]))
        array = np.load(feature_path, allow_pickle=False)
        if tuple(array.shape) != expected_shape:
            raise ValueError(
                f"{sample_id}: expected feature shape {expected_shape}, got {tuple(array.shape)}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{sample_id}: feature contains NaN or Inf")
        values = np.asarray(array, dtype=np.float64)
        squares = values * values
        total_norms[sample_index] = float(np.sqrt(np.sum(squares)))
        axis_values["filtration"][sample_index] = np.sqrt(
            np.sum(squares, axis=(1, 2))
        )
        axis_values["element_pair"][sample_index] = np.sqrt(
            np.sum(squares, axis=(0, 2))
        )
        axis_values["component"][sample_index] = np.sqrt(
            np.sum(squares, axis=(0, 1))
        )
        if sample_id in flagged_set:
            flagged_arrays[sample_id] = values

    filtration_values = tuple(
        float(profile.start + index * profile.step)
        for index in range(profile.num_points)
    )
    labels: dict[str, tuple[Any, ...]] = {
        "filtration": filtration_values,
        "element_pair": tuple(representation.pair_names),
        "component": _component_names(name, expected_shape[2]),
    }
    sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    diagnostics = []
    for sample_id in flagged_sample_ids:
        index = sample_index[sample_id]
        array = flagged_arrays[sample_id]
        maximum_coordinate = _maximum_coordinate(
            array,
            filtration_values=filtration_values,
            pair_names=representation.pair_names,
            component_names=labels["component"],
        )
        diagnostics.append(
            {
                "sample_id": sample_id,
                "feature_path": str(rows[sample_id]["output_path"]),
                "l2_norm": float(total_norms[index]),
                "l2_norm_robust_z": _robust_z_value(total_norms, index),
                "nonzero_fraction": float(np.count_nonzero(array) / array.size),
                "maximum_absolute_coordinate": maximum_coordinate,
                "selection_structure_context": _selection_structure_context(
                    sample_id,
                    sample_ids=sample_ids,
                    sample_metadata=sample_metadata,
                ),
                "dominant_fprc_graph_context": _dominant_fprc_graph_context(
                    invariant=name,
                    row=rows[sample_id],
                    representation=representation,
                    maximum_coordinate=maximum_coordinate,
                ),
                "axes": {
                    axis: _axis_diagnostic(
                        axis_values[axis],
                        index=index,
                        labels=labels[axis],
                        total_norm=float(total_norms[index]),
                        top_k=top_k,
                    )
                    for axis in ("filtration", "element_pair", "component")
                },
            }
        )

    payload: dict[str, Any] = {
        "report_schema": REPORT_SCHEMA,
        "status": "COMPLETE",
        "dataset_id": dataset_id,
        "evidence_scope": evidence_scope,
        "invariant": name,
        "representation_hash": representation.spec_hash,
        "selection_hash": selection_hash,
        "sample_count": len(sample_ids),
        "sample_ids": list(sample_ids),
        "sample_order_hash": stable_hash(sample_ids),
        "flagged_sample_ids": list(flagged_sample_ids),
        "expected_shape": list(expected_shape),
        "axis_semantics": {
            "axis_0": f"{profile.scale_kind}_filtration",
            "axis_1": "element_pair",
            "axis_2": "legacy_feature_component",
        },
        "automatic_sample_removal": False,
        "automatic_representation_mutation": False,
        "source": {
            "feature_manifest": str(manifest_path),
            "feature_manifest_sha256": file_sha256(manifest_path),
            "feature_qc_hash": qc_hash,
        },
        "population_l2_norm": _population_summary(total_norms),
        "population_structure_context": _population_structure_context(
            sample_ids=sample_ids,
            sample_metadata=sample_metadata,
            total_norms=total_norms,
        ),
        "diagnostics": diagnostics,
    }
    payload["diagnostic_hash"] = stable_hash(
        {
            "representation_hash": representation.spec_hash,
            "selection_hash": selection_hash,
            "invariant": name,
            "feature_manifest_sha256": payload["source"]["feature_manifest_sha256"],
            "flagged_sample_ids": flagged_sample_ids,
            "top_k": top_k,
            "diagnostics": diagnostics,
        }
    )
    return payload


def _load_manifest(
    path: Path,
    *,
    invariant: str,
    sample_ids: tuple[str, ...],
    representation_hash: str,
) -> dict[str, Mapping[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"Feature manifest contains a non-object row: {path}")
        sample_id = str(row.get("sample_id", ""))
        if sample_id in by_id:
            raise ValueError(f"Duplicate sample in feature manifest: {sample_id}")
        by_id[sample_id] = row
    if set(by_id) != set(sample_ids):
        raise ValueError("Feature manifest sample IDs do not match the selected population")
    for sample_id, row in by_id.items():
        if str(row.get("invariant", "")).upper() != invariant:
            raise ValueError(f"{sample_id}: feature manifest invariant mismatch")
        if row.get("status") not in {"computed", "cached"}:
            raise ValueError(f"{sample_id}: feature generation did not succeed")
        feature_path = Path(str(row.get("output_path", "")))
        if not feature_path.exists():
            raise ValueError(f"{sample_id}: feature file does not exist: {feature_path}")
        if f"/repr-{representation_hash}/" not in feature_path.as_posix():
            raise ValueError(f"{sample_id}: feature path has the wrong representation hash")
    return by_id


def _axis_diagnostic(
    population: np.ndarray,
    *,
    index: int,
    labels: Sequence[Any],
    total_norm: float,
    top_k: int,
) -> dict[str, Any]:
    values = population[index]
    medians = np.median(population, axis=0)
    mads = np.median(np.abs(population - medians), axis=0)
    robust_z = np.full(values.shape, np.nan, dtype=np.float64)
    valid = mads > 0.0
    robust_z[valid] = 0.6745 * (values[valid] - medians[valid]) / mads[valid]
    denominator = total_norm * total_norm
    contributions = values * values / denominator if denominator > 0.0 else np.zeros_like(values)

    contribution_order = sorted(
        range(values.size), key=lambda item: (-float(contributions[item]), item)
    )[: min(top_k, values.size)]
    deviation_order = sorted(
        (item for item in range(values.size) if np.isfinite(robust_z[item])),
        key=lambda item: (-abs(float(robust_z[item])), item),
    )[: min(top_k, values.size)]
    return {
        "top_contributors": [
            _axis_entry(
                item,
                labels[item],
                values[item],
                contributions[item],
                robust_z[item],
                medians[item],
                mads[item],
            )
            for item in contribution_order
        ],
        "top_population_deviations": [
            _axis_entry(
                item,
                labels[item],
                values[item],
                contributions[item],
                robust_z[item],
                medians[item],
                mads[item],
            )
            for item in deviation_order
        ],
    }


def _axis_entry(
    index: int,
    label: Any,
    value: float,
    contribution: float,
    robust_z: float,
    median: float,
    mad: float,
) -> dict[str, Any]:
    return {
        "index": index,
        "label": label,
        "axis_l2_norm": float(value),
        "squared_l2_fraction": float(contribution),
        "population_robust_z": float(robust_z) if np.isfinite(robust_z) else None,
        "population_median": float(median),
        "population_mad": float(mad),
    }


def _maximum_coordinate(
    array: np.ndarray,
    *,
    filtration_values: Sequence[float],
    pair_names: Sequence[str],
    component_names: Sequence[str],
) -> dict[str, Any]:
    flat_index = int(np.argmax(np.abs(array)))
    filtration_index, pair_index, component_index = (
        int(value) for value in np.unravel_index(flat_index, array.shape)
    )
    return {
        "value": float(array[filtration_index, pair_index, component_index]),
        "absolute_value": float(
            abs(array[filtration_index, pair_index, component_index])
        ),
        "filtration_index": filtration_index,
        "filtration_value": float(filtration_values[filtration_index]),
        "element_pair_index": pair_index,
        "element_pair": pair_names[pair_index],
        "component_index": component_index,
        "component": component_names[component_index],
    }


def _selection_structure_context(
    sample_id: str,
    *,
    sample_ids: tuple[str, ...],
    sample_metadata: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    sizes = _structure_sizes(sample_ids, sample_metadata)
    if sizes is None or sample_metadata is None:
        return None
    index = sample_ids.index(sample_id)
    size = float(sizes[index])
    metadata = sample_metadata[sample_id]
    return {
        "structure_size": size,
        "size_bin": metadata.get("size_bin"),
        "structure_size_robust_z": _robust_z_value(sizes, index),
        "descending_size_rank": int(1 + np.count_nonzero(sizes > size)),
        "empirical_percentile": float(np.count_nonzero(sizes <= size) / sizes.size),
    }


def _population_structure_context(
    *,
    sample_ids: tuple[str, ...],
    sample_metadata: Mapping[str, Mapping[str, Any]] | None,
    total_norms: np.ndarray,
) -> dict[str, Any] | None:
    sizes = _structure_sizes(sample_ids, sample_metadata)
    if sizes is None:
        return None
    positive = (sizes > 0.0) & (total_norms > 0.0)
    correlation = None
    if np.count_nonzero(positive) >= 2:
        correlation = float(
            np.corrcoef(np.log1p(sizes[positive]), np.log1p(total_norms[positive]))[0, 1]
        )
    return {
        **_population_summary(sizes),
        "log_structure_size_vs_log_l2_norm_pearson": correlation,
    }


def _structure_sizes(
    sample_ids: tuple[str, ...],
    sample_metadata: Mapping[str, Mapping[str, Any]] | None,
) -> np.ndarray | None:
    if sample_metadata is None:
        return None
    try:
        values = np.asarray(
            [float(sample_metadata[sample_id]["structure_size"]) for sample_id in sample_ids],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return values if np.isfinite(values).all() else None


def _dominant_fprc_graph_context(
    *,
    invariant: str,
    row: Mapping[str, Any],
    representation: RepresentationSpec,
    maximum_coordinate: Mapping[str, Any],
) -> dict[str, Any] | None:
    if invariant != "FPRC" or representation.system_type != "protein_ligand":
        return None
    protein_path = row.get("protein_path")
    ligand_path = row.get("ligand_path")
    if not protein_path or not ligand_path:
        return None
    pair_index = int(maximum_coordinate["element_pair_index"])
    protein_element, ligand_element = representation.pair_order[pair_index]
    protein = load_atom_cloud(str(protein_path))
    ligand = load_atom_cloud(str(ligand_path))
    protein_coordinates = protein.coordinates_for(protein_element)
    ligand_coordinates = ligand.coordinates_for(ligand_element)
    protein_count = int(len(protein_coordinates))
    ligand_count = int(len(ligand_coordinates))
    possible_edge_count = protein_count * ligand_count
    if possible_edge_count == 0:
        return {
            "protein_element": protein_element,
            "ligand_element": ligand_element,
            "protein_element_count": protein_count,
            "ligand_element_count": ligand_count,
            "possible_edge_count": 0,
            "edge_count": 0,
            "edge_density": 0.0,
        }
    differences = protein_coordinates[:, None, :] - ligand_coordinates[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    adjacency = distances <= float(maximum_coordinate["filtration_value"])
    protein_degrees = np.sum(adjacency, axis=1).astype(np.float64)
    ligand_degrees = np.sum(adjacency, axis=0).astype(np.float64)
    protein_edge_indices, ligand_edge_indices = np.nonzero(adjacency)
    edge_curvature = (
        4.0
        - protein_degrees[protein_edge_indices]
        - ligand_degrees[ligand_edge_indices]
    )
    edge_count = int(edge_curvature.size)
    return {
        "protein_element": protein_element,
        "ligand_element": ligand_element,
        "protein_element_count": protein_count,
        "ligand_element_count": ligand_count,
        "possible_edge_count": possible_edge_count,
        "edge_count": edge_count,
        "edge_density": edge_count / float(possible_edge_count),
        "protein_degree": _population_summary(protein_degrees),
        "ligand_degree": _population_summary(ligand_degrees),
        "edge_curvature": _population_summary(edge_curvature)
        if edge_count
        else None,
        "formula": "edge_curvature = 4 - protein_degree - ligand_degree",
    }


def _component_names(invariant: str, size: int) -> tuple[str, ...]:
    if invariant == "FPRC" and size == len(FPRC_COMPONENT_NAMES):
        return FPRC_COMPONENT_NAMES
    return tuple(f"component_{index}" for index in range(size))


def _robust_z_value(values: np.ndarray, index: int) -> float | None:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0.0:
        return None
    return float(0.6745 * (values[index] - median) / mad)


def _population_summary(values: np.ndarray) -> dict[str, float]:
    median = float(np.median(values))
    return {
        "minimum": float(np.min(values)),
        "median": median,
        "mad": float(np.median(np.abs(values - median))),
        "maximum": float(np.max(values)),
    }


def _optional_string(
    mapping: Mapping[str, Any] | None, key: str
) -> str | None:
    if mapping is None or mapping.get(key) is None:
        return None
    return str(mapping[key])


def _probe_sample_metadata(
    probe_selection: Path | None,
) -> dict[str, Mapping[str, Any]] | None:
    if probe_selection is None:
        return None
    payload = json.loads(probe_selection.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return None
    rows = payload.get("probe_samples")
    if not isinstance(rows, list):
        return None
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and row.get("sample_id") is not None:
            result[str(row["sample_id"])] = row
    return result or None


if __name__ == "__main__":
    raise SystemExit(main())
