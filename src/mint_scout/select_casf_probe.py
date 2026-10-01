from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from mint_scout.compute_feature_batch import _selected_records
from mint_scout.config import load_yaml
from mint_scout.data.element_inventory import iter_elements_from_file
from mint_scout.data.splits import (
    assert_no_test_leakage,
    make_group_kfold_assignments,
    make_kfold_assignments,
)
from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation import RepresentationMode, RepresentationSpec
from mint_scout.scout.pipeline import ScoutConfig
from mint_scout.scout.sampling import (
    select_probe_samples,
    select_uniform_random_probe_samples,
)
from mint_scout.scout.sampling import ProbeSelection, quantile_bin_assignments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a deterministic protein-ligand Scout probe from the modeling pool."
    )
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--scout-config", type=Path, required=True)
    parser.add_argument("--representation-spec", type=Path, required=True)
    parser.add_argument(
        "--frozen-sample-source",
        type=Path,
        default=None,
        help="Reuse sample IDs from an existing probe while recomputing representation-specific pair coverage.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("stratified_pair_coverage", "uniform_random"),
        default=None,
        help="Override the configured Probe strategy for a controlled experiment.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Override the configured Probe seed for a controlled experiment.",
    )
    parser.add_argument(
        "--probe-max-samples",
        type=int,
        default=None,
        help="Cap Probe size for a controlled equal-budget comparison.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    task_config = load_yaml(args.task_config)
    scout_config = ScoutConfig.from_mapping(load_yaml(args.scout_config))
    if args.frozen_sample_source is not None and (
        args.sampling_strategy is not None
        or args.random_seed is not None
        or args.probe_max_samples is not None
    ):
        raise ValueError(
            "frozen-sample-source cannot be combined with a sampling-strategy or random-seed override"
        )
    if (
        args.sampling_strategy is not None
        or args.random_seed is not None
        or args.probe_max_samples is not None
    ):
        scout_config = replace(
            scout_config,
            probe=replace(
                scout_config.probe,
                sampling_strategy=args.sampling_strategy or scout_config.probe.sampling_strategy,
                random_seed=(
                    args.random_seed
                    if args.random_seed is not None
                    else scout_config.probe.random_seed
                ),
                max_samples=(
                    args.probe_max_samples
                    if args.probe_max_samples is not None
                    else scout_config.probe.max_samples
                ),
            ),
        )
    representation = RepresentationSpec.read(args.representation_spec)
    _validate_representation(representation)

    train_records = _selected_records(
        task_config,
        sample_ids=None,
        split="train",
        offset=0,
        limit=None,
        config_path=args.task_config,
    )
    test_records = _selected_records(
        task_config,
        sample_ids=None,
        split="test",
        offset=0,
        limit=None,
        config_path=args.task_config,
    )
    modeling_ids = tuple(record.pdb_id for record in train_records)
    _validate_design_scope(args.representation_spec, modeling_ids)

    targets = np.asarray([record.label for record in train_records], dtype=float)
    structure_sizes: list[float] = []
    pair_presence: dict[str, tuple[tuple[str, str], ...]] = {}
    retained_pairs = tuple(representation.pair_order)
    for record in train_records:
        if record.protein_path is None or record.ligand_path is None:
            raise ValueError(f"Sample {record.pdb_id!r} is missing protein/ligand paths")
        protein_elements = tuple(iter_elements_from_file(record.protein_path))
        ligand_elements = tuple(iter_elements_from_file(record.ligand_path))
        structure_sizes.append(float(len(protein_elements) + len(ligand_elements)))
        protein_set = set(protein_elements)
        ligand_set = set(ligand_elements)
        pair_presence[record.pdb_id] = tuple(
            pair for pair in retained_pairs if pair[0] in protein_set and pair[1] in ligand_set
        )

    frozen_source = (
        _load_frozen_source(args.frozen_sample_source)
        if args.frozen_sample_source is not None
        else None
    )
    selection = (
        _selection_from_frozen_source(
            frozen_source,
            modeling_ids=modeling_ids,
            targets=targets,
            structure_sizes=structure_sizes,
            retained_pairs=retained_pairs,
            pair_presence=pair_presence,
            min_pair_support=scout_config.probe.min_pair_support,
        )
        if frozen_source is not None
        else _select_probe(
            sample_ids=modeling_ids,
            targets=targets,
            structure_sizes=structure_sizes,
            retained_pairs=retained_pairs,
            pair_presence=pair_presence,
            scout_config=scout_config,
        )
    )
    cv_config = task_config.get("cv", {})
    if not isinstance(cv_config, dict):
        raise ValueError("task cv config must be a mapping")
    group_identifier = (
        str(cv_config.get("group_identifier") or "").strip() or None
    )
    if group_identifier is None:
        fold_strategy = "shuffled_sample_kfold"
        group_by_sample: dict[str, str] = {}
        full_folds = make_kfold_assignments(
            modeling_ids,
            n_splits=scout_config.cv_folds,
            seed=scout_config.fold_seed,
        )
    else:
        fold_strategy = "balanced_group_kfold"
        missing_groups = [
            record.pdb_id for record in train_records if record.group_id is None
        ]
        if missing_groups:
            raise ValueError(
                f"CV group identifier {group_identifier!r} is missing for samples: "
                + ", ".join(missing_groups[:10])
            )
        group_by_sample = {
            record.pdb_id: str(record.group_id).strip().casefold()
            for record in train_records
        }
        full_folds = make_group_kfold_assignments(
            modeling_ids,
            group_by_sample=group_by_sample,
            n_splits=scout_config.cv_folds,
            seed=scout_config.fold_seed,
        )
    fold_sample_counts = Counter(full_folds.values())
    if group_by_sample:
        fold_by_group: dict[str, int] = {}
        for sample_id, group_id in group_by_sample.items():
            fold_id = full_folds[sample_id]
            prior = fold_by_group.setdefault(group_id, fold_id)
            if prior != fold_id:
                raise AssertionError(f"CV group {group_id!r} crosses folds")
        fold_group_counts = Counter(fold_by_group.values())
    else:
        fold_group_counts = Counter(full_folds.values())
    assert_no_test_leakage(
        test_sample_ids=[record.pdb_id for record in test_records],
        fold_ids=full_folds,
        fidelity_sample_ids={1.0: selection.sample_ids},
        evidence_sample_ids=selection.sample_ids,
    )
    target_by_id = {record.pdb_id: record.label for record in train_records}
    size_by_id = dict(zip(modeling_ids, structure_sizes))
    probe_folds = {
        sample_id: full_folds[sample_id] for sample_id in selection.sample_ids
    }
    if frozen_source is not None:
        source_modeling_ids = tuple(
            str(value) for value in frozen_source.get("modeling_sample_ids", ())
        )
        source_folds = {
            str(key): int(value)
            for key, value in frozen_source.get("modeling_fold_assignment", {}).items()
        }
        if source_modeling_ids != modeling_ids or source_folds != full_folds:
            raise ValueError(
                "Frozen probe source does not share the modeling pool and fold assignment"
            )
    shared_probe_hash = stable_hash(
        {
            "modeling_sample_ids": modeling_ids,
            "probe_sample_ids": selection.sample_ids,
            "modeling_fold_assignment": full_folds,
            "probe_fold_assignment": probe_folds,
            "fold_strategy": fold_strategy,
            "group_identifier": group_identifier,
        }
    )
    selection_hash = stable_hash(
        {
            "representation_hash": representation.spec_hash,
            "sample_ids": selection.sample_ids,
            "probe_folds": probe_folds,
            "shared_probe_hash": shared_probe_hash,
            "pair_support": {
                _pair_name(pair): support
                for pair, support in selection.pair_support.items()
            },
            "probe_config": asdict(scout_config.probe),
            "sampling_strategy": scout_config.probe.sampling_strategy,
            "fold_seed": scout_config.fold_seed,
            "fold_strategy": fold_strategy,
            "group_identifier": group_identifier,
            "group_by_sample": group_by_sample,
        }
    )
    payload = {
        "report_schema": "mint-agent.casf-probe-selection.v1",
        "task_id": task_config.get("task_id"),
        "representation_hash": representation.spec_hash,
        "modeling_scope": "train",
        "modeling_sample_count": len(modeling_ids),
        "test_sample_count_excluded": len(test_records),
        "modeling_sample_ids": list(modeling_ids),
        "probe_sample_ids": list(selection.sample_ids),
        "probe_base_size": selection.base_size,
        "probe_final_size": selection.final_size,
        "probe_config": asdict(scout_config.probe),
        "sampling_strategy": scout_config.probe.sampling_strategy,
        "sampling_seed": scout_config.probe.random_seed,
        "cv_folds": scout_config.cv_folds,
        "fold_seed": scout_config.fold_seed,
        "fold_strategy": fold_strategy,
        "group_identifier": group_identifier,
        "modeling_group_count": (
            len(set(group_by_sample.values())) or len(modeling_ids)
        ),
        "group_by_sample": group_by_sample,
        "fold_sample_counts": {
            str(fold): count for fold, count in sorted(fold_sample_counts.items())
        },
        "fold_group_counts": {
            str(fold): count for fold, count in sorted(fold_group_counts.items())
        },
        "group_leakage_check_passed": True,
        "modeling_fold_assignment": full_folds,
        "probe_fold_assignment": probe_folds,
        "probe_samples": [
            {
                "sample_id": sample_id,
                "label": target_by_id[sample_id],
                "structure_size": size_by_id[sample_id],
                "target_bin": selection.target_bins[sample_id],
                "size_bin": selection.size_bins[sample_id],
                "present_pairs": [list(pair) for pair in pair_presence[sample_id]],
            }
            for sample_id in selection.sample_ids
        ],
        "pair_support": {
            _pair_name(pair): support for pair, support in selection.pair_support.items()
        },
        "undercovered_pairs": [list(pair) for pair in selection.undercovered_pairs],
        "warnings": list(selection.warnings),
        "shared_probe_hash": shared_probe_hash,
        "frozen_sample_source": (
            str(args.frozen_sample_source.resolve())
            if args.frozen_sample_source is not None
            else None
        ),
        "selection_hash": selection_hash,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"selection_hash={selection_hash} modeling={len(modeling_ids)} "
        f"probe={selection.final_size} test_excluded={len(test_records)} output={args.output}"
    )
    return 0


def _validate_representation(spec: RepresentationSpec) -> None:
    spec.assert_frozen()
    if spec.mode != RepresentationMode.DATASET_ADAPTIVE:
        raise ValueError("Probe selection requires a dataset_adaptive RepresentationSpec")
    if spec.system_type != "protein_ligand":
        raise ValueError("Probe selection requires a protein_ligand RepresentationSpec")


def _select_probe(
    *,
    sample_ids: tuple[str, ...],
    targets: np.ndarray,
    structure_sizes: list[float],
    retained_pairs: tuple[tuple[str, str], ...],
    pair_presence: dict[str, tuple[tuple[str, str], ...]],
    scout_config: ScoutConfig,
) -> ProbeSelection:
    kwargs = {
        "sample_ids": sample_ids,
        "targets": targets,
        "structure_sizes": structure_sizes,
        "retained_pairs": retained_pairs,
        "pair_presence_by_sample": pair_presence,
        "config": scout_config.probe,
    }
    if scout_config.probe.sampling_strategy == "uniform_random":
        return select_uniform_random_probe_samples(**kwargs)
    return select_probe_samples(**kwargs)


def _validate_design_scope(path: Path, modeling_ids: tuple[str, ...]) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "representation_spec" not in raw:
        return
    if raw.get("report_schema") == "mint-agent.filtration-repair.v1":
        if raw.get("evidence_scope") != "design":
            raise ValueError("Filtration repair report must have evidence_scope=design")
        if str(raw.get("audit_evidence_scope") or "").lower() in {"test", "external_test"}:
            raise ValueError("Filtration repair report must not use test evidence")
        return
    if raw.get("modeling_scope") != "train":
        raise ValueError("Representation design report must use modeling_scope=train")
    designed_ids = tuple(str(sample_id) for sample_id in raw.get("sample_ids", ()))
    if designed_ids and designed_ids != modeling_ids:
        raise ValueError("Representation design sample IDs do not match the modeling pool")


def _pair_name(pair: tuple[str, str]) -> str:
    return f"protein:{pair[0]}|ligand:{pair[1]}"


def _load_frozen_source(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Frozen probe source must contain a JSON object")
    if payload.get("report_schema") == "mint-agent.controlled-probe-selection.v1":
        selected = payload.get("selected_probe_selection")
        if not isinstance(selected, str) or not selected:
            raise ValueError("Controlled probe selection does not identify its selected artifact")
        return _load_frozen_source(Path(selected))
    if payload.get("report_schema") != "mint-agent.casf-probe-selection.v1":
        raise ValueError("Frozen probe source must be a probe-selection artifact")
    return payload


def _selection_from_frozen_source(
    source: dict,
    *,
    modeling_ids: tuple[str, ...],
    targets: np.ndarray,
    structure_sizes: list[float],
    retained_pairs: tuple[tuple[str, str], ...],
    pair_presence: dict[str, tuple[tuple[str, str], ...]],
    min_pair_support: int,
) -> ProbeSelection:
    sample_ids = tuple(str(value) for value in source.get("probe_sample_ids", ()))
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Frozen probe sample IDs are missing or duplicated")
    if not set(sample_ids).issubset(modeling_ids):
        raise ValueError("Frozen probe includes samples outside the modeling pool")
    target_bins = quantile_bin_assignments(
        targets, modeling_ids, max_bins=int(source.get("probe_config", {}).get("target_quantile_bins", 10))
    )
    size_bins = quantile_bin_assignments(
        structure_sizes,
        modeling_ids,
        max_bins=int(source.get("probe_config", {}).get("size_quantile_bins", 5)),
    )
    support = {
        pair: sum(pair in pair_presence[sample_id] for sample_id in sample_ids)
        for pair in retained_pairs
    }
    undercovered = tuple(
        pair for pair in retained_pairs if support[pair] < min_pair_support
    )
    warnings = ()
    if undercovered:
        encoded = ", ".join(f"{left}|{right}" for left, right in undercovered[:10])
        warnings = (f"PROBE_PAIR_UNDERCOVERED: {encoded}",)
    return ProbeSelection(
        sample_ids=sample_ids,
        base_size=int(source.get("probe_base_size") or len(sample_ids)),
        final_size=len(sample_ids),
        target_bins={sample_id: target_bins[sample_id] for sample_id in sample_ids},
        size_bins={sample_id: size_bins[sample_id] for sample_id in sample_ids},
        pair_support=support,
        undercovered_pairs=undercovered,
        warnings=warnings,
    )


if __name__ == "__main__":
    raise SystemExit(main())
