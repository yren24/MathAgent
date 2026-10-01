from __future__ import annotations

import json

import numpy as np

from mint_scout.data.element_pairs import ElementPairSchema
from mint_scout.filtration import FiltrationProfile
from mint_scout.representation import RepresentationMode, RepresentationSpec
from mint_scout.run_casf_scout import main


def test_run_casf_scout_consumes_frozen_probe_and_manifests(tmp_path):
    train_ids = tuple(f"a{index:03d}" for index in range(8))
    index = tmp_path / "index"
    index.mkdir()
    (index / "2016_INDEX_refined.data").write_text(
        "\n".join(
            f"{sample_id} 2.0 2000 {4.0 + index_value:.2f} Kd=1nM"
            for index_value, sample_id in enumerate(train_ids)
        )
        + "\n",
        encoding="utf-8",
    )
    (index / "train_data_2016.txt").write_text(repr(list(train_ids)), encoding="utf-8")
    (index / "test_data_2016.txt").write_text("[]", encoding="utf-8")
    task = tmp_path / "task.yaml"
    task.write_text(
        "task_id: toy\n"
        "data_audit:\n"
        "  casf:\n"
        "    year: 2016\n"
        f"    index_root: {index}\n"
        f"    structures_root: {tmp_path / 'structures'}\n",
        encoding="utf-8",
    )
    scout = tmp_path / "scout.yaml"
    scout.write_text(
        "probe:\n"
        "  fraction: 1.0\n"
        "  min_samples: 2\n"
        "  max_samples: 8\n"
        "  target_quantile_bins: 2\n"
        "  size_quantile_bins: 2\n"
        "  min_pair_support: 1\n"
        "  augmentation_fraction_limit: 0.0\n"
        "  cv_folds: 2\n"
        "  random_seed: 3\n"
        "bootstrap:\n"
        "  enabled: true\n"
        "  replicates: 5\n"
        "  stratified: true\n"
        "  ci_level: 0.95\n"
        "  random_seed: 4\n"
        "model:\n"
        "  family: fixed_gbt\n"
        "  hpo: false\n"
        "  params:\n"
        "    config_id: toy\n"
        "    n_estimators: 3\n"
        "    max_depth: 2\n"
        "    min_samples_split: 2\n"
        "    learning_rate: 0.1\n"
        "    subsample: 1.0\n"
        "    max_features: sqrt\n"
        "    random_state: 5\n"
        "    n_runs: 1\n"
        "    normalize: StandardScaler\n",
        encoding="utf-8",
    )
    schema = ElementPairSchema(
        schema_id="toy",
        system_type="protein_ligand",
        left_role="protein",
        right_role="ligand",
        left_elements=("C",),
        right_elements=("N",),
        global_element_order=None,
        pair_generation_rule="toy",
        role_aware=True,
        exclude_self_pairs=False,
        deduplicate_unordered_pairs=False,
        pair_order=(("C", "N"),),
        expected_pair_count=1,
    )
    profile = FiltrationProfile("distance", 0.0, 1.0, 1.0, 2, "toy")
    representation = RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=schema,
        filtration_profiles={"PH": profile, "PL": profile},
    ).freeze()
    representation_path = tmp_path / "representation.json"
    representation.write(representation_path)

    probe_ids = train_ids[:6]
    labels = {sample_id: 4.0 + index_value for index_value, sample_id in enumerate(train_ids)}
    full_folds = {sample_id: index_value % 2 for index_value, sample_id in enumerate(train_ids)}
    probe_payload = {
        "representation_hash": representation.spec_hash,
        "selection_hash": "selection-test",
        "modeling_sample_ids": list(train_ids),
        "probe_sample_ids": list(probe_ids),
        "probe_base_size": 6,
        "probe_final_size": 6,
        "modeling_fold_assignment": full_folds,
        "probe_samples": [
            {
                "sample_id": sample_id,
                "label": labels[sample_id],
                "structure_size": 100 + index_value,
                "target_bin": index_value % 2,
                "size_bin": index_value % 2,
                "present_pairs": [["C", "N"]],
            }
            for index_value, sample_id in enumerate(probe_ids)
        ],
        "pair_support": {"protein:C|ligand:N": 6},
        "undercovered_pairs": [],
        "warnings": [],
    }
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps(probe_payload), encoding="utf-8")

    manifest_args = []
    for invariant, width in (("PH", 2), ("PL", 8)):
        feature_root = tmp_path / f"repr-{representation.spec_hash}" / invariant
        feature_root.mkdir(parents=True)
        manifest = tmp_path / f"{invariant}.jsonl"
        rows = []
        for index_value, sample_id in enumerate(probe_ids):
            feature_path = feature_root / f"{sample_id}.npy"
            np.save(feature_path, np.full((2, 1, width), index_value + 1.0))
            rows.append(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "invariant": invariant,
                        "status": "computed",
                        "output_path": str(feature_path),
                    }
                )
            )
        manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
        manifest_args.extend(["--feature-manifest", f"{invariant}={manifest}"])

    report = tmp_path / "scout.json"
    artifact = tmp_path / "execution.json"
    code = main(
        [
            "--task-config",
            str(task),
            "--scout-config",
            str(scout),
            "--representation-spec",
            str(representation_path),
            "--probe-selection",
            str(probe_path),
            "--output",
            str(report),
            "--execution-artifact",
            str(artifact),
        ]
        + manifest_args
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    execution = json.loads(artifact.read_text(encoding="utf-8"))
    assert code == 0
    assert len(payload["scout"]["frozen_priority_order"]) == 3
    assert set(payload["scout"]["invariant_oof_predictions"]) == {"PH", "PL"}
    assert execution["modeling_sample_ids"] == list(train_ids)
    assert execution["representation_hash"] == representation.spec_hash
