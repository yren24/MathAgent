from __future__ import annotations

import json

import pytest

from mint_scout.combine_scout_oof import _estimate_full_acquisition_costs, main
from mint_scout.config import load_yaml
from mint_scout.data.element_pairs import ElementPairSchema
from mint_scout.filtration import FiltrationProfile
from mint_scout.representation import RepresentationMode, RepresentationSpec
from mint_scout.scout.pipeline import ScoutConfig


def _fixture(tmp_path):
    config_path = tmp_path / "scout.yaml"
    config_path.write_text(
        "probe:\n"
        "  fraction: 1.0\n"
        "  min_samples: 2\n"
        "  max_samples: 6\n"
        "  target_quantile_bins: 2\n"
        "  size_quantile_bins: 2\n"
        "  min_pair_support: 0\n"
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
    representation = RepresentationSpec.from_schema(
        mode=RepresentationMode.DATASET_ADAPTIVE,
        schema=schema,
        filtration_profiles={"PL": FiltrationProfile("distance", 0.0, 1.0, 1.0, 2, "toy")},
    ).freeze()
    representation_path = tmp_path / "representation.json"
    representation.write(representation_path)
    ids = tuple(f"s{index}" for index in range(6))
    folds = {sample_id: index % 2 for index, sample_id in enumerate(ids)}
    probe = {
        "representation_hash": representation.spec_hash,
        "selection_hash": "selection",
        "modeling_sample_ids": list(ids),
        "probe_sample_ids": list(ids),
        "modeling_fold_assignment": folds,
        "probe_fold_assignment": folds,
        "probe_samples": [
            {"sample_id": sample_id, "label": float(index), "target_bin": index % 2}
            for index, sample_id in enumerate(ids)
        ],
    }
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps(probe), encoding="utf-8")
    gbt_hash = ScoutConfig.from_mapping(load_yaml(config_path)).gbt.parameter_hash
    reports = []
    for invariant, values in (("PH", [0, 1, 2, 3, 4, 5]), ("PL", [0, 1, 1, 3, 5, 5])):
        report = tmp_path / f"{invariant}.json"
        report.write_text(
            json.dumps(
                {
                    "representation_hash": representation.spec_hash,
                    "selection_hash": "selection",
                    "scout": {
                        "gbt_parameter_hash": gbt_hash,
                        "probe_sample_ids": list(ids),
                        "invariant_oof_predictions": {
                            invariant: dict(zip(ids, values))
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        reports.append(report)
    return config_path, representation_path, probe_path, reports


def test_combine_scout_oof_ranks_all_available_subsets(tmp_path):
    config, representation, probe, reports = _fixture(tmp_path)
    output = tmp_path / "combined.json"
    artifact = tmp_path / "execution.json"
    args = [
        "--scout-config",
        str(config),
        "--representation-spec",
        str(representation),
        "--probe-selection",
        str(probe),
        "--output",
        str(output),
        "--execution-artifact",
        str(artifact),
    ]
    for report in reports:
        args.extend(["--scout-report", str(report)])

    assert main(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    execution = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["invariants"] == ["PH", "PL"]
    assert len(payload["consensus_metrics"]) == 3
    assert len(payload["frozen_priority_order"]) == 3
    assert execution["frozen_priority_order"] == payload["frozen_priority_order"]
    assert payload["cost_profile"]["PH"]["source"] == "unit_fallback_missing_manifest"
    assert payload["priority_policy"]["ranking_method"] == "hierarchical_empirical"
    assert payload["priority_policy"]["candidate_count"] == 3
    assert payload["priority_policy"]["llm_prior_used"] is False
    assert payload["priority_policy"]["diagnostic_composite_only"] is True
    assert payload["priority_policy"]["run_aggregation"] == "mean_predictions_then_compute_metric"
    assert all(
        candidate["final_score"] is not None
        for candidate in payload["consensus_metrics"]
    )


def test_combine_scout_oof_allows_explicit_gbt_config_hash(tmp_path):
    config, representation, probe, reports = _fixture(tmp_path)
    actual_hash = "explicit-gbt-config-hash"
    for report in reports:
        payload = json.loads(report.read_text(encoding="utf-8"))
        payload["scout"]["gbt_parameter_hash"] = actual_hash
        report.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "combined.json"
    artifact = tmp_path / "execution.json"
    args = [
        "--scout-config",
        str(config),
        "--representation-spec",
        str(representation),
        "--probe-selection",
        str(probe),
        "--output",
        str(output),
        "--execution-artifact",
        str(artifact),
    ]
    for report in reports:
        args.extend(["--scout-report", str(report)])

    assert main(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    execution = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["gbt_parameter_hash"] == actual_hash
    assert execution["gbt_parameter_hash"] == actual_hash


def test_combine_scout_oof_rejects_inconsistent_source_gbt_hashes(tmp_path):
    config, representation, probe, reports = _fixture(tmp_path)
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    payload["scout"]["gbt_parameter_hash"] = "wrong"
    reports[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="one shared GBT configuration"):
        main(
            [
                "--scout-config",
                str(config),
                "--representation-spec",
                str(representation),
                "--probe-selection",
                str(probe),
                "--scout-report",
                str(reports[0]),
                "--scout-report",
                str(reports[1]),
                "--output",
                str(tmp_path / "combined.json"),
                "--execution-artifact",
                str(tmp_path / "execution.json"),
            ]
        )


def test_feature_manifest_timings_estimate_full_acquisition_cost(tmp_path):
    manifest = tmp_path / "PL.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"status": "computed", "wall_seconds": 1.0},
                {"status": "computed", "wall_seconds": 3.0},
                {"status": "cached", "wall_seconds": 0.01},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    profile = _estimate_full_acquisition_costs(
        feature_manifests={"PL": manifest},
        invariant_names=("PL",),
        modeling_sample_count=10,
    )

    assert profile["PL"]["source"] == "computed_manifest_median"
    assert profile["PL"]["observed_sample_count"] == 2
    assert profile["PL"]["per_sample_wall_seconds"] == 2.0
    assert profile["PL"]["estimated_full_wall_seconds"] == 20.0
