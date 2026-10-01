from pathlib import Path

from mint_scout.mof.legacy import LegacyMofPaths, legacy_feature_command, legacy_gbt_command, topology_for
from mint_scout.mof.standard_metrics import regression_metrics, summarize_predictions


def _paths(tmp_path: Path) -> LegacyMofPaths:
    return LegacyMofPaths(
        legacy_root=tmp_path / "legacy",
        cif_dir=tmp_path / "cifs",
        data_dir=tmp_path / "data",
        feature_dir=tmp_path / "features",
        result_dir=tmp_path / "results",
    )


def test_feature_command_delegates_to_the_audited_legacy_script(tmp_path):
    command = legacy_feature_command(
        paths=_paths(tmp_path),
        tool_name="MOF_HOMOLOGY",
        property_name="O2uptakemolkg",
        start=10,
        end=20,
    )

    assert command[1].endswith("mof_topology_features_forman120_final.py")
    assert command[command.index("--topology") + 1] == "homology"
    assert command[command.index("--start") + 1] == "10"
    assert command[command.index("--end") + 1] == "20"


def test_gbt_command_uses_the_same_topology_key_as_feature_generation(tmp_path):
    command = legacy_gbt_command(
        paths=_paths(tmp_path),
        tool_name="MOF_FORMAN",
        property_name="O2uptakemolkg",
        folds=5,
        repeat=3,
    )

    assert command[1].endswith("mof_topology_gbt.py")
    assert command[command.index("--topology") + 1] == "forman"
    assert command[command.index("--repeat") + 1] == "3"
    assert topology_for("mof_curvature") == "curvature"


def test_standard_metrics_are_residual_based_and_fold_scoped(tmp_path):
    prediction_csv = tmp_path / "predictions.csv"
    prediction_csv.write_text(
        "repeat,fold,MOFRefcodes,true,pred\n"
        "0,0,A,1.0,1.0\n"
        "0,0,B,3.0,2.0\n"
        "0,1,C,2.0,4.0\n"
        "0,1,D,4.0,4.0\n"
    )

    assert regression_metrics([1.0, 3.0], [1.0, 2.0]) == {"rmse": 2**-0.5, "r2_standard": 0.5}
    summary = summarize_predictions(prediction_csv)

    assert summary["aggregation"] == "macro_mean_over_legacy_test_folds"
    assert len(summary["fold_metrics"]) == 2
    assert summary["mean_rmse"] > 0
    assert summary["mean_r2_standard"] is not None
