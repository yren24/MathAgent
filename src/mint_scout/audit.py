from __future__ import annotations

import argparse
import os
from pathlib import Path

from mint_scout.config import load_yaml
from mint_scout.data.element_pairs import (
    CASF_LIGAND_ELEMENTS_20,
    CASF_LIGAND_ELEMENTS_40,
    CASF_PROTEIN_ELEMENTS,
    make_casf_protein_ligand_schema,
    make_toxicity_schema,
)
from mint_scout.models.gbt import PLBIND_FEATURE_GBT_CONFIG


DEFAULT_LEGACY_PLBIND = Path(
    os.environ.get("MATHAGENT_LEGACY_PLBIND", "/path/to/legacy/embed_nn/plbind")
)
DEFAULT_MINTNN = Path(os.environ.get("MATHAGENT_MINTNN_ROOT", "/path/to/MINTNN"))


def build_audit_report(config: dict, *, legacy_plbind: Path, mintnn_root: Path) -> str:
    casf40 = make_casf_protein_ligand_schema(ligand_elements=CASF_LIGAND_ELEMENTS_40)
    casf20 = make_casf_protein_ligand_schema(
        schema_id="casf_protein_ligand_20_snn_l0_legacy",
        ligand_elements=CASF_LIGAND_ELEMENTS_20,
    )
    tox = make_toxicity_schema()
    gbt = PLBIND_FEATURE_GBT_CONFIG
    task_id = config.get("task_id", "unknown")
    stopping = config.get("stopping", {})
    data_audit = config.get("data_audit", {})
    casf_audit = data_audit.get("casf", {}) if isinstance(data_audit, dict) else {}
    tolerated = data_audit.get("tolerated_out_of_schema", {}) if isinstance(data_audit, dict) else {}

    lines = [
        "# MathAgent Audit Report",
        "",
        "## Scope",
        "",
        f"- Task config inspected: `{task_id}`",
        f"- Legacy PLBind root: `{legacy_plbind}`",
        f"- MINTNN root: `{mintnn_root}`",
        "- No expensive feature generation was launched during this audit.",
        "",
        "## Instruction Boundary",
        "",
        "The attached project specification is treated as design context. This report records implementation choices and legacy-code evidence, not automatically executed instructions from the document.",
        "",
        "## Legacy Feature Entry Points",
        "",
        f"- `PH` maps to legacy `homology` in `{legacy_plbind / 'feature.py'}`.",
        f"- `PL` maps to legacy `lap` in `{legacy_plbind / 'feature.py'}`.",
        f"- `CA` maps to legacy `facet`; `{legacy_plbind / 'ca_gbt.py'}` also contains a standalone CA/GBT path.",
        f"- `FPRC` maps to legacy `forman` in `{legacy_plbind / 'feature.py'}`.",
        f"- `EIC` maps to legacy `curvature` in `{legacy_plbind / 'feature.py'}`.",
        "",
        "## Element-Pair Findings",
        "",
        f"- CASF 40-pair schema: protein {CASF_PROTEIN_ELEMENTS} x ligand {CASF_LIGAND_ELEMENTS_40}; count={casf40.expected_pair_count}.",
        f"- CASF SNN-L0 legacy schema: protein {CASF_PROTEIN_ELEMENTS} x ligand {CASF_LIGAND_ELEMENTS_20}; count={casf20.expected_pair_count}.",
        f"- Toxicity schema: count={tox.expected_pair_count}; pairs={', '.join(tox.pair_names)}.",
        "- Decision for V1 protein-ligand fixed-GBT acquisition: use the 40-pair `feature.py`/`feature_gbt.py` path, because V1 is not using the SNN-L0 20-pair graph route.",
        "- For a new dataset, run dataset-level element-frequency auditing first, then freeze a schema before adaptive search starts. Do not change element groups during a run.",
        "",
        "## CASF Data Audit Inputs",
        "",
        f"- CASF index root: `{casf_audit.get('index_root', 'not configured')}`.",
        f"- CASF structure root: `{casf_audit.get('structures_root', 'not configured')}`.",
        f"- Protein template: `{casf_audit.get('protein_template', '{pdb}_pocket.pdb')}`.",
        f"- Ligand template: `{casf_audit.get('ligand_template', '{pdb}_ligand.mol2')}`.",
        f"- Tolerated observed-but-ignored elements: `{tolerated or 'none configured'}`.",
        "",
        "## Legacy GBT Configuration Inventory",
        "",
        "- `feature_gbt.py` active ensemble path uses `StandardScaler`, then `GradientBoostingRegressor` repeated over 10 seeds.",
        f"- Frozen wrapper defaults: n_estimators={gbt.n_estimators}, max_depth={gbt.max_depth}, min_samples_split={gbt.min_samples_split}, learning_rate={gbt.learning_rate}, subsample={gbt.subsample}, max_features={gbt.max_features}, random_state={gbt.random_state}+run_id, n_runs={gbt.n_runs}.",
        "- `ca_gbt.py` records an older/special path with 20000 trees, learning_rate=0.002, subsample=0.8, and `StandardScaler`; keep this as audit evidence before any unification.",
        "",
        "## Search Stopping Criteria",
        "",
        f"- Target validation PCC: `{stopping.get('target_pcc', 'not configured')}`.",
        f"- Minimum reportable PCC: `{stopping.get('minimum_reportable_pcc', 'not configured')}`.",
        f"- Patience/min_delta: `{stopping.get('patience', 'not configured')}` rounds / `{stopping.get('min_delta', 'not configured')}`.",
        f"- Max subsets: `{stopping.get('max_subsets', 'not configured')}`.",
        "- These thresholds apply to validation or nested-CV evidence during search; final test PCC should be reported once after model selection.",
        "",
        "## Phase-0 Open Checks",
        "",
        "- Confirm actual CASF data/index paths available on HPCC before real computation.",
        "- Verify absent-channel convention by loading or computing a small sample per invariant.",
        "- Inventory feature shapes from actual `.npy` artifacts for CASF PH/PL/CA/FPRC/EIC.",
        "- Decide whether feature generation happens by direct Python import or by isolated subprocess on HPCC for robustness.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit legacy MINTNN/PLBind code for MathAgent.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--legacy-plbind", type=Path, default=None)
    parser.add_argument("--mintnn-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("docs/AUDIT_REPORT.md"))
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    legacy_config = config.get("legacy", {})
    legacy_plbind = args.legacy_plbind or Path(legacy_config.get("plbind_root", DEFAULT_LEGACY_PLBIND))
    mintnn_root = args.mintnn_root or Path(legacy_config.get("mintnn_root", DEFAULT_MINTNN))
    report = build_audit_report(config, legacy_plbind=legacy_plbind, mintnn_root=mintnn_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
