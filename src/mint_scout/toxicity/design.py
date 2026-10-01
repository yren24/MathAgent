from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mint_scout.data.element_pairs import (
    AdaptivePairBuild,
    SupportThresholds,
    build_dataset_adaptive_schema,
    make_toxicity_schema,
)
from mint_scout.data.geometry import AtomCloud, load_atom_cloud
from mint_scout.data.manifest_io import load_dataset_manifest
from mint_scout.filtration import FiltrationProfile
from mint_scout.invariants.manifest import stable_hash
from mint_scout.representation import (
    RepresentationMode,
    RepresentationSpec,
    make_legacy_toxicity_representation_spec,
)
from mint_scout.schemas import SampleSplit
from mint_scout.toxicity.legacy import COVALENT_RADII, TOXICITY_ELEMENT_UNIVERSE


DESIGN_SCHEMA = "mint-agent.toxicity-representation-design.v1"
ALLOWED_DISTANCE_QUANTILES = (0.90, 0.95, 0.975)
ALLOWED_MARGIN_FACTORS = (1.0, 1.1, 1.2)
ALLOWED_SUPPORT_FRACTIONS = (0.001, 0.005, 0.01)
ALLOWED_SUPPORT_SAMPLES = (5, 10, 25)
ALLOWED_PAIR_CAPS = (30, 45, 50)


@dataclass(frozen=True)
class ToxicityAdaptiveProposal:
    proposal_id: str = "deterministic-q95"
    source: str = "deterministic"
    include_hydrogen: bool = True
    min_element_support_fraction: float = 0.005
    min_pair_support_fraction: float = 0.005
    min_support_samples: int = 10
    max_pair_channels: int = 50
    local_distance_quantile: float = 0.95
    dataset_distance_quantile: float = 0.95
    margin_factor: float = 1.10
    fixed_point_count: int = 50
    bond_delta: float = 0.45

    def __post_init__(self) -> None:
        if not self.proposal_id or "/" in self.proposal_id or "\\" in self.proposal_id:
            raise ValueError("proposal_id must be a non-empty safe identifier")
        if self.source not in {"deterministic", "llm"}:
            raise ValueError("proposal source must be deterministic or llm")
        if self.min_element_support_fraction not in ALLOWED_SUPPORT_FRACTIONS:
            raise ValueError("unsupported min_element_support_fraction")
        if self.min_pair_support_fraction not in ALLOWED_SUPPORT_FRACTIONS:
            raise ValueError("unsupported min_pair_support_fraction")
        if self.min_support_samples not in ALLOWED_SUPPORT_SAMPLES:
            raise ValueError("unsupported min_support_samples")
        if self.max_pair_channels not in ALLOWED_PAIR_CAPS:
            raise ValueError("unsupported max_pair_channels")
        if self.local_distance_quantile not in ALLOWED_DISTANCE_QUANTILES:
            raise ValueError("unsupported local_distance_quantile")
        if self.dataset_distance_quantile not in ALLOWED_DISTANCE_QUANTILES:
            raise ValueError("unsupported dataset_distance_quantile")
        if self.margin_factor not in ALLOWED_MARGIN_FACTORS:
            raise ValueError("unsupported margin_factor")
        if self.fixed_point_count != 50:
            raise ValueError("toxicity adaptive candidates must keep exactly 50 distance points")
        if self.bond_delta != 0.45:
            raise ValueError("Phase 1 toxicity candidates must retain audited bond_delta=0.45")

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, source: str = "llm"
    ) -> "ToxicityAdaptiveProposal":
        allowed = {
            "proposal_id",
            "include_hydrogen",
            "min_element_support_fraction",
            "min_pair_support_fraction",
            "min_support_samples",
            "max_pair_channels",
            "local_distance_quantile",
            "dataset_distance_quantile",
            "margin_factor",
            "fixed_point_count",
            "bond_delta",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown toxicity adaptive proposal fields: {unknown}")
        return cls(source=source, **dict(raw))

    @property
    def support_thresholds(self) -> SupportThresholds:
        return SupportThresholds(
            min_element_support_fraction=self.min_element_support_fraction,
            min_element_support_samples=self.min_support_samples,
            min_pair_support_fraction=self.min_pair_support_fraction,
            min_pair_support_samples=self.min_support_samples,
            max_element_pair_channels=self.max_pair_channels,
            small_molecule_self_pairs=False,
        )


@dataclass(frozen=True)
class ToxicityCandidate:
    candidate_id: str
    source: str
    ablation: str
    proposal_id: str | None
    representation_spec: RepresentationSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "ablation": self.ablation,
            "proposal_id": self.proposal_id,
            "representation_hash": self.representation_spec.spec_hash,
            "schema_id": self.representation_spec.schema_id,
            "pair_count": len(self.representation_spec.pair_order),
            "pair_names": list(self.representation_spec.pair_names),
            "filtration_profiles": {
                name: profile.to_dict()
                for name, profile in self.representation_spec.filtration_profiles.items()
            },
        }


@dataclass(frozen=True)
class PairGeometrySummary:
    sample_count: int
    local_summary_count: int
    local_distance_quantile: float
    dataset_distance_quantile: float
    raw_distance_quantile: float
    margin_factor: float
    max_filtration_angstrom: float
    pair_observation_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pair_observation_counts"] = dict(sorted(self.pair_observation_counts.items()))
        return payload


@dataclass(frozen=True)
class ToxicityDesignResult:
    training_sample_count: int
    observed_element_presence: Mapping[str, int]
    out_of_schema_element_presence: Mapping[str, int]
    candidates: tuple[ToxicityCandidate, ...]
    proposal_audits: tuple[Mapping[str, Any], ...]
    training_fingerprint: str
    primary_metric: str = "PCC2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_schema": DESIGN_SCHEMA,
            "evidence_scope": "train_only",
            "test_structures_used": False,
            "test_labels_used": False,
            "training_sample_count": self.training_sample_count,
            "training_fingerprint": self.training_fingerprint,
            "observed_element_presence": dict(sorted(self.observed_element_presence.items())),
            "out_of_schema_element_presence": dict(
                sorted(self.out_of_schema_element_presence.items())
            ),
            "scientific_controls": {
                "legacy_baseline_always_retained": True,
                "llm_can_add_but_not_remove_candidates": True,
                "llm_used_for_numeric_ranking": False,
                "unordered_pairs": True,
                "self_pairs": False,
                "fixed_distance_point_count": 50,
                "primary_metric": self.primary_metric,
            },
            "proposal_audits": [dict(audit) for audit in self.proposal_audits],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def design_toxicity_candidates(
    *,
    training_paths: Mapping[str, str | Path],
    deterministic_proposal: ToxicityAdaptiveProposal = ToxicityAdaptiveProposal(),
    llm_proposals: Sequence[ToxicityAdaptiveProposal] = (),
    primary_metric: str = "PCC2",
) -> ToxicityDesignResult:
    if primary_metric not in {"PCC2", "PCC", "R2"}:
        raise ValueError("toxicity design primary_metric must be PCC2, PCC, or R2")
    if not training_paths:
        raise ValueError("toxicity representation design requires training structures")
    if deterministic_proposal.source != "deterministic":
        raise ValueError("deterministic_proposal must have source=deterministic")
    if len(llm_proposals) > 2:
        raise ValueError("at most two LLM toxicity representation proposals are allowed")
    if any(proposal.source != "llm" for proposal in llm_proposals):
        raise ValueError("llm_proposals must have source=llm")
    proposal_ids = [deterministic_proposal.proposal_id] + [
        proposal.proposal_id for proposal in llm_proposals
    ]
    if len(set(proposal_ids)) != len(proposal_ids):
        raise ValueError("toxicity adaptive proposal ids must be unique")

    clouds = {
        sample_id: load_atom_cloud(path)
        for sample_id, path in sorted(training_paths.items())
    }
    element_presence = _element_presence(clouds.values())
    allowed = frozenset(TOXICITY_ELEMENT_UNIVERSE)
    out_of_schema = {
        element: count for element, count in element_presence.items() if element not in allowed
    }
    training_fingerprint = stable_hash(
        [
            {"sample_id": sample_id, "content_hash": cloud.content_hash}
            for sample_id, cloud in sorted(clouds.items())
        ]
    )
    legacy_spec = make_legacy_toxicity_representation_spec()
    candidates: list[ToxicityCandidate] = [
        ToxicityCandidate(
            candidate_id="legacy-30-pairs-legacy-grid",
            source="deterministic_baseline",
            ablation="legacy_baseline",
            proposal_id=None,
            representation_spec=legacy_spec,
        )
    ]
    audits: list[Mapping[str, Any]] = []
    seen_hashes = {legacy_spec.spec_hash}
    for proposal in (deterministic_proposal, *llm_proposals):
        pair_build = _adaptive_pairs(clouds, proposal)
        legacy_geometry = _geometry_summary(
            clouds,
            make_toxicity_schema().pair_order,
            proposal,
        )
        adaptive_geometry = _geometry_summary(
            clouds,
            pair_build.schema.pair_order,
            proposal,
        )
        adaptive_profiles = _adaptive_profiles(adaptive_geometry, proposal)
        legacy_pair_adaptive_profiles = _adaptive_profiles(legacy_geometry, proposal)
        proposal_specs = (
            (
                "pairs-only",
                pair_build.schema,
                dict(legacy_spec.filtration_profiles),
            ),
            (
                "grid-only",
                make_toxicity_schema(),
                legacy_pair_adaptive_profiles,
            ),
            (
                "pairs-and-grid",
                pair_build.schema,
                adaptive_profiles,
            ),
        )
        generated_ids: list[str] = []
        for ablation, schema, profiles in proposal_specs:
            spec = RepresentationSpec.from_schema(
                mode=RepresentationMode.DATASET_ADAPTIVE,
                schema=schema,
                filtration_profiles=profiles,
                parameters={
                    "system_type": "small_molecule",
                    "task": "toxicity_regression",
                    "proposal": asdict(proposal),
                    "ablation": ablation,
                    "training_fingerprint": training_fingerprint,
                    "legacy_math_reused": True,
                    "pair_support": [asdict(record) for record in pair_build.pair_support],
                    "out_of_schema_element_presence": out_of_schema,
                    "geometry": (
                        legacy_geometry.to_dict()
                        if ablation == "grid-only"
                        else adaptive_geometry.to_dict()
                    ),
                },
            ).freeze()
            if spec.spec_hash in seen_hashes:
                continue
            seen_hashes.add(spec.spec_hash)
            candidate_id = f"{proposal.proposal_id}-{ablation}"
            candidates.append(
                ToxicityCandidate(
                    candidate_id=candidate_id,
                    source=proposal.source,
                    ablation=ablation,
                    proposal_id=proposal.proposal_id,
                    representation_spec=spec,
                )
            )
            generated_ids.append(candidate_id)
        audits.append(
            {
                "proposal": asdict(proposal),
                "element_support": [asdict(record) for record in pair_build.element_support],
                "pair_support": [asdict(record) for record in pair_build.pair_support],
                "selected_pair_count": len(pair_build.schema.pair_order),
                "selected_pairs": [list(pair) for pair in pair_build.schema.pair_order],
                "pair_cap_applied": pair_build.cap_applied,
                "warnings": list(pair_build.warnings),
                "legacy_pair_geometry": legacy_geometry.to_dict(),
                "adaptive_pair_geometry": adaptive_geometry.to_dict(),
                "candidate_ids": generated_ids,
            }
        )
    return ToxicityDesignResult(
        training_sample_count=len(clouds),
        observed_element_presence=element_presence,
        out_of_schema_element_presence=out_of_schema,
        candidates=tuple(candidates),
        proposal_audits=tuple(audits),
        training_fingerprint=training_fingerprint,
        primary_metric=primary_metric,
    )


def write_design(result: ToxicityDesignResult, output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    candidate_dir = destination / "representations"
    candidate_dir.mkdir(exist_ok=True)
    for candidate in result.candidates:
        candidate.representation_spec.write(candidate_dir / f"{candidate.candidate_id}.json")
    report_path = destination / "toxicity_representation_design.json"
    report_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def _adaptive_pairs(
    clouds: Mapping[str, AtomCloud], proposal: ToxicityAdaptiveProposal
) -> AdaptivePairBuild:
    allowed = set(TOXICITY_ELEMENT_UNIVERSE)
    if not proposal.include_hydrogen:
        allowed.remove("H")
    elements_by_sample = {
        sample_id: tuple(element for element in cloud.elements if element in allowed)
        for sample_id, cloud in clouds.items()
    }
    build = build_dataset_adaptive_schema(
        system_type="small_molecule",
        elements_by_role={"molecule": elements_by_sample},
        thresholds=proposal.support_thresholds,
        schema_prefix=f"toxicity_{proposal.proposal_id}",
    )
    if not build.schema.pair_order:
        raise ValueError(
            f"No adaptive element pairs survived proposal {proposal.proposal_id!r}"
        )
    return build


def _geometry_summary(
    clouds: Mapping[str, AtomCloud],
    pairs: Sequence[tuple[str, str]],
    proposal: ToxicityAdaptiveProposal,
) -> PairGeometrySummary:
    local_summaries: list[float] = []
    observation_counts = {f"{left}-{right}": 0 for left, right in pairs}
    for cloud in clouds.values():
        molecule_distances: list[np.ndarray] = []
        for left, right in pairs:
            left_points = cloud.coordinates_for(left)
            right_points = cloud.coordinates_for(right)
            if len(left_points) == 0 or len(right_points) == 0:
                continue
            distances = np.linalg.norm(
                left_points[:, None, :] - right_points[None, :, :], axis=2
            ).reshape(-1)
            cutoff = (
                COVALENT_RADII[left]
                + COVALENT_RADII[right]
                + proposal.bond_delta
            )
            distances = distances[distances > cutoff]
            if distances.size == 0:
                continue
            molecule_distances.append(distances)
            observation_counts[f"{left}-{right}"] += 1
        if molecule_distances:
            local_summaries.append(
                float(
                    np.quantile(
                        np.concatenate(molecule_distances),
                        proposal.local_distance_quantile,
                    )
                )
            )
    if not local_summaries:
        raise ValueError("No non-covalent distances survived toxicity geometry audit")
    raw_quantile = float(
        np.quantile(local_summaries, proposal.dataset_distance_quantile)
    )
    rmax = raw_quantile * proposal.margin_factor
    if not math.isfinite(rmax) or rmax <= 0.0:
        raise ValueError("Adaptive toxicity filtration maximum must be finite and positive")
    return PairGeometrySummary(
        sample_count=len(clouds),
        local_summary_count=len(local_summaries),
        local_distance_quantile=proposal.local_distance_quantile,
        dataset_distance_quantile=proposal.dataset_distance_quantile,
        raw_distance_quantile=raw_quantile,
        margin_factor=proposal.margin_factor,
        max_filtration_angstrom=rmax,
        pair_observation_counts=observation_counts,
    )


def _adaptive_profiles(
    geometry: PairGeometrySummary,
    proposal: ToxicityAdaptiveProposal,
) -> dict[str, FiltrationProfile]:
    point_count = proposal.fixed_point_count
    step = geometry.max_filtration_angstrom / (point_count - 1)
    distance = FiltrationProfile(
        "distance",
        0.0,
        geometry.max_filtration_angstrom,
        step,
        point_count,
        f"train_only_q{proposal.dataset_distance_quantile:g}_margin{proposal.margin_factor:g}",
    )
    profiles = {name: distance for name in ("PH", "PL", "CA", "FPRC")}
    profiles["EIC"] = FiltrationProfile(
        "tau", 0.2, 5.0, 0.2, 25, "audited_legacy_tau"
    )
    return profiles


def _element_presence(clouds: Iterable[AtomCloud]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cloud in clouds:
        for element in set(cloud.elements):
            counts[element] = counts.get(element, 0) + 1
    return counts


def _load_llm_proposals(path: Path | None) -> tuple[ToxicityAdaptiveProposal, ...]:
    if path is None:
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("toxicity LLM proposals must be a JSON list")
    malformed = [index for index, item in enumerate(payload) if not isinstance(item, Mapping)]
    if malformed:
        raise ValueError(
            "each toxicity LLM proposal must be a JSON object; "
            f"invalid list positions: {malformed}"
        )
    return tuple(
        ToxicityAdaptiveProposal.from_mapping(item, source="llm")
        for item in payload
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Design bounded train-only toxicity representation candidates."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="LD50")
    parser.add_argument("--primary-metric", default="PCC2")
    parser.add_argument("--llm-proposals", type=Path, default=None)
    args = parser.parse_args(argv)
    manifest = load_dataset_manifest(
        args.manifest,
        dataset_id=args.dataset_id,
        system_type="small_molecule",
        columns={"identifiers": {"smiles": "smiles", "source_filename": "source_filename"}},
        label_name=args.dataset_id,
    )
    training_paths = {
        sample.sample_id: sample.role_paths["molecule"]
        for sample in manifest.samples_in_split(SampleSplit.TRAIN)
    }
    result = design_toxicity_candidates(
        training_paths=training_paths,
        llm_proposals=_load_llm_proposals(args.llm_proposals),
        primary_metric=args.primary_metric,
    )
    report_path = write_design(result, args.output_dir)
    print(
        f"report={report_path} train={result.training_sample_count} "
        f"candidates={len(result.candidates)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
