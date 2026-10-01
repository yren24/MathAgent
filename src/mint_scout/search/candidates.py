from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CandidatePipeline:
    """One model-conditioned experiment candidate.

    A candidate is the unit of search. Feature evidence is only interpreted
    under the model family and training protocol used to produce it.
    """

    name: str
    features: tuple[str, ...]
    feature_schema: str
    model: str
    model_family: str
    input_contract: str
    normalize: str
    training_protocol: str
    fidelities: tuple[float, ...]
    enabled: bool = True
    notes: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, default_fidelities: Iterable[float]) -> "CandidatePipeline":
        model = data.get("model", {})
        if not isinstance(model, dict):
            raise ValueError("candidate.model must be a mapping")

        features = tuple(data.get("features", ()))
        if not features:
            raise ValueError("candidate.features must contain at least one feature")

        fidelities = tuple(float(value) for value in data.get("fidelities", default_fidelities))
        if not fidelities:
            raise ValueError("candidate.fidelities must contain at least one fidelity level")

        return cls(
            name=str(data["name"]),
            features=features,
            feature_schema=str(data["feature_schema"]),
            model=str(model["name"]),
            model_family=str(model["family"]),
            input_contract=str(data["input_contract"]),
            normalize=str(data.get("normalize", "none")),
            training_protocol=str(data["training_protocol"]),
            fidelities=fidelities,
            enabled=bool(data.get("enabled", True)),
            notes=str(data.get("notes", "")),
        )


def load_candidate_pipelines(config: dict[str, Any]) -> tuple[CandidatePipeline, ...]:
    search = config.get("search", {})
    if not isinstance(search, dict):
        raise ValueError("search config must be a mapping")

    raw_candidates = search.get("candidates", ())
    if not isinstance(raw_candidates, list):
        raise ValueError("search.candidates must be a list")

    default_fidelities = config.get("fidelity", {}).get("fractions", (1.0,))
    candidates = tuple(
        CandidatePipeline.from_mapping(candidate, default_fidelities=default_fidelities)
        for candidate in raw_candidates
    )
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("search.candidates names must be unique")
    return candidates


def enabled_candidate_pipelines(config: dict[str, Any]) -> tuple[CandidatePipeline, ...]:
    return tuple(candidate for candidate in load_candidate_pipelines(config) if candidate.enabled)


def assert_model_conditioned_candidates(candidates: Iterable[CandidatePipeline]) -> None:
    for candidate in candidates:
        if candidate.model_family == "neural_network" and candidate.model == "legacy_gbt":
            raise ValueError(f"{candidate.name} mixes neural_network family with legacy_gbt model")
        if candidate.input_contract not in {"fixed_vector", "graph", "sequence", "hybrid"}:
            raise ValueError(f"{candidate.name} has unsupported input_contract {candidate.input_contract!r}")
