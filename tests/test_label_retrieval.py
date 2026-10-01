from dataclasses import replace

import pytest

from mint_scout.labels import (
    LabelRetrievalPolicy,
    MatchStatus,
    RetrievedLabel,
    TargetLabelSpec,
    retrieve_and_validate_labels,
)
from mint_scout.schemas import SampleRecord


class MockProvider:
    def __init__(self, records_by_sample):
        self.records_by_sample = records_by_sample

    def retrieve(self, sample, target_spec):
        return self.records_by_sample.get(sample.sample_id, ())


def _sample():
    return SampleRecord("mol-1", identifiers={"pubchem_cid": "123"})


def _record(**changes):
    record = RetrievedLabel.grounded(
        sample_id="mol-1",
        identifier_type="pubchem_cid",
        identifier_value="123",
        target_name="LD50",
        value=2.0,
        unit="g/kg",
        endpoint_definition="oral rat LD50",
        source_name="ExampleDB",
        source_url_or_record_id="record:123",
        retrieval_timestamp="2026-09-07T00:00:00+00:00",
    )
    return replace(record, **changes)


def _run(records, *, target=None, policy=None):
    return retrieve_and_validate_labels(
        samples=[_sample()],
        target_spec=target or TargetLabelSpec("LD50", "g/kg", "oral rat LD50"),
        provider=MockProvider({"mol-1": records}),
        policy=policy
        or LabelRetrievalPolicy(min_labeled_samples=1),
    )


def test_exact_identifier_and_endpoint_is_accepted():
    result = _run([_record()])

    assert result.accepted_values == {"mol-1": 2.0}
    assert result.sufficient_for_modeling is True
    assert result.decisions[0].status == "accepted"


def test_wrong_endpoint_is_quarantined():
    result = _run([_record(endpoint_definition="dermal rabbit LD50")])

    assert not result.accepted_values
    assert result.decisions[0].status == "rejected"
    assert "endpoint mismatch" in result.decisions[0].reason


def test_configured_unit_conversion_preserves_raw_provenance():
    policy = LabelRetrievalPolicy(
        min_labeled_samples=1,
        unit_conversion_factors={("mg/kg", "g/kg"): 0.001},
    )
    result = _run([_record(value=2000.0, unit="mg/kg", raw_value=2000.0, raw_unit="mg/kg")], policy=policy)

    assert result.accepted_values["mol-1"] == 2.0
    normalized = result.decisions[0].records[0]
    assert normalized.unit == "g/kg"
    assert normalized.raw_value == 2000.0
    assert normalized.raw_unit == "mg/kg"


def test_conflicting_credible_values_are_not_silently_selected():
    result = _run([_record(value=2.0), _record(value=3.0, source_url_or_record_id="record:other")])

    assert not result.accepted_values
    assert result.decisions[0].status == "conflict"
    assert all(record.match_status == MatchStatus.CONFLICT for record in result.decisions[0].records)


def test_not_found_remains_unlabeled_and_triggers_sample_guard():
    result = _run([], policy=LabelRetrievalPolicy(min_labeled_samples=300))

    assert result.decisions[0].status == "not_found"
    assert result.sufficient_for_modeling is False
    assert result.warnings[0].startswith("INSUFFICIENT_LABELS")


def test_numeric_label_without_source_provenance_cannot_enter_schema():
    with pytest.raises(ValueError, match="grounded provenance"):
        _record(source_name="", source_url_or_record_id="")


def test_probable_match_is_rejected_unless_policy_explicitly_allows_it():
    record = _record(match_status=MatchStatus.PROBABLE)
    rejected = _run([record])
    accepted = _run(
        [record],
        policy=LabelRetrievalPolicy(
            min_labeled_samples=1,
            accepted_match_statuses=(MatchStatus.EXACT, MatchStatus.PROBABLE),
        ),
    )

    assert rejected.decisions[0].status == "rejected"
    assert accepted.decisions[0].status == "accepted"


def test_retrieval_policy_defaults_are_configurable_from_mapping():
    policy = LabelRetrievalPolicy.from_mapping(
        {
            "min_labeled_samples": 12,
            "allow_small_data_override": True,
            "accepted_match_statuses": ["exact", "probable"],
            "conflict_absolute_tolerance": 0.01,
            "unit_conversions": [{"from": "mg/kg", "to": "g/kg", "factor": 0.001}],
        }
    )

    assert policy.min_labeled_samples == 12
    assert policy.allow_small_data_override is True
    assert policy.accepted_match_statuses == (MatchStatus.EXACT, MatchStatus.PROBABLE)
    assert policy.unit_conversion_factors[("mg/kg", "g/kg")] == 0.001
