from pathlib import Path

from mint_scout.invariants.cache import FeatureCache


def test_feature_cache_reuses_existing_sample(tmp_path: Path):
    cache = FeatureCache(tmp_path)
    calls = []

    def compute_one(sample_id: str, out_path: Path) -> None:
        calls.append(sample_id)
        out_path.write_bytes(b"feature")

    first = cache.compute_missing(
        dataset_id="d",
        schema_id="s",
        invariant_name="PH",
        parameter_hash="h",
        sample_ids=["a"],
        compute_one=compute_one,
    )
    second = cache.compute_missing(
        dataset_id="d",
        schema_id="s",
        invariant_name="PH",
        parameter_hash="h",
        sample_ids=["a"],
        compute_one=compute_one,
    )
    assert [item.status for item in first] == ["computed"]
    assert [item.status for item in second] == ["cached"]
    assert calls == ["a"]


def test_feature_cache_can_key_by_representation_hash(tmp_path: Path):
    cache = FeatureCache(tmp_path)

    old_path = cache.representation_sample_path(
        dataset_id="casf",
        representation_hash="hash40",
        invariant_name="PL",
        parameter_hash="gbt",
        sample_id="10gs",
    )
    new_path = cache.representation_sample_path(
        dataset_id="casf",
        representation_hash="hash50",
        invariant_name="PL",
        parameter_hash="gbt",
        sample_id="10gs",
    )

    assert old_path != new_path
    assert "repr-hash40" in old_path.parts
    assert old_path.name == "10gs.npy"
