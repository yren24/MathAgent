from mint_scout.mof.categories import LEGACY_CATEGORY_ORDER, legacy_category_schema, load_category_schema
from mint_scout.mof.category_audit import audit_category_assignments
from mint_scout.mof.manifest import MofManifestRecord


def test_legacy_category_schema_matches_the_audited_order():
    schema = legacy_category_schema()

    assert schema.category_order == LEGACY_CATEGORY_ORDER
    assert schema.category_for("Fe") == "C1"
    assert schema.category_for("c") == "C5"
    assert schema.category_for("H") == "C4"
    assert schema.category_for("Xe") is None
    assert schema.all_atoms_category == "Call"


def test_yaml_schema_round_trips_the_legacy_categories(tmp_path):
    path = tmp_path / "schema.yaml"
    path.write_text(
        "schema_id: test\n"
        "category_order: [A, Call]\n"
        "all_atoms_category: Call\n"
        "unknown_policy: report_only\n"
        "categories:\n"
        "  A: [C, O]\n",
        encoding="utf-8",
    )

    schema = load_category_schema(path)

    assert schema.category_for("O") == "A"
    assert schema.category_for("Zn") is None


def test_category_audit_reports_unassigned_elements_without_dropping_call(tmp_path):
    records = (
        MofManifestRecord("A", 1.0, tmp_path / "A.cif"),
        MofManifestRecord("B", 2.0, tmp_path / "B.cif"),
    )

    report = audit_category_assignments(
        records,
        {"A": ("C", "O", "Zn"), "B": ("C", "Xe")},
        legacy_category_schema(),
    )

    assert report["status"] == "WARN"
    assert report["category_sample_coverage"]["Call"] == 2
    assert report["category_sample_coverage"]["C5"] == 2
    assert report["unassigned_non_call_elements"] == {"Xe": 1}
    assert report["unassigned_sample_count"] == 1
