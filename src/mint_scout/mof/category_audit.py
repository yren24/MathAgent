"""Audit whether a MOF dataset is compatible with a category-specific schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .agent_workflow import MofAgentRecord, load_agent_manifest
from .categories import CategorySchema, legacy_category_schema, load_category_schema, normalize_element


CATEGORY_AUDIT_SCHEMA = "mint-agent.mof-category-audit.v1"


def audit_category_assignments(
    records: Iterable[MofAgentRecord],
    elements_by_sample: Mapping[str, Iterable[str]],
    schema: CategorySchema,
    *,
    read_failures: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Summarize schema coverage; unknown elements remain represented by Call."""
    record_list = tuple(records)
    failures = dict(read_failures or {})
    observed = Counter()
    category_coverage = Counter()
    unknown = Counter()
    unknown_samples: dict[str, list[str]] = {}
    parsed = 0

    for record in record_list:
        if record.sample_id in failures:
            continue
        raw_elements = elements_by_sample.get(record.sample_id)
        if raw_elements is None:
            failures[record.sample_id] = "no_elements_extracted"
            continue
        parsed += 1
        categories = set()
        missing = set()
        for raw in raw_elements:
            element = normalize_element(raw)
            observed[element] += 1
            category = schema.category_for(element)
            if category is None:
                unknown[element] += 1
                missing.add(element)
            else:
                categories.add(category)
        categories.add(schema.all_atoms_category)
        for category in categories:
            category_coverage[category] += 1
        if missing:
            unknown_samples[record.sample_id] = sorted(missing)

    status = "FAIL" if failures else "WARN" if unknown else "PASS"
    return {
        "report_schema": CATEGORY_AUDIT_SCHEMA,
        "status": status,
        "category_schema": schema.to_dict(),
        "sample_count": len(record_list),
        "parsed_cif_count": parsed,
        "cif_read_failure_count": len(failures),
        "cif_read_failures": failures,
        "observed_elements": dict(sorted(observed.items())),
        "category_sample_coverage": {
            category: category_coverage.get(category, 0) for category in schema.category_order
        },
        "unassigned_non_call_elements": dict(sorted(unknown.items())),
        "unassigned_sample_count": len(unknown_samples),
        "unassigned_samples": unknown_samples,
        "interpretation": (
            "Unassigned elements remain in the all-atoms Call channel under the frozen legacy schema; "
            "they are reported for scientific review, not silently discarded."
        ),
    }


def audit_manifest_categories(
    manifest_path: str | Path, schema: CategorySchema
) -> dict[str, Any]:
    records = load_agent_manifest(manifest_path)
    elements: dict[str, tuple[str, ...]] = {}
    failures: dict[str, str] = {}
    for record in records:
        try:
            elements[record.sample_id] = _elements_from_cif(record.cif_path)
        except Exception as exc:  # Dataset audit must report malformed input, not hide it.
            failures[record.sample_id] = f"{type(exc).__name__}: {exc}"
    report = audit_category_assignments(records, elements, schema, read_failures=failures)
    report["manifest_path"] = str(Path(manifest_path))
    return report


def _elements_from_cif(path: Path) -> tuple[str, ...]:
    try:
        from pymatgen.core import Structure
    except ImportError as exc:  # pragma: no cover - environment-specific message
        raise RuntimeError("MOF category audit requires the pymatgen module") from exc
    structure = Structure.from_file(path)
    return tuple(sorted({str(element.symbol) for element in structure.composition.elements}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a MOF manifest against a category-specific schema")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", type=Path)
    args = parser.parse_args()
    schema = load_category_schema(args.schema) if args.schema else legacy_category_schema()
    report = audit_manifest_categories(args.manifest, schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"status={report['status']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
