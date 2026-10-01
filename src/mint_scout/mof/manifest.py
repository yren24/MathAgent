"""Create portable CSV manifests from legacy MOF label workbooks."""

from __future__ import annotations

import csv
import math
import zipfile
import xml.etree.ElementTree as element_tree
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MofManifestRecord:
    sample_id: str
    target: float
    cif_path: Path
    split: str = "train"


def load_legacy_property_records(
    *, property_name: str, data_dir: str | Path, cif_dir: str | Path
) -> tuple[MofManifestRecord, ...]:
    rows = _read_xlsx_rows(Path(data_dir) / f"{property_name}.xlsx")
    records: list[MofManifestRecord] = []
    for row in rows:
        mof_id = str(row.get("MOFRefcodes", "")).strip()
        value = row.get(property_name)
        if not mof_id or value in {None, ""}:
            continue
        try:
            target = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(target):
            continue
        records.append(
            MofManifestRecord(
                sample_id=mof_id,
                target=target,
                cif_path=Path(cif_dir) / f"{mof_id}.cif",
            )
        )
    if not records:
        raise ValueError(f"No finite labels found for {property_name!r}")
    return tuple(records)


def write_manifest(records: Iterable[MofManifestRecord], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "target", "split", "cif_path"))
        writer.writeheader()
        for record in records:
            if record.sample_id in seen:
                raise ValueError(f"Duplicate MOF sample_id: {record.sample_id}")
            seen.add(record.sample_id)
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "target": f"{record.target:.17g}",
                    "split": record.split,
                    "cif_path": str(record.cif_path),
                }
            )
            count += 1
    if count == 0:
        raise ValueError("Cannot write an empty MOF manifest")
    return destination


def _read_xlsx_rows(path: Path) -> tuple[dict[str, str], ...]:
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive, namespace)
        sheet = element_tree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    raw_rows: list[dict[str, str]] = []
    for row in sheet.findall(".//a:sheetData/a:row", namespace):
        values: dict[str, str] = {}
        for cell in row.findall("a:c", namespace):
            reference = cell.get("r", "")
            column = "".join(character for character in reference if character.isalpha())
            values[column] = _cell_value(cell, shared, namespace)
        raw_rows.append(values)
    if not raw_rows:
        return ()
    headers = raw_rows[0]
    return tuple(
        {name: row.get(column, "") for column, name in headers.items() if name}
        for row in raw_rows[1:]
    )


def _cell_value(
    cell: element_tree.Element,
    shared: list[str],
    namespace: dict[str, str],
) -> str:
    if cell.get("t") == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//a:t", namespace))
    value_node = cell.find("a:v", namespace)
    if value_node is None:
        return ""
    value = value_node.text or ""
    return shared[int(value)] if cell.get("t") == "s" else value


def _shared_strings(archive: zipfile.ZipFile, namespace: dict[str, str]) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = element_tree.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.text or "" for node in item.findall(".//a:t", namespace)) for item in root.findall("a:si", namespace)]
