import csv
import zipfile

import pytest

from mint_scout.mof.manifest import MofManifestRecord, _read_xlsx_rows, write_manifest


def test_write_manifest_emits_portable_mof_csv(tmp_path):
    path = write_manifest(
        [
            MofManifestRecord("ABCDEF", 2.5, tmp_path / "ABCDEF.cif"),
            MofManifestRecord("GHIJKL", 3.5, tmp_path / "GHIJKL.cif", split="test"),
        ],
        tmp_path / "manifest.csv",
    )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"sample_id": "ABCDEF", "target": "2.5", "split": "train", "cif_path": str(tmp_path / "ABCDEF.cif")},
        {"sample_id": "GHIJKL", "target": "3.5", "split": "test", "cif_path": str(tmp_path / "GHIJKL.cif")},
    ]


def test_write_manifest_rejects_duplicate_mof_ids(tmp_path):
    record = MofManifestRecord("ABCDEF", 2.5, tmp_path / "ABCDEF.cif")
    with pytest.raises(ValueError, match="Duplicate MOF"):
        write_manifest([record, record], tmp_path / "manifest.csv")


def test_legacy_xlsx_reader_supports_inline_string_headers(tmp_path):
    workbook = tmp_path / "inline.xlsx"
    with zipfile.ZipFile(workbook, "w") as archive:
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """<?xml version=\"1.0\"?>
            <worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">
              <sheetData>
                <row r=\"1\"><c r=\"A1\" t=\"inlineStr\"><is><t>MOFRefcodes</t></is></c><c r=\"B1\" t=\"inlineStr\"><is><t>O2</t></is></c></row>
                <row r=\"2\"><c r=\"A2\" t=\"inlineStr\"><is><t>ABCDEF</t></is></c><c r=\"B2\"><v>1.25</v></c></row>
              </sheetData>
            </worksheet>""",
        )

    assert _read_xlsx_rows(workbook) == ({"MOFRefcodes": "ABCDEF", "O2": "1.25"},)
