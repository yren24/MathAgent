from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = ("filename", "smiles", "label")


@dataclass(frozen=True)
class LD50ManifestRecord:
    sample_id: str
    target: float
    split: str
    molecule_path: Path
    smiles: str
    source_filename: str

    def to_row(self) -> dict[str, str | float]:
        return {
            "sample_id": self.sample_id,
            "target": self.target,
            "split": self.split,
            "molecule_path": str(self.molecule_path),
            "smiles": self.smiles,
            "source_filename": self.source_filename,
        }


def normalized_cas_key(name: str) -> tuple[str, str, str] | None:
    """Normalize CAS-like names, including spreadsheet-formatted dates."""
    value = name.strip()
    slash_match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", value)
    if slash_match:
        month, day, year = slash_match.groups()
        return str(int(year)), str(int(month)), str(int(day))
    dash_match = re.fullmatch(r"(\d+)-(\d+)-(\d+)", value)
    if dash_match:
        first, second, third = dash_match.groups()
        return str(int(first)), str(int(second)), str(int(third))
    return None


def build_ld50_manifest(dataset_dir: str | Path) -> tuple[LD50ManifestRecord, ...]:
    root = Path(dataset_dir).expanduser().resolve()
    records: list[LD50ManifestRecord] = []
    seen_ids: set[str] = set()
    for split in ("train", "test"):
        csv_path = root / f"LD50_{split}.csv"
        molecule_dir = root / f"LD50_{split}_x"
        records.extend(
            _read_split(
                csv_path=csv_path,
                molecule_dir=molecule_dir,
                split=split,
                seen_ids=seen_ids,
            )
        )
    return tuple(records)


def write_ld50_manifest(records: tuple[LD50ManifestRecord, ...], output: str | Path) -> Path:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "target",
        "split",
        "molecule_path",
        "smiles",
        "source_filename",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.to_row() for record in records)
    return output_path


def _read_split(
    *,
    csv_path: Path,
    molecule_dir: Path,
    split: str,
    seen_ids: set[str],
) -> list[LD50ManifestRecord]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"LD50 split CSV does not exist: {csv_path}")
    if not molecule_dir.is_dir():
        raise FileNotFoundError(f"LD50 molecule directory does not exist: {molecule_dir}")
    molecule_index = _molecule_index(molecule_dir)
    output: list[LD50ManifestRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(f"{csv_path} is missing required columns: {missing_columns}")
        for row_number, row in enumerate(reader, start=2):
            source_name = row["filename"].strip()
            if not source_name:
                raise ValueError(f"{csv_path}:{row_number} has an empty filename")
            molecule_path = _resolve_molecule(source_name, molecule_dir, molecule_index)
            sample_id = molecule_path.stem
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate resolved LD50 sample id: {sample_id}")
            seen_ids.add(sample_id)
            try:
                target = float(row["label"])
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}:{row_number} has an invalid label: {row['label']!r}"
                ) from exc
            output.append(
                LD50ManifestRecord(
                    sample_id=sample_id,
                    target=target,
                    split=split,
                    molecule_path=molecule_path.resolve(),
                    smiles=row["smiles"].strip(),
                    source_filename=source_name,
                )
            )
    return output


def _molecule_index(molecule_dir: Path) -> dict[tuple[str, str, str], tuple[Path, ...]]:
    mutable: dict[tuple[str, str, str], list[Path]] = {}
    for path in sorted(molecule_dir.glob("*.mol2")):
        key = normalized_cas_key(path.stem)
        if key is not None:
            mutable.setdefault(key, []).append(path)
    return {key: tuple(paths) for key, paths in mutable.items()}


def _resolve_molecule(
    source_name: str,
    molecule_dir: Path,
    molecule_index: dict[tuple[str, str, str], tuple[Path, ...]],
) -> Path:
    exact = molecule_dir / f"{source_name}.mol2"
    if exact.is_file():
        return exact
    key = normalized_cas_key(source_name)
    candidates = molecule_index.get(key, ()) if key is not None else ()
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous MOL2 match for {source_name!r}: {list(candidates)}")
    raise FileNotFoundError(f"No MOL2 file matches LD50 sample {source_name!r} in {molecule_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a mint-agent manifest for legacy LD50 data."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    records = build_ld50_manifest(args.dataset_dir)
    output = write_ld50_manifest(records, args.output)
    split_counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "test")
    }
    print(
        f"manifest={output} total={len(records)} "
        f"train={split_counts['train']} test={split_counts['test']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
