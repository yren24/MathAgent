from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import pickle
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

from mint_scout.data.structure_staging import file_sha256
from mint_scout.invariants.manifest import stable_hash


SOURCE_DOI = "10.5281/zenodo.4914718"
SOURCE_URL = (
    "https://zenodo.org/api/records/4914718/files/"
    "LBA-split-by-sequence-identity-30.tar.gz/content"
)
SOURCE_MD5 = "578ef179947bc5e3621bb24db89cbd55"
SUPPORTED_SOURCE_SPLITS = {"train", "val", "test"}
SUPPORTED_OUTPUT_SPLITS = {"train", "validation", "test", "inference"}


def prepare_atom3d_lba(
    *,
    source_root: Path,
    source_archive: Path,
    extract_root: Path,
    output_root: Path,
    manifest_path: Path,
    audit_report_path: Path,
    source_splits: tuple[str, ...] = ("test",),
    output_split: str | None = None,
    output_split_by_source: Mapping[str, str] | None = None,
    source_url: str = SOURCE_URL,
    source_archive_md5: str = SOURCE_MD5,
    max_samples: int | None = None,
    expected_sample_count: int | None = None,
) -> dict[str, Any]:
    normalized_splits = tuple(str(value).lower() for value in source_splits)
    if not normalized_splits or len(set(normalized_splits)) != len(normalized_splits):
        raise ValueError("ATOM3D-LBA source_splits must be nonempty and unique")
    unknown = sorted(set(normalized_splits) - SUPPORTED_SOURCE_SPLITS)
    if unknown:
        raise ValueError(f"Unsupported ATOM3D-LBA source splits: {unknown}")
    if max_samples is not None and max_samples < 1:
        raise ValueError("ATOM3D-LBA max_samples must be positive")
    if expected_sample_count is not None and expected_sample_count < 1:
        raise ValueError("ATOM3D-LBA expected_sample_count must be positive")
    if output_split is not None and output_split not in SUPPORTED_OUTPUT_SPLITS:
        raise ValueError(f"Unsupported ATOM3D-LBA output_split: {output_split!r}")
    normalized_mapping = {
        str(source).lower(): str(destination).lower()
        for source, destination in (output_split_by_source or {}).items()
    }
    if output_split is not None and normalized_mapping:
        raise ValueError(
            "ATOM3D-LBA output_split and output_split_by_source are mutually exclusive"
        )
    unknown_mapping_sources = sorted(set(normalized_mapping) - set(normalized_splits))
    if unknown_mapping_sources:
        raise ValueError(
            "ATOM3D-LBA split mapping contains unselected source splits: "
            f"{unknown_mapping_sources}"
        )
    missing_mapping_sources = sorted(set(normalized_splits) - set(normalized_mapping))
    if normalized_mapping and missing_mapping_sources:
        raise ValueError(
            "ATOM3D-LBA split mapping is incomplete for source splits: "
            f"{missing_mapping_sources}"
        )
    invalid_destinations = sorted(
        set(normalized_mapping.values()) - SUPPORTED_OUTPUT_SPLITS
    )
    if invalid_destinations:
        raise ValueError(
            "ATOM3D-LBA split mapping contains unsupported output splits: "
            f"{invalid_destinations}"
        )

    source_root = source_root.expanduser().resolve()
    source_archive = source_archive.expanduser().resolve()
    extract_root = extract_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    downloaded = False
    extracted = False
    if not source_archive.is_file():
        _download_https(source_url, source_archive)
        downloaded = True
    archive_md5 = _file_digest(source_archive, "md5")
    if archive_md5.lower() != source_archive_md5.lower():
        raise ValueError(
            "ATOM3D-LBA archive checksum mismatch: "
            f"expected {source_archive_md5.lower()}, observed {archive_md5.lower()}"
        )
    required_split_paths = tuple(source_root / value for value in normalized_splits)
    if not all(path.exists() for path in required_split_paths):
        _extract_tar_archive(source_archive, extract_root)
        extracted = True
    if not all(path.exists() for path in required_split_paths):
        raise FileNotFoundError(
            "ATOM3D-LBA archive did not create the configured split paths under "
            f"source_root: {source_root}"
        )

    manifest_rows: list[dict[str, str]] = []
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_split in normalized_splits:
        split_path = source_root / source_split
        for source_index, item in enumerate(_iter_lmdb(split_path)):
            sample_id = _sample_id(item.get("id"), source_split, source_index)
            if sample_id in seen:
                raise ValueError(f"Duplicate ATOM3D-LBA sample ID: {sample_id}")
            seen.add(sample_id)
            target = _target(item)
            pocket_rows = _atom_rows(item.get("atoms_pocket"), "atoms_pocket")
            ligand_rows = _atom_rows(item.get("atoms_ligand"), "atoms_ligand")
            sample_root = output_root / sample_id
            protein_path = sample_root / f"{sample_id}_pocket.pdb"
            ligand_path = sample_root / f"{sample_id}_ligand.mol2"
            _write_pdb(protein_path, pocket_rows)
            _write_mol2(ligand_path, ligand_rows, sample_id)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "target": f"{target:.15g}",
                    "split": output_split or normalized_mapping.get(source_split, ""),
                    "protein_path": str(protein_path),
                    "ligand_path": str(ligand_path),
                    "atom3d_id": str(item.get("id")),
                    "source_split": source_split,
                }
            )
            samples.append(
                {
                    "sample_id": sample_id,
                    "atom3d_id": str(item.get("id")),
                    "source_split": source_split,
                    "source_index": source_index,
                    "target_neglog_aff": target,
                    "pocket_atom_count": len(pocket_rows),
                    "ligand_atom_count": len(ligand_rows),
                    "source_item_hash": stable_hash(
                        {
                            "id": str(item.get("id")),
                            "target": target,
                            "pocket": pocket_rows,
                            "ligand": ligand_rows,
                        }
                    ),
                    "prepared_protein": str(protein_path),
                    "prepared_ligand": str(ligand_path),
                    "prepared_protein_sha256": file_sha256(protein_path),
                    "prepared_ligand_sha256": file_sha256(ligand_path),
                }
            )
            if max_samples is not None and len(samples) >= max_samples:
                break
        if max_samples is not None and len(samples) >= max_samples:
            break
    if not samples:
        raise ValueError("ATOM3D-LBA selection contains no samples")
    if expected_sample_count is not None and len(samples) != expected_sample_count:
        raise ValueError(
            "ATOM3D-LBA sample-count mismatch: "
            f"expected {expected_sample_count}, observed {len(samples)}"
        )

    _write_manifest(manifest_path, manifest_rows)
    payload = {
        "report_schema": "mint-agent.atom3d-lba-preparation.v1",
        "dataset_id": "atom3d-lba",
        "evidence_scope": "design",
        "status": "PASS",
        "source_doi": SOURCE_DOI,
        "source_url": source_url,
        "source_license": "CC-BY-NC-ND; source data are not redistributed by mint-agent",
        "source_archive": str(source_archive),
        "source_archive_md5": archive_md5,
        "source_archive_downloaded": downloaded,
        "source_archive_extracted": extracted,
        "source_root": str(source_root),
        "source_splits": list(normalized_splits),
        "output_split": output_split,
        "output_split_by_source": normalized_mapping or None,
        "max_samples": max_samples,
        "expected_sample_count": expected_sample_count,
        "sample_count": len(samples),
        "sample_order_hash": stable_hash([row["sample_id"] for row in manifest_rows]),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "preprocessing": {
            "protein": "ATOM3D atoms_pocket coordinates written to fixed-column PDB",
            "ligand": "ATOM3D atoms_ligand coordinates written to coordinate-only MOL2",
            "coordinate_transform": "none",
            "hydrogens_added": False,
            "bonds_used_by_invariant_tools": False,
        },
        "target": {
            "source_field": "scores.neglog_aff",
            "interpretation": "source-provided pK = -log10 molar affinity",
            "used_for_representation_design": False,
        },
        "samples": samples,
    }
    audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    audit_report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _iter_lmdb(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"ATOM3D-LBA split is missing: {path}")
    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError(
            "ATOM3D-LBA preparation requires the optional mint-agent[datasets] dependency"
        ) from exc
    environment = lmdb.open(
        str(path),
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=path.is_dir(),
    )
    try:
        with environment.begin(write=False) as transaction:
            raw_count = transaction.get(b"num_examples")
            raw_format = transaction.get(b"serialization_format")
            if raw_count is None or raw_format is None:
                raise ValueError(f"Invalid ATOM3D LMDB metadata: {path}")
            count = int(raw_count)
            serialization_format = raw_format.decode("utf-8")
        for index in range(count):
            with environment.begin(write=False) as transaction:
                compressed = transaction.get(str(index).encode("ascii"))
            if compressed is None:
                raise ValueError(f"ATOM3D LMDB entry {index} is missing: {path}")
            serialized = gzip.decompress(compressed)
            item = _deserialize(serialized, serialization_format)
            if not isinstance(item, dict):
                raise ValueError(f"ATOM3D LMDB entry {index} is not an object")
            yield item
    finally:
        environment.close()


def _deserialize(payload: bytes, serialization_format: str) -> Any:
    if serialization_format == "json":
        return json.loads(payload)
    if serialization_format == "pkl":
        return pickle.loads(payload)
    if serialization_format == "msgpack":
        try:
            import msgpack
        except ImportError as exc:
            raise RuntimeError("msgpack is required to read this ATOM3D LMDB") from exc
        return msgpack.unpackb(payload, raw=False)
    raise ValueError(f"Unsupported ATOM3D serialization format: {serialization_format!r}")


def _atom_rows(frame: object, name: str) -> list[dict[str, Any]]:
    if hasattr(frame, "to_dict"):
        raw_rows = frame.to_dict(orient="records")
    elif isinstance(frame, Mapping):
        columns = frame.get("columns")
        data = frame.get("data")
        if not isinstance(columns, list) or not isinstance(data, list):
            raise ValueError(f"ATOM3D {name} does not use DataFrame split encoding")
        raw_rows = [dict(zip(columns, values)) for values in data]
    else:
        raise ValueError(f"ATOM3D item lacks {name}")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"ATOM3D {name} row {index} is invalid")
        try:
            coordinates = tuple(float(raw[key]) for key in ("x", "y", "z"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"ATOM3D {name} row {index} lacks numeric coordinates") from exc
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"ATOM3D {name} row {index} has non-finite coordinates")
        element = _element(raw.get("element"))
        rows.append(
            {
                "x": coordinates[0],
                "y": coordinates[1],
                "z": coordinates[2],
                "element": element,
                "name": str(raw.get("name") or f"{element}{index}"),
                "resname": str(raw.get("resname") or "UNK"),
                "chain": str(raw.get("chain") or "A"),
                "residue": raw.get("residue", 1),
                "insertion_code": str(raw.get("insertion_code") or ""),
                "occupancy": raw.get("occupancy", 1.0),
                "bfactor": raw.get("bfactor", 0.0),
            }
        )
    if not rows:
        raise ValueError(f"ATOM3D {name} contains no atoms")
    return rows


def _target(item: Mapping[str, Any]) -> float:
    scores = item.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError("ATOM3D-LBA item lacks scores")
    try:
        value = float(scores["neglog_aff"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ATOM3D-LBA item lacks numeric scores.neglog_aff") from exc
    if not math.isfinite(value):
        raise ValueError("ATOM3D-LBA scores.neglog_aff must be finite")
    return value


def _sample_id(value: object, source_split: str, source_index: int) -> str:
    sample_id = str(value or "").strip()
    if not sample_id:
        raise ValueError(
            f"ATOM3D-LBA {source_split} entry {source_index} has no sample ID"
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", sample_id):
        raise ValueError(f"Unsafe ATOM3D-LBA sample ID: {sample_id!r}")
    return sample_id


def _element(value: object) -> str:
    text = re.sub(r"[^A-Za-z]", "", str(value or ""))
    if not text:
        raise ValueError(f"Invalid ATOM3D element: {value!r}")
    return text[0].upper() + text[1:].lower()


def _write_pdb(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for serial, row in enumerate(rows, start=1):
        atom_name = str(row["name"]).strip()[:4] or row["element"]
        resname = str(row["resname"]).strip()[:3].upper() or "UNK"
        chain = str(row["chain"]).strip()[:1] or "A"
        try:
            residue = int(row["residue"])
        except (TypeError, ValueError):
            residue = 1
        try:
            occupancy = float(row["occupancy"])
        except (TypeError, ValueError):
            occupancy = 1.0
        try:
            bfactor = float(row["bfactor"])
        except (TypeError, ValueError):
            bfactor = 0.0
        lines.append(
            f"ATOM  {serial:5d} {atom_name:>4} {resname:>3} {chain}{residue:4d}    "
            f"{row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}"
            f"{occupancy:6.2f}{bfactor:6.2f}          {row['element']:>2}\n"
        )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(lines) + "END\n", encoding="utf-8")
    temporary.replace(path)


def _write_mol2(path: Path, rows: list[dict[str, Any]], sample_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "@<TRIPOS>MOLECULE\n",
        f"{sample_id}\n",
        f"{len(rows)} 0 0 0 0\n",
        "SMALL\n",
        "NO_CHARGES\n\n",
        "@<TRIPOS>ATOM\n",
    ]
    for serial, row in enumerate(rows, start=1):
        atom_name = re.sub(r"\s+", "", str(row["name"]))[:8] or f"A{serial}"
        lines.append(
            f"{serial:7d} {atom_name:<8} {row['x']:12.6f} {row['y']:12.6f} "
            f"{row['z']:12.6f} {row['element']:<6} 1 LIG 0.0000\n"
        )
    lines.append("@<TRIPOS>BOND\n")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(lines), encoding="utf-8")
    temporary.replace(path)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "target",
                "split",
                "protein_path",
                "ligand_path",
                "atom3d_id",
                "source_split",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _download_https(url: str, destination: Path) -> None:
    if not url.startswith("https://"):
        raise ValueError("ATOM3D-LBA source_url must use HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "mint-agent/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as handle:
            if not response.geturl().startswith("https://"):
                raise ValueError("ATOM3D-LBA download redirected away from HTTPS")
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _extract_tar_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, mode="r:*") as handle:
        members = handle.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"archive member escapes extraction root: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name}")
        handle.extractall(root, members=members)


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
