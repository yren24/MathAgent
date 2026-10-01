"""Small CLI for the MOF legacy-parity pipeline."""

from __future__ import annotations

import argparse

from .manifest import load_legacy_property_records, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="MOF legacy-parity utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("make-manifest", help="Create a CSV manifest from a legacy MOF workbook")
    manifest_parser.add_argument("--property", required=True)
    manifest_parser.add_argument("--data-dir", required=True)
    manifest_parser.add_argument("--cif-dir", required=True)
    manifest_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "make-manifest":
        records = load_legacy_property_records(
            property_name=args.property,
            data_dir=args.data_dir,
            cif_dir=args.cif_dir,
        )
        output = write_manifest(records, args.output)
        print(f"wrote {len(records)} MOF records to {output}")


if __name__ == "__main__":
    main()
