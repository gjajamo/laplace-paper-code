from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine method-result CSV files produced by separate staged runs. "
            "This is useful for the Warfarin manuscript workflow, where AD, FD, "
            "and staged combo methods can be run as separate batches before "
            "creating the six-method figures."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input CSV files to concatenate in the requested order.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Combined CSV path to write.",
    )
    parser.add_argument(
        "--add-source-column",
        action="store_true",
        help="Add a source_file column identifying each input CSV.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        return list(reader.fieldnames), list(reader)


def main() -> None:
    args = parse_args()
    all_fields: list[str] = []
    all_rows: list[dict[str, str]] = []

    for path in args.inputs:
        fields, rows = read_rows(path)
        for field in fields:
            if field not in all_fields:
                all_fields.append(field)
        if args.add_source_column and "source_file" not in all_fields:
            all_fields.append("source_file")
        for row in rows:
            if args.add_source_column:
                row = dict(row)
                row["source_file"] = path.as_posix()
            all_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({field: row.get(field, "") for field in all_fields})

    methods = sorted({row.get("method", "") for row in all_rows if row.get("method", "")})
    print(f"Wrote {len(all_rows)} rows to {args.output}")
    if methods:
        print("Methods: " + ", ".join(methods))


if __name__ == "__main__":
    main()
