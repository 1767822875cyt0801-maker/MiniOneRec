#!/usr/bin/env python3
"""Build the S6 validation-only development split manifest.

This utility reads the existing validation CSV only. It does not create split
CSVs, does not read test data, and does not run any model workload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORY = "Industrial_and_Scientific"
DEFAULT_SOURCE_VALID = ROOT / "data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv"
DEFAULT_OUTPUT = (
    ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific/s6_validation_split_manifest.json"
)
DEFAULT_SEED = 42
SPLIT_BOUNDS = [
    ("valid_fit", 60),
    ("valid_select", 80),
    ("valid_gate", 100),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build S6 validation split manifest.")
    parser.add_argument("--source-valid-csv", type=Path, default=DEFAULT_SOURCE_VALID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-key", default="user_id")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def no_test_path(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    return "test" not in parts and path.name.lower() != "test.csv"


def split_for_key(value: str, seed: int) -> str:
    digest = hashlib.sha256(f"s6_validation_split.v1:{seed}:{value}".encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) % 100
    for split_name, upper in SPLIT_BOUNDS:
        if bucket < upper:
            return split_name
    raise AssertionError("unreachable split bucket")


def canonical_split_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_manifest(source_valid_csv: Path, category: str, seed: int, split_key: str) -> dict[str, Any]:
    if not no_test_path(source_valid_csv):
        raise ValueError(f"S6 validation split refuses test paths: {source_valid_csv}")
    if not source_valid_csv.is_file():
        raise FileNotFoundError(source_valid_csv)

    with open(source_valid_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if split_key not in fieldnames:
        raise ValueError(f"split key {split_key!r} is not present in {source_valid_csv}")

    user_to_split: dict[str, str] = {}
    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        key = str(row.get(split_key, "")).strip()
        if not key:
            raise ValueError(f"missing split key {split_key!r} at source row {row_index}")
        split = user_to_split.setdefault(key, split_for_key(key, seed))
        split_rows[split].append(
            {
                "source_row_index": row_index,
                split_key: key,
                "item_id": str(row.get("item_id", "")),
                "history_item_id": str(row.get("history_item_id", "")),
            }
        )

    split_summary: dict[str, dict[str, Any]] = {}
    all_users_by_split: dict[str, set[str]] = {}
    for split_name, _upper in SPLIT_BOUNDS:
        rows_for_split = split_rows.get(split_name, [])
        users = {str(row[split_key]) for row in rows_for_split}
        all_users_by_split[split_name] = users
        split_summary[split_name] = {
            "row_count": len(rows_for_split),
            "user_count": len(users),
            "row_order": "source CSV row order preserved within this split",
            "sha256": canonical_split_hash(rows_for_split),
        }

    overlaps: dict[str, int] = {}
    split_names = [name for name, _upper in SPLIT_BOUNDS]
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            overlaps[f"{left}__{right}"] = len(all_users_by_split[left] & all_users_by_split[right])

    return {
        "schema": "s6_validation_split_manifest.v1",
        "category": category,
        "source_valid_csv": relative_or_absolute(source_valid_csv),
        "source_valid_csv_sha256": file_sha256(source_valid_csv),
        "source_valid_rows": len(rows),
        "source_valid_columns": fieldnames,
        "split_algorithm": {
            "name": "stable_user_hash_mod_100",
            "version": "s6_validation_split.v1",
            "seed": seed,
            "split_key": split_key,
            "assignment": "sha256('s6_validation_split.v1:{seed}:{user_id}') mod 100",
            "proportions": {
                "valid_fit": "bucket < 60",
                "valid_select": "60 <= bucket < 80",
                "valid_gate": "80 <= bucket < 100",
            },
            "row_policy": "preserve original source row order inside each split",
        },
        "splits": split_summary,
        "overlap_checks": overlaps,
        "total_user_count": len(user_to_split),
        "total_split_rows": sum(item["row_count"] for item in split_summary.values()),
        "development_gate": True,
        "independent_final_test": False,
        "test_read": False,
    }


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.source_valid_csv, args.category, args.seed, args.split_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
