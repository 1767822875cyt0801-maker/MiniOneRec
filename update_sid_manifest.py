#!/usr/bin/env python3
"""Safely update MiniOneRec SID version entries in experiment_manifest.json."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from utils_sid import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update MiniOneRec SID experiment manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("data/Amazon/sid_maps/experiment_manifest.json"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--sid-version", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--item2sid", type=Path, required=True)
    parser.add_argument("--sid2items", type=Path, required=True)
    parser.add_argument("--valid-sid-set", type=Path, required=True)
    parser.add_argument("--item-mapping", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--valid-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    return parser.parse_args()


def validate_paths(paths: dict[str, Path]) -> None:
    missing = [f"{key}={path}" for key, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Manifest update input paths do not exist: " + "; ".join(missing))


def backup_manifest(manifest_path: Path) -> Path | None:
    if not manifest_path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = manifest_path.with_name(f"{manifest_path.name}.bak.{timestamp}")
    shutil.copy2(manifest_path, backup_path)
    return backup_path


def main() -> None:
    args = parse_args()
    required_paths = {
        "index": args.index,
        "info": args.info,
        "item2sid": args.item2sid,
        "sid2items": args.sid2items,
        "valid_sid_set": args.valid_sid_set,
        "item_mapping": args.item_mapping,
    }
    optional_paths = {
        "train_csv": args.train_csv,
        "valid_csv": args.valid_csv,
        "test_csv": args.test_csv,
    }
    validate_paths(required_paths)
    validate_paths({key: path for key, path in optional_paths.items() if path is not None})

    if args.manifest.exists():
        manifest: dict[str, Any] = load_json(args.manifest)
        if not isinstance(manifest, dict):
            raise TypeError(f"Expected manifest JSON object: {args.manifest}")
    else:
        manifest = {"sid_versions": {}}

    backup_path = backup_manifest(args.manifest)

    entry = {key: path.as_posix() for key, path in required_paths.items()}
    for key, path in optional_paths.items():
        if path is not None:
            entry[key] = path.as_posix()

    manifest.setdefault("sid_versions", {}).setdefault(args.sid_version, {})[args.category] = entry
    save_json(manifest, args.manifest)

    print("Updated SID manifest")
    print(f"  manifest: {args.manifest}")
    print(f"  backup: {backup_path if backup_path is not None else 'not_available'}")
    print(f"  sid_version: {args.sid_version}")
    print(f"  category: {args.category}")
    print(f"  entry_keys: {sorted(entry)}")
    for key in sorted(entry):
        print(f"    {key}: {entry[key]}")


if __name__ == "__main__":
    main()
