#!/usr/bin/env python3
"""Stage-0 SID mapping verifier for MiniOneRec.

Run from the project root after build_sid_mapping.py has generated data/Amazon/sid_maps/.
Only uses Python standard library plus utils_sid.py from the project.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

from utils_sid import load_json, normalize_sid, parse_sid_tokens


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    value_str = "" if value is None else str(value).strip()
    if not value_str:
        return []
    parsed = ast.literal_eval(value_str)
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


def rate(num: int, den: int) -> float:
    return 1.0 if den == 0 else num / den


def load_index(index_path: Path) -> dict[str, str]:
    raw = load_json(index_path)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict index JSON: {index_path}")
    out: dict[str, str] = {}
    for item_id, sid_value in raw.items():
        if isinstance(sid_value, list):
            sid = "".join(str(x).strip() for x in sid_value)
        else:
            sid = str(sid_value)
        out[str(item_id)] = normalize_sid(sid)
    return out


def invert_item2sid(item2sid: dict[str, str]) -> dict[str, list[str]]:
    sid2items: dict[str, list[str]] = {}
    for item_id, sid in item2sid.items():
        sid2items.setdefault(normalize_sid(sid), []).append(str(item_id))
    for sid in sid2items:
        sid2items[sid] = sorted(sid2items[sid], key=item_sort_key)
    return dict(sorted(sid2items.items()))


def item_sort_key(x: Any) -> tuple[int, int | str]:
    s = str(x)
    try:
        return (0, int(s))
    except ValueError:
        return (1, s)


def csv_check(csv_path: Path, item2sid: dict[str, str]) -> dict[str, Any]:
    rows = 0
    target_sid_ok = 0
    all_sid_ok = 0
    all_sid_total = 0
    history_sid_ok_rows = 0
    history_len_ok = 0
    csv_items: set[str] = set()
    missing_items: set[str] = set()
    mapping_conflicts: list[dict[str, str]] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"item_id", "item_sid", "history_item_id", "history_item_sid"}
        missing_cols = sorted(required - set(reader.fieldnames or []))
        if missing_cols:
            raise ValueError(f"{csv_path} missing columns: {missing_cols}")

        for row_no, row in enumerate(reader, start=2):
            rows += 1
            item_id = str(row.get("item_id", "")).strip()
            item_sid = normalize_sid(row.get("item_sid", ""))
            if item_id:
                csv_items.add(item_id)
                if item_id not in item2sid:
                    missing_items.add(item_id)
                elif item_sid and item2sid[item_id] != item_sid:
                    mapping_conflicts.append({
                        "row": str(row_no),
                        "item_id": item_id,
                        "csv_sid": item_sid,
                        "map_sid": item2sid[item_id],
                    })

            valid_target = bool(parse_sid_tokens(item_sid))
            target_sid_ok += int(valid_target)
            all_sid_ok += int(valid_target)
            all_sid_total += 1

            try:
                history_ids = parse_list(row.get("history_item_id", "[]"))
                history_sids = parse_list(row.get("history_item_sid", "[]"))
            except Exception as exc:
                raise ValueError(f"Unable to parse history lists at {csv_path}:{row_no}") from exc

            if len(history_ids) == len(history_sids):
                history_len_ok += 1

            history_valid = True
            for hid, hsid_raw in zip(history_ids, history_sids):
                hid = str(hid).strip()
                hsid = normalize_sid(str(hsid_raw))
                if hid:
                    csv_items.add(hid)
                    if hid not in item2sid:
                        missing_items.add(hid)
                    elif hsid and item2sid[hid] != hsid:
                        mapping_conflicts.append({
                            "row": str(row_no),
                            "item_id": hid,
                            "csv_sid": hsid,
                            "map_sid": item2sid[hid],
                        })
                valid = bool(parse_sid_tokens(hsid))
                history_valid = history_valid and valid
                all_sid_ok += int(valid)
                all_sid_total += 1
            history_sid_ok_rows += int(history_valid)

    covered = sum(1 for item in csv_items if item in item2sid)
    return {
        "rows": rows,
        "sid_parse_rate": rate(all_sid_ok, all_sid_total),
        "item_sid_parse_rate": rate(target_sid_ok, rows),
        "history_sid_parse_rate": rate(history_sid_ok_rows, rows),
        "history_len_consistency_rate": rate(history_len_ok, rows),
        "csv_item_coverage_rate": rate(covered, len(csv_items)),
        "unique_csv_items": len(csv_items),
        "missing_items_count": len(missing_items),
        "missing_items_sample": sorted(missing_items, key=item_sort_key)[:20],
        "mapping_conflicts_count": len(mapping_conflicts),
        "mapping_conflicts_sample": mapping_conflicts[:20],
    }


def check_category_paths(
    category: str,
    sid_version: str,
    item2sid_path: Path,
    sid2items_path: Path,
    valid_sid_set_path: Path,
    item_mapping_path: Path,
    split_csv_paths: dict[str, Path],
    index_path: Path | None = None,
) -> dict[str, Any]:
    for path in [item2sid_path, sid2items_path, valid_sid_set_path, item_mapping_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    for split, path in split_csv_paths.items():
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unexpected split name: {split}")
        if not path.exists():
            raise FileNotFoundError(path)

    item2sid = {str(k): normalize_sid(v) for k, v in load_json(item2sid_path).items()}
    sid2items = {normalize_sid(k): [str(x) for x in v] for k, v in load_json(sid2items_path).items()}
    valid_sid_set = set(normalize_sid(x) for x in load_json(valid_sid_set_path))
    item_mapping = load_json(item_mapping_path)

    bad_item_sids = {item: sid for item, sid in item2sid.items() if not parse_sid_tokens(sid)}
    inverse = invert_item2sid(item2sid)
    inverse_ok = inverse == {sid: sorted(items, key=item_sort_key) for sid, items in sid2items.items()}
    valid_set_ok = valid_sid_set == set(sid2items.keys())

    item_mapping_mismatches = []
    for item_id, sid in item2sid.items():
        rec = item_mapping.get(item_id)
        if not rec or normalize_sid(rec.get("sid", "")) != sid or rec.get("sid_tokens") != parse_sid_tokens(sid):
            item_mapping_mismatches.append(item_id)

    index_stats: dict[str, Any] | None = None
    if index_path is not None and index_path.exists():
        index_map = load_index(index_path)
        shared = set(index_map) & set(item2sid)
        conflicts = [item for item in shared if index_map[item] != item2sid[item]]
        index_stats = {
            "index_items": len(index_map),
            "item2sid_items": len(item2sid),
            "shared_items": len(shared),
            "index_count_equals_item2sid": len(index_map) == len(item2sid),
            "index_mapping_conflicts_count": len(conflicts),
            "index_mapping_conflicts_sample": sorted(conflicts, key=item_sort_key)[:20],
        }

    split_stats = {}
    for split in ["train", "valid", "test"]:
        split_stats[split] = csv_check(split_csv_paths[split], item2sid)

    ok = (
        not bad_item_sids
        and inverse_ok
        and valid_set_ok
        and not item_mapping_mismatches
        and all(v["sid_parse_rate"] == 1.0 for v in split_stats.values())
        and all(v["csv_item_coverage_rate"] == 1.0 for v in split_stats.values())
        and all(v["history_len_consistency_rate"] == 1.0 for v in split_stats.values())
        and all(v["mapping_conflicts_count"] == 0 for v in split_stats.values())
        and (index_stats is None or (index_stats["index_count_equals_item2sid"] and index_stats["index_mapping_conflicts_count"] == 0))
    )

    return {
        "category": category,
        "sid_version": sid_version,
        "ok": ok,
        "num_items": len(item2sid),
        "num_unique_sid": len(sid2items),
        "bad_item_sids_count": len(bad_item_sids),
        "bad_item_sids_sample": list(bad_item_sids.items())[:20],
        "sid2items_inverse_ok": inverse_ok,
        "valid_sid_set_equals_sid2items_keys": valid_set_ok,
        "item_mapping_mismatches_count": len(item_mapping_mismatches),
        "item_mapping_mismatches_sample": sorted(item_mapping_mismatches, key=item_sort_key)[:20],
        "index_stats": index_stats,
        "split_stats": split_stats,
    }


def check_category(data_root: Path, sid_map_dir: Path, category: str, sid_version: str) -> dict[str, Any]:
    cat_dir = sid_map_dir / category
    split_csv_paths = {}
    for split in ["train", "valid", "test"]:
        matches = sorted((data_root / split).glob(f"{category}_*.csv"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {split} CSV for {category}, found {matches}")
        split_csv_paths[split] = matches[0]

    return check_category_paths(
        category=category,
        sid_version=sid_version,
        item2sid_path=cat_dir / f"item2sid_{sid_version}.json",
        sid2items_path=cat_dir / f"sid2items_{sid_version}.json",
        valid_sid_set_path=cat_dir / f"valid_sid_set_{sid_version}.json",
        item_mapping_path=cat_dir / f"item_mapping_{sid_version}.json",
        split_csv_paths=split_csv_paths,
        index_path=data_root / "index" / f"{category}.index.json",
    )


def check_manifest_category(manifest: dict[str, Any], category: str, sid_version: str) -> dict[str, Any]:
    try:
        entry = manifest["sid_versions"][sid_version][category]
    except KeyError as exc:
        raise KeyError(f"Missing manifest entry sid_versions[{sid_version!r}][{category!r}]") from exc

    required = [
        "item2sid",
        "sid2items",
        "valid_sid_set",
        "item_mapping",
        "train_csv",
        "valid_csv",
        "test_csv",
    ]
    missing = [key for key in required if key not in entry]
    if missing:
        raise KeyError(f"Manifest entry for {sid_version}/{category} missing keys: {missing}")

    index_path = Path(entry["index"]) if "index" in entry else None
    return check_category_paths(
        category=category,
        sid_version=sid_version,
        item2sid_path=Path(entry["item2sid"]),
        sid2items_path=Path(entry["sid2items"]),
        valid_sid_set_path=Path(entry["valid_sid_set"]),
        item_mapping_path=Path(entry["item_mapping"]),
        split_csv_paths={
            "train": Path(entry["train_csv"]),
            "valid": Path(entry["valid_csv"]),
            "test": Path(entry["test_csv"]),
        },
        index_path=index_path,
    )


def discover_categories(sid_map_dir: Path, sid_version: str) -> list[str]:
    cats = []
    for d in sorted(p for p in sid_map_dir.iterdir() if p.is_dir()):
        if (d / f"item2sid_{sid_version}.json").exists():
            cats.append(d.name)
    return cats


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MiniOneRec stage-0 SID mapping artifacts.")
    parser.add_argument("--data-root", type=Path, default=Path("data/Amazon"))
    parser.add_argument("--sid-map-dir", type=Path, default=Path("data/Amazon/sid_maps"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--sid-version", default="text")
    parser.add_argument("--report-path", type=Path, default=Path("data/Amazon/sid_maps/stage0_check_report.json"))
    args = parser.parse_args()

    if args.manifest is not None:
        manifest = load_json(args.manifest)
        sid_versions = manifest.get("sid_versions", {})
        if args.sid_version not in sid_versions:
            raise KeyError(f"sid_version={args.sid_version!r} not found in {args.manifest}")
        categories = []
        if args.category:
            categories.append(args.category)
        if args.categories:
            categories.extend(args.categories)
        if not categories:
            categories = sorted(sid_versions[args.sid_version])
        reports = [check_manifest_category(manifest, c, args.sid_version) for c in categories]
    else:
        categories = []
        if args.category:
            categories.append(args.category)
        if args.categories:
            categories.extend(args.categories)
        if not categories:
            categories = discover_categories(args.sid_map_dir, args.sid_version)
        reports = [check_category(args.data_root, args.sid_map_dir, c, args.sid_version) for c in categories]

    if not categories:
        raise ValueError(f"No categories found for sid_version={args.sid_version}")
    final = {
        "ok": all(r["ok"] for r in reports),
        "sid_version": args.sid_version,
        "categories": reports,
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote report: {args.report_path}")
    if not final["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
