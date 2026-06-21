#!/usr/bin/env python3
import argparse
import ast
import csv
from pathlib import Path
from typing import Any

from utils_sid import (
    build_item2sid_from_csv_or_info,
    build_sid2items,
    load_json,
    normalize_sid,
    parse_sid_tokens,
    save_json,
)


def discover_categories(data_root: Path) -> list[str]:
    index_dir = data_root / "index"
    return sorted(path.name[: -len(".index.json")] for path in index_dir.glob("*.index.json"))


def find_one(base_dir: Path, pattern: str) -> Path:
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched {base_dir / pattern}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous files for {base_dir / pattern}: {matches}")
    return matches[0]


def load_index_mapping(index_path: Path) -> dict[str, str]:
    raw_index = load_json(index_path)
    if not isinstance(raw_index, dict):
        raise TypeError(f"Expected dict index JSON: {index_path}")

    mapping: dict[str, str] = {}
    for item_id, sid_value in raw_index.items():
        if isinstance(sid_value, list):
            sid = "".join(str(token).strip() for token in sid_value)
        else:
            sid = str(sid_value)
        mapping[str(item_id)] = normalize_sid(sid)
    return mapping


def load_info_records(info_path: Path) -> tuple[dict[str, str], dict[str, str], int]:
    item2sid: dict[str, str] = {}
    item2title: dict[str, str] = {}
    malformed = 0
    with open(info_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                malformed += 1
                continue
            sid, title, item_id = parts[0], parts[1], parts[2]
            item_id = str(item_id).strip()
            item2sid[item_id] = normalize_sid(sid)
            item2title[item_id] = title.strip()
    return item2sid, item2title, malformed


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


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def collect_csv_titles(csv_paths: list[Path]) -> dict[str, str]:
    titles: dict[str, str] = {}
    for csv_path in csv_paths:
        for row in read_csv_rows(csv_path):
            item_id = str(row.get("item_id", "")).strip()
            item_title = str(row.get("item_title", "")).strip()
            if item_id and item_title and item_id not in titles:
                titles[item_id] = item_title

            if "history_item_id" not in row or "history_item_title" not in row:
                continue
            try:
                history_ids = parse_list(row["history_item_id"])
                history_titles = parse_list(row["history_item_title"])
            except (SyntaxError, ValueError):
                continue
            for history_id, history_title in zip(history_ids, history_titles):
                history_id = str(history_id).strip()
                history_title = str(history_title).strip()
                if history_id and history_title and history_id not in titles:
                    titles[history_id] = history_title
    return titles


def info_index_stats(index_map: dict[str, str], info_map: dict[str, str]) -> dict[str, int]:
    shared_items = set(index_map) & set(info_map)
    matches = sum(1 for item_id in shared_items if index_map[item_id] == info_map[item_id])
    conflicts = sum(1 for item_id in shared_items if index_map[item_id] != info_map[item_id])
    return {
        "index_items": len(index_map),
        "info_items": len(info_map),
        "shared_items": len(shared_items),
        "matches": matches,
        "conflicts": conflicts,
    }


def csv_stats(csv_path: Path, item2sid: dict[str, str]) -> dict[str, float | int]:
    rows = read_csv_rows(csv_path)
    target_sid_ok = 0
    history_sid_ok_rows = 0
    all_sid_ok = 0
    all_sid_total = 0
    length_ok = 0
    csv_item_ids: set[str] = set()
    mapping_conflicts = 0

    for row in rows:
        item_id = str(row.get("item_id", "")).strip()
        item_sid = normalize_sid(row.get("item_sid", ""))
        if item_id:
            csv_item_ids.add(item_id)
            if item_id in item2sid and item_sid and item2sid[item_id] != item_sid:
                mapping_conflicts += 1

        item_sid_valid = bool(parse_sid_tokens(item_sid))
        target_sid_ok += int(item_sid_valid)
        all_sid_ok += int(item_sid_valid)
        all_sid_total += 1

        try:
            history_item_ids = parse_list(row.get("history_item_id", "[]"))
            history_item_sids = parse_list(row.get("history_item_sid", "[]"))
        except (SyntaxError, ValueError):
            history_item_ids = []
            history_item_sids = []

        length_ok += int(len(history_item_ids) == len(history_item_sids))
        for history_id in history_item_ids:
            csv_item_ids.add(str(history_id).strip())

        history_valid = True
        for sid in history_item_sids:
            valid = bool(parse_sid_tokens(str(sid)))
            history_valid = history_valid and valid
            all_sid_ok += int(valid)
            all_sid_total += 1
        history_sid_ok_rows += int(history_valid)

        for history_id, history_sid in zip(history_item_ids, history_item_sids):
            history_id = str(history_id).strip()
            history_sid = normalize_sid(str(history_sid))
            if history_id in item2sid and history_sid and item2sid[history_id] != history_sid:
                mapping_conflicts += 1

    covered_items = sum(1 for item_id in csv_item_ids if item_id in item2sid)
    return {
        "rows": len(rows),
        "sid_parse_rate": _rate(all_sid_ok, all_sid_total),
        "item_sid_parse_rate": _rate(target_sid_ok, len(rows)),
        "history_sid_parse_rate": _rate(history_sid_ok_rows, len(rows)),
        "csv_item_coverage_rate": _rate(covered_items, len(csv_item_ids)),
        "history_len_consistency_rate": _rate(length_ok, len(rows)),
        "unique_csv_items": len(csv_item_ids),
        "mapping_conflicts": mapping_conflicts,
    }


def build_item_mapping(item2sid: dict[str, str], item2title: dict[str, str]) -> dict[str, dict[str, Any]]:
    return {
        item_id: {
            "sid": sid,
            "title": item2title.get(item_id, ""),
            "sid_tokens": parse_sid_tokens(sid),
        }
        for item_id, sid in item2sid.items()
    }


def process_category(
    data_root: Path,
    output_dir: Path,
    category: str,
    sid_version: str,
    strict: bool,
) -> dict[str, str]:
    train_csv = find_one(data_root / "train", f"{category}_*.csv")
    valid_csv = find_one(data_root / "valid", f"{category}_*.csv")
    test_csv = find_one(data_root / "test", f"{category}_*.csv")
    index_path = data_root / "index" / f"{category}.index.json"
    info_path = find_one(data_root / "info", f"{category}_*.txt")

    csv_paths = [train_csv, valid_csv, test_csv]
    index_map = load_index_mapping(index_path)
    info_map, info_titles, malformed_info_rows = load_info_records(info_path)
    consistency = info_index_stats(index_map, info_map)

    if strict and consistency["conflicts"]:
        raise ValueError(f"{category}: info/index SID conflicts found")

    item2sid = build_item2sid_from_csv_or_info(
        csv_paths=csv_paths,
        info_path=info_path,
        index_path=index_path,
        strict=strict,
    )
    sid2items = build_sid2items(item2sid)
    csv_titles = collect_csv_titles(csv_paths)
    item2title = {**csv_titles, **info_titles}
    item_mapping = build_item_mapping(item2sid, item2title)
    valid_sid_set = sorted(sid2items.keys())

    category_dir = output_dir / category
    item2sid_path = category_dir / f"item2sid_{sid_version}.json"
    sid2items_path = category_dir / f"sid2items_{sid_version}.json"
    valid_sid_set_path = category_dir / f"valid_sid_set_{sid_version}.json"
    item_mapping_path = category_dir / f"item_mapping_{sid_version}.json"

    save_json(item2sid, item2sid_path)
    save_json(sid2items, sid2items_path)
    save_json(valid_sid_set, valid_sid_set_path)
    save_json(item_mapping, item_mapping_path)

    print(
        f"[{category}] info/index: "
        f"index_items={consistency['index_items']} "
        f"info_items={consistency['info_items']} "
        f"shared_items={consistency['shared_items']} "
        f"matches={consistency['matches']} "
        f"conflicts={consistency['conflicts']} "
        f"malformed_info_rows={malformed_info_rows}"
    )
    for split, csv_path in [("train", train_csv), ("valid", valid_csv), ("test", test_csv)]:
        stats = csv_stats(csv_path, item2sid)
        print(
            f"[{category}][{split}] "
            f"rows={stats['rows']} "
            f"sid_parse_rate={stats['sid_parse_rate']:.6f} "
            f"item_sid_parse_rate={stats['item_sid_parse_rate']:.6f} "
            f"history_sid_parse_rate={stats['history_sid_parse_rate']:.6f} "
            f"csv_item_coverage_rate={stats['csv_item_coverage_rate']:.6f} "
            f"history_len_consistency_rate={stats['history_len_consistency_rate']:.6f} "
            f"unique_csv_items={stats['unique_csv_items']} "
            f"mapping_conflicts={stats['mapping_conflicts']}"
        )

    return {
        "train_csv": _path_str(train_csv),
        "valid_csv": _path_str(valid_csv),
        "test_csv": _path_str(test_csv),
        "index": _path_str(index_path),
        "info": _path_str(info_path),
        "item2sid": _path_str(item2sid_path),
        "sid2items": _path_str(sid2items_path),
        "valid_sid_set": _path_str(valid_sid_set_path),
        "item_mapping": _path_str(item_mapping_path),
    }


def update_manifest(
    manifest_path: Path,
    sid_version: str,
    category_entries: dict[str, dict[str, str]],
) -> None:
    if manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = {"sid_versions": {}}

    manifest.setdefault("sid_versions", {}).setdefault(sid_version, {}).update(category_entries)
    save_json(manifest, manifest_path)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _path_str(path: Path) -> str:
    return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Text-SID mapping files from MiniOneRec data.")
    parser.add_argument("--data-root", type=Path, default=Path("data/Amazon"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/Amazon/sid_maps"))
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--sid-version", default="text")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--no-strict", action="store_true", help="Keep priority winner on conflicts.")
    args = parser.parse_args()

    categories = args.categories or discover_categories(args.data_root)
    if not categories:
        raise ValueError(f"No categories discovered under {args.data_root / 'index'}")

    strict = not args.no_strict
    manifest_path = args.manifest_path or args.output_dir / "experiment_manifest.json"
    category_entries: dict[str, dict[str, str]] = {}
    for category in categories:
        category_entries[category] = process_category(
            data_root=args.data_root,
            output_dir=args.output_dir,
            category=category,
            sid_version=args.sid_version,
            strict=strict,
        )

    update_manifest(manifest_path, args.sid_version, category_entries)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
