#!/usr/bin/env python3
"""Audit Amazon feedback semantics for raw and processed MiniOneRec CSVs."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


USER_ALIASES = ["user_id", "reviewerID", "reviewer_id", "user", "uid"]
ITEM_ALIASES = ["item_id", "asin", "item", "iid", "product_id"]
HISTORY_ALIASES = ["history_item_id", "history_items", "history"]
RATING_ALIASES = ["rating", "overall", "score", "stars", "rate"]
TIMESTAMP_ALIASES = ["timestamp", "unixReviewTime", "reviewTime", "time"]
ACTION_ALIASES = ["action_type", "event_type", "behavior", "event"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Amazon raw/processed feedback semantics.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--valid-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--raw-interactions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-raw-rows", type=int, default=0, help="0 means scan all raw rows.")
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def choose_alias(columns: Iterable[str], aliases: list[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        raise ValueError(f"cannot parse list: {text[:80]}")
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def inventory_processed_csv(path: Path, split: str) -> tuple[dict[str, Any], set[str], set[str], Counter[str], list[Any]]:
    columns, rows = read_csv_rows(path)
    null_counts = {column: 0 for column in columns}
    users: set[str] = set()
    targets: set[str] = set()
    all_items: set[str] = set()
    history_parse_errors = 0
    history_lengths: list[int] = []
    rating_counter: Counter[str] = Counter()
    timestamps: list[Any] = []

    user_col = choose_alias(columns, USER_ALIASES)
    item_col = choose_alias(columns, ITEM_ALIASES)
    history_col = choose_alias(columns, HISTORY_ALIASES)
    rating_col = choose_alias(columns, RATING_ALIASES)
    timestamp_col = choose_alias(columns, TIMESTAMP_ALIASES)
    action_col = choose_alias(columns, ACTION_ALIASES)

    for row in rows:
        for column in columns:
            if row.get(column) in (None, ""):
                null_counts[column] += 1
        if user_col and row.get(user_col):
            users.add(str(row[user_col]))
        if item_col and row.get(item_col):
            item_id = str(row[item_col])
            targets.add(item_id)
            all_items.add(item_id)
        if history_col:
            try:
                history = parse_list(row.get(history_col))
            except ValueError:
                history_parse_errors += 1
                history = []
            history_lengths.append(len(history))
            for item in history:
                all_items.add(str(item))
        if rating_col:
            value = row.get(rating_col)
            rating_counter[str(value)] += 1
        if timestamp_col and row.get(timestamp_col):
            timestamps.append(row[timestamp_col])

    info = {
        "path": path.as_posix(),
        "split": split,
        "columns": columns,
        "row_count": len(rows),
        "null_counts": null_counts,
        "field_aliases": {
            "user": user_col,
            "item": item_col,
            "history": history_col,
            "rating": rating_col,
            "timestamp": timestamp_col,
            "action_type": action_col,
        },
        "has_user_field": user_col is not None,
        "has_item_field": item_col is not None,
        "has_history_field": history_col is not None,
        "has_rating_field": rating_col is not None,
        "has_timestamp_field": timestamp_col is not None,
        "has_action_type_field": action_col is not None,
        "unique_users": len(users),
        "unique_target_items": len(targets),
        "unique_all_items": len(all_items),
        "history_parse_errors": history_parse_errors,
        "history_length_min": min(history_lengths) if history_lengths else 0,
        "history_length_max": max(history_lengths) if history_lengths else 0,
        "history_length_mean": (sum(history_lengths) / len(history_lengths)) if history_lengths else 0.0,
        "timestamp_min_raw": min(timestamps) if timestamps else None,
        "timestamp_max_raw": max(timestamps) if timestamps else None,
    }
    return info, users, targets, rating_counter, timestamps


def raw_rows(path: Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as f:
        first = f.readline()
        if not first:
            return
        f.seek(0)
        path_name = path.name.lower()
        if path_name.endswith(".csv") or path_name.endswith(".csv.gz") or path_name.endswith(".tsv") or path_name.endswith(".tsv.gz"):
            dialect = "excel-tab" if path_name.endswith(".tsv") or path_name.endswith(".tsv.gz") else "excel"
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                yield dict(row)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def inventory_raw(path: Path | None, max_rows: int = 0) -> tuple[dict[str, Any], Counter[str], dict[tuple[str, str], list[float]], Counter[str]]:
    if path is None:
        return (
            {
                "available": False,
                "path": None,
                "reason": "raw_interactions_not_provided",
                "conclusion": "raw_unavailable_cannot_verify_rating",
            },
            Counter(),
            {},
            Counter(),
        )
    if not path.exists():
        return (
            {
                "available": False,
                "path": path.as_posix(),
                "reason": "raw_interactions_path_missing",
                "conclusion": "raw_unavailable_cannot_verify_rating",
            },
            Counter(),
            {},
            Counter(),
        )

    columns: set[str] = set()
    null_counts: Counter[str] = Counter()
    row_count = 0
    rating_distribution: Counter[str] = Counter()
    action_distribution: Counter[str] = Counter()
    target_ratings: dict[tuple[str, str], list[float]] = defaultdict(list)
    sample_rows: list[dict[str, Any]] = []

    user_col = item_col = rating_col = timestamp_col = action_col = None
    for row in raw_rows(path):
        if max_rows and row_count >= max_rows:
            break
        row_count += 1
        if len(sample_rows) < 3:
            sample_rows.append(dict(row))
        columns.update(row.keys())
        if user_col is None:
            user_col = choose_alias(row.keys(), USER_ALIASES)
        if item_col is None:
            item_col = choose_alias(row.keys(), ITEM_ALIASES)
        if rating_col is None:
            rating_col = choose_alias(row.keys(), RATING_ALIASES)
        if timestamp_col is None:
            timestamp_col = choose_alias(row.keys(), TIMESTAMP_ALIASES)
        if action_col is None:
            action_col = choose_alias(row.keys(), ACTION_ALIASES)
        for key, value in row.items():
            if value in (None, ""):
                null_counts[str(key)] += 1
        if rating_col:
            rating_value = row.get(rating_col)
            rating_distribution[str(rating_value)] += 1
            parsed = parse_float(rating_value)
            if parsed is not None and user_col and item_col and row.get(user_col) and row.get(item_col):
                target_ratings[(str(row[user_col]), str(row[item_col]))].append(parsed)
        if action_col:
            action_distribution[str(row.get(action_col))] += 1

    field_aliases = {
        "user": user_col,
        "item": item_col,
        "rating": rating_col,
        "timestamp": timestamp_col,
        "action_type": action_col,
    }
    inventory = {
        "available": True,
        "path": path.as_posix(),
        "row_count_scanned": row_count,
        "columns": sorted(columns),
        "null_counts": dict(null_counts),
        "field_aliases": field_aliases,
        "has_rating_field": rating_col is not None,
        "has_timestamp_field": timestamp_col is not None,
        "has_action_type_field": action_col is not None,
        "sample_rows": sample_rows,
        "conclusion": "raw_contains_rating" if rating_col is not None else "raw_available_no_rating_field_detected",
    }
    return inventory, rating_distribution, dict(target_ratings), action_distribution


def target_rating_distribution(
    split_rows: dict[str, tuple[list[str], list[dict[str, str]]]],
    raw_rating_map: dict[tuple[str, str], list[float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, (columns, records) in split_rows.items():
        user_col = choose_alias(columns, USER_ALIASES)
        item_col = choose_alias(columns, ITEM_ALIASES)
        if not user_col or not item_col:
            continue
        counts: Counter[str] = Counter()
        matched = 0
        for record in records:
            key = (str(record.get(user_col, "")), str(record.get(item_col, "")))
            ratings = raw_rating_map.get(key, [])
            if not ratings:
                continue
            matched += 1
            counts[f"{ratings[-1]:g}"] += 1
        for rating, count in sorted(counts.items(), key=lambda kv: kv[0]):
            rows.append({"scope": f"processed_target_{split}", "rating": rating, "count": count})
        rows.append({"scope": f"processed_target_{split}", "rating": "__matched_rows__", "count": matched})
    return rows


def split_consistency(
    processed: dict[str, dict[str, Any]],
    users: dict[str, set[str]],
    targets: dict[str, set[str]],
) -> dict[str, Any]:
    train_users = users.get("train", set())
    train_items = targets.get("train", set())
    valid_users = users.get("valid", set())
    test_users = users.get("test", set())
    valid_items = targets.get("valid", set())
    test_items = targets.get("test", set())
    return {
        "row_counts": {split: info["row_count"] for split, info in processed.items()},
        "columns_by_split": {split: info["columns"] for split, info in processed.items()},
        "valid_users_not_in_train_count": len(valid_users - train_users),
        "test_users_not_in_train_count": len(test_users - train_users),
        "valid_target_items_not_in_train_targets_count": len(valid_items - train_items),
        "test_target_items_not_in_train_targets_count": len(test_items - train_items),
        "valid_test_target_overlap_count": len(valid_items & test_items),
        "time_boundaries_raw": {
            split: {
                "timestamp_min_raw": info.get("timestamp_min_raw"),
                "timestamp_max_raw": info.get("timestamp_max_raw"),
            }
            for split, info in processed.items()
        },
    }


def make_summary(category: str, raw_inventory: dict[str, Any], processed_inventory: dict[str, Any], semantics: dict[str, Any]) -> str:
    lines = [
        f"# Amazon Feedback Semantics Audit: {category}",
        "",
        "## Conclusions",
        "",
        f"- raw_rating_status: `{semantics['raw_rating_status']}`",
        f"- processed_rating_status: `{semantics['processed_rating_status']}`",
        f"- interaction_semantics: `{semantics['interaction_semantics']}`",
        "",
        "## Raw",
        "",
        f"- available: `{raw_inventory.get('available')}`",
        f"- path: `{raw_inventory.get('path')}`",
        f"- conclusion: `{raw_inventory.get('conclusion')}`",
        "",
        "## Processed CSV",
        "",
    ]
    for split, info in processed_inventory.items():
        lines.append(
            f"- {split}: rows=`{info['row_count']}`, rating_field=`{info['field_aliases']['rating']}`, "
            f"timestamp_field=`{info['field_aliases']['timestamp']}`, action_type_field=`{info['field_aliases']['action_type']}`"
        )
    lines.extend(
        [
            "",
            "This audit does not rewrite ratings into click/cart/buy labels.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {"train": args.train_csv, "valid": args.valid_csv, "test": args.test_csv}
    processed: dict[str, dict[str, Any]] = {}
    users: dict[str, set[str]] = {}
    targets: dict[str, set[str]] = {}
    processed_rating_counts: dict[str, Counter[str]] = {}
    split_raw_rows: dict[str, tuple[list[str], list[dict[str, str]]]] = {}

    for split, path in split_paths.items():
        columns, records = read_csv_rows(path)
        split_raw_rows[split] = (columns, records)
        info, split_users, split_targets, rating_counter, _ = inventory_processed_csv(path, split)
        processed[split] = info
        users[split] = split_users
        targets[split] = split_targets
        processed_rating_counts[split] = rating_counter

    raw_inventory, raw_rating_counts, raw_rating_map, action_counts = inventory_raw(args.raw_interactions, args.max_raw_rows)
    rating_rows: list[dict[str, Any]] = []
    for rating, count in sorted(raw_rating_counts.items(), key=lambda kv: kv[0]):
        rating_rows.append({"scope": "raw", "rating": rating, "count": count})
    for split, counts in processed_rating_counts.items():
        for rating, count in sorted(counts.items(), key=lambda kv: kv[0]):
            rating_rows.append({"scope": f"processed_{split}", "rating": rating, "count": count})
    rating_rows.extend(target_rating_distribution(split_raw_rows, raw_rating_map))
    if not rating_rows:
        rating_rows.append({"scope": "not_available", "rating": "not_available", "count": 0})

    processed_has_rating = any(info["has_rating_field"] for info in processed.values())
    raw_status = raw_inventory["conclusion"]
    processed_status = "processed_csv_contains_rating" if processed_has_rating else "processed_csv_missing_rating"
    semantics = {
        "category": args.category,
        "raw_rating_status": raw_status,
        "processed_rating_status": processed_status,
        "raw_available": raw_inventory.get("available", False),
        "raw_field_can_verify_rating": bool(raw_inventory.get("available") and raw_inventory.get("has_rating_field")),
        "interaction_semantics": (
            "raw rating is available, but current processed train/valid/test CSV does not carry rating"
            if raw_inventory.get("has_rating_field") and not processed_has_rating
            else "raw unavailable, cannot verify whether original interactions had rating"
            if not raw_inventory.get("available")
            else "see raw and processed field inventories"
        ),
        "rating_not_rewritten_as_actions": True,
        "action_type_distribution_raw": dict(action_counts),
    }
    split_report = split_consistency(processed, users, targets)

    write_json(args.output_dir / "raw_field_inventory.json", raw_inventory)
    write_json(args.output_dir / "processed_field_inventory.json", processed)
    write_csv(args.output_dir / "rating_distribution.csv", ["scope", "rating", "count"], rating_rows)
    write_json(args.output_dir / "split_consistency.json", split_report)
    write_json(args.output_dir / "interaction_semantics.json", semantics)
    with open(args.output_dir / "audit_summary.md", "w", encoding="utf-8") as f:
        f.write(make_summary(args.category, raw_inventory, processed, semantics))

    return {
        "raw_field_inventory": raw_inventory,
        "processed_field_inventory": processed,
        "split_consistency": split_report,
        "interaction_semantics": semantics,
        "output_dir": args.output_dir.as_posix(),
    }


def main() -> None:
    args = parse_args()
    result = run_audit(args)
    semantics = result["interaction_semantics"]
    print(
        "Feedback audit summary: "
        f"category={args.category} "
        f"raw_rating_status={semantics['raw_rating_status']} "
        f"processed_rating_status={semantics['processed_rating_status']} "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
