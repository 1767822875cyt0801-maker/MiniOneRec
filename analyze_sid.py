#!/usr/bin/env python3
"""Static SID quality diagnostics for MiniOneRec.

This script is read-only with respect to the source data and mapping files. It
only writes diagnostic reports under the requested analysis output directory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utils_sid import (
    load_json,
    load_json_set,
    normalize_sid,
    parse_sid_tokens,
    sid_prefixes,
)


RESERVED_METRICS = {
    "semantic_cohesion": None,
    "semantic_separation": None,
    "collision_semantic_similarity": None,
    "invalid_prediction_rate": "not_available",
    "sid_level_hr_ndcg": "not_available",
    "item_level_hr_ndcg": "not_available",
}

REQUIRED_MANIFEST_KEYS = [
    "item2sid",
    "sid2items",
    "valid_sid_set",
    "item_mapping",
    "train_csv",
    "valid_csv",
    "test_csv",
    "index",
    "info",
]


def item_sort_key(value: Any) -> tuple[int, int | str]:
    value_str = str(value)
    try:
        return (0, int(value_str))
    except ValueError:
        return (1, value_str)


def rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator) / float(denominator)


def safe_mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def safe_std(values: list[int | float]) -> float:
    if not values:
        return 0.0
    mean_value = safe_mean(values)
    return math.sqrt(sum((float(value) - mean_value) ** 2 for value in values) / len(values))


def percentile(values: list[int | float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[int(pos)]
    weight = pos - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def gini(values: list[int | float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    total = sum(sorted_values)
    if total == 0:
        return 0.0
    n = len(sorted_values)
    weighted_sum = sum((idx + 1) * value for idx, value in enumerate(sorted_values))
    return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n


def parse_list(value: Any) -> tuple[list[Any], str | None]:
    if isinstance(value, list):
        return value, None

    value_str = "" if value is None else str(value).strip()
    if not value_str:
        return [], None

    try:
        parsed = ast.literal_eval(value_str)
    except (SyntaxError, ValueError) as exc:
        return [], str(exc)

    if isinstance(parsed, (list, tuple)):
        return list(parsed), None
    return [parsed], None


def read_csv_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    errors: list[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"item_id", "item_sid", "history_item_id", "history_item_sid"}
        missing_cols = sorted(required - set(reader.fieldnames or []))
        if missing_cols:
            errors.append(f"{csv_path} missing required columns: {missing_cols}")
        return list(reader), errors


def json_for_csv(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def normalize_item2sid(raw: dict[Any, Any]) -> dict[str, str]:
    return {
        str(item_id): normalize_sid(str(sid))
        for item_id, sid in raw.items()
    }


def normalize_sid2items(raw: dict[Any, Any]) -> dict[str, list[str]]:
    sid2items: dict[str, list[str]] = {}
    for sid, items in raw.items():
        normalized_sid = normalize_sid(str(sid))
        if isinstance(items, list):
            item_list = [str(item_id) for item_id in items]
        else:
            item_list = [str(items)]
        sid2items[normalized_sid] = sorted(item_list, key=item_sort_key)
    return dict(sorted(sid2items.items()))


def load_index_mapping(index_path: Path) -> dict[str, str]:
    raw = load_json(index_path)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict index JSON: {index_path}")

    mapping: dict[str, str] = {}
    for item_id, sid_value in raw.items():
        if isinstance(sid_value, list):
            sid = "".join(str(token).strip() for token in sid_value)
        else:
            sid = str(sid_value)
        mapping[str(item_id)] = normalize_sid(sid)
    return mapping


def load_info_mapping(info_path: Path) -> tuple[dict[str, str], int]:
    mapping: dict[str, str] = {}
    malformed_rows = 0
    with open(info_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                malformed_rows += 1
                continue
            mapping[str(parts[2]).strip()] = normalize_sid(parts[0])
    return mapping, malformed_rows


def mapping_conflict_sample(
    left: dict[str, str],
    right: dict[str, str],
    left_name: str,
    right_name: str,
) -> tuple[int, list[dict[str, str]]]:
    conflicts = []
    shared = sorted(set(left) & set(right), key=item_sort_key)
    for item_id in shared:
        if left[item_id] != right[item_id]:
            conflicts.append({
                "item_id": item_id,
                left_name: left[item_id],
                right_name: right[item_id],
            })
    return len(conflicts), conflicts[:20]


def compute_source_consistency(
    item2sid: dict[str, str],
    index_map: dict[str, str],
    info_map: dict[str, str],
    malformed_info_rows: int,
) -> dict[str, Any]:
    index_item2sid_conflicts, index_item2sid_sample = mapping_conflict_sample(
        index_map, item2sid, "index_sid", "item2sid"
    )
    info_item2sid_conflicts, info_item2sid_sample = mapping_conflict_sample(
        info_map, item2sid, "info_sid", "item2sid"
    )
    index_info_conflicts, index_info_sample = mapping_conflict_sample(
        index_map, info_map, "index_sid", "info_sid"
    )
    return {
        "index_items": len(index_map),
        "info_items": len(info_map),
        "item2sid_items": len(item2sid),
        "index_item2sid_shared_items": len(set(index_map) & set(item2sid)),
        "info_item2sid_shared_items": len(set(info_map) & set(item2sid)),
        "index_info_shared_items": len(set(index_map) & set(info_map)),
        "index_item2sid_conflicts_count": index_item2sid_conflicts,
        "index_item2sid_conflicts_sample": index_item2sid_sample,
        "info_item2sid_conflicts_count": info_item2sid_conflicts,
        "info_item2sid_conflicts_sample": info_item2sid_sample,
        "index_info_conflicts_count": index_info_conflicts,
        "index_info_conflicts_sample": index_info_sample,
        "malformed_info_rows": malformed_info_rows,
    }


def sid_bucket_sizes(item2sid: dict[str, str], sid2items: dict[str, list[str]]) -> dict[str, int]:
    sizes = {sid: len(items) for sid, items in sid2items.items()}
    return {item_id: sizes.get(sid, 0) for item_id, sid in item2sid.items()}


def compute_basic_uniqueness(
    item2sid: dict[str, str],
    sid2items: dict[str, list[str]],
) -> dict[str, Any]:
    bucket_sizes = [len(items) for items in sid2items.values()]
    num_items = len(item2sid)
    num_unique_sid = len(sid2items)
    num_collision_groups = sum(1 for size in bucket_sizes if size > 1)
    collided_items_count = sum(size for size in bucket_sizes if size > 1)

    return {
        "num_items": num_items,
        "num_unique_sid": num_unique_sid,
        "collision_rate": 1.0 - rate(num_unique_sid, num_items),
        "collided_item_rate": rate(collided_items_count, num_items),
        "collided_items_count": collided_items_count,
        "num_collision_groups": num_collision_groups,
        "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
        "avg_bucket_size": safe_mean(bucket_sizes),
        "p95_bucket_size": percentile(bucket_sizes, 95),
    }


def compute_legality(
    item2sid: dict[str, str],
    sid2items: dict[str, list[str]],
    valid_sid_set: set[str],
) -> dict[str, Any]:
    bad_item_sids = {
        item_id: sid
        for item_id, sid in item2sid.items()
        if not parse_sid_tokens(sid)
    }
    return {
        "bad_item_sids_count": len(bad_item_sids),
        "bad_item_sids_sample": list(sorted(bad_item_sids.items(), key=lambda item: item_sort_key(item[0])))[:20],
        "valid_sid_set_equals_sid2items_keys": valid_sid_set == set(sid2items.keys()),
    }


def compute_split_stats(csv_path: Path, item2sid: dict[str, str]) -> dict[str, Any]:
    rows, read_errors = read_csv_rows(csv_path)
    target_sid_ok = 0
    all_sid_ok = 0
    all_sid_total = 0
    history_sid_ok_rows = 0
    history_len_ok = 0
    csv_item_ids: set[str] = set()
    missing_items: set[str] = set()
    mapping_conflicts: list[dict[str, str]] = []
    parse_error_samples: list[dict[str, str]] = []
    bad_history_parse_rows = 0

    for row_index, row in enumerate(rows, start=2):
        item_id = str(row.get("item_id", "")).strip()
        item_sid = normalize_sid(row.get("item_sid", ""))
        if item_id:
            csv_item_ids.add(item_id)
            if item_id not in item2sid:
                missing_items.add(item_id)
            elif item_sid and item2sid[item_id] != item_sid:
                mapping_conflicts.append({
                    "row": str(row_index),
                    "item_id": item_id,
                    "csv_sid": item_sid,
                    "map_sid": item2sid[item_id],
                })

        valid_target = bool(parse_sid_tokens(item_sid))
        target_sid_ok += int(valid_target)
        all_sid_ok += int(valid_target)
        all_sid_total += 1

        history_ids, history_id_error = parse_list(row.get("history_item_id", "[]"))
        history_sids, history_sid_error = parse_list(row.get("history_item_sid", "[]"))
        if history_id_error or history_sid_error:
            bad_history_parse_rows += 1
            if len(parse_error_samples) < 20:
                parse_error_samples.append({
                    "row": str(row_index),
                    "history_item_id_error": history_id_error or "",
                    "history_item_sid_error": history_sid_error or "",
                })
            history_sid_ok_rows += 0
            continue

        if len(history_ids) == len(history_sids):
            history_len_ok += 1

        history_valid = True
        for history_id, history_sid_raw in zip(history_ids, history_sids):
            history_id = str(history_id).strip()
            history_sid = normalize_sid(str(history_sid_raw))
            if history_id:
                csv_item_ids.add(history_id)
                if history_id not in item2sid:
                    missing_items.add(history_id)
                elif history_sid and item2sid[history_id] != history_sid:
                    mapping_conflicts.append({
                        "row": str(row_index),
                        "item_id": history_id,
                        "csv_sid": history_sid,
                        "map_sid": item2sid[history_id],
                    })

            valid_history_sid = bool(parse_sid_tokens(history_sid))
            history_valid = history_valid and valid_history_sid
            all_sid_ok += int(valid_history_sid)
            all_sid_total += 1

        history_sid_ok_rows += int(history_valid)

    covered_items = sum(1 for item_id in csv_item_ids if item_id in item2sid)
    return {
        "rows": len(rows),
        "sid_parse_rate": rate(all_sid_ok, all_sid_total),
        "item_sid_parse_rate": rate(target_sid_ok, len(rows)),
        "history_sid_parse_rate": rate(history_sid_ok_rows, len(rows)),
        "csv_item_coverage_rate": rate(covered_items, len(csv_item_ids)),
        "history_len_consistency_rate": rate(history_len_ok, len(rows)),
        "mapping_conflicts_count": len(mapping_conflicts),
        "mapping_conflicts_sample": mapping_conflicts[:20],
        "missing_items_count": len(missing_items),
        "missing_items_sample": sorted(missing_items, key=item_sort_key)[:20],
        "unique_csv_items": len(csv_item_ids),
        "bad_history_parse_rows": bad_history_parse_rows,
        "parse_error_samples": parse_error_samples,
        "errors": read_errors,
    }


def compute_level_stats(
    item2sid: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    level_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for sid in item2sid.values():
        for level_index, token in enumerate(parse_sid_tokens(sid), start=1):
            level_counts[level_index][token] += 1

    token_rows: list[dict[str, Any]] = []
    level_summaries: list[dict[str, Any]] = []
    for level in sorted(level_counts):
        counts = level_counts[level]
        count_values = list(counts.values())
        total = sum(count_values)
        mean_count = safe_mean(count_values)
        std_count = safe_std(count_values)
        hotspot_threshold = mean_count + 2.0 * std_count
        entropy = 0.0
        for count in count_values:
            p = rate(count, total)
            if p > 0:
                entropy -= p * math.log2(p)
        num_tokens = len(counts)
        normalized_entropy = entropy / math.log2(num_tokens) if num_tokens > 1 else 1.0
        summary = {
            "level": level,
            "num_tokens": num_tokens,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "perplexity": 2.0 ** entropy,
            "min_count": min(count_values) if count_values else 0,
            "max_count": max(count_values) if count_values else 0,
            "mean_count": mean_count,
            "p95_count": percentile(count_values, 95),
            "gini": gini(count_values),
            "num_hotspots": sum(1 for count in count_values if count > hotspot_threshold),
        }
        level_summaries.append(summary)

        for token, count in sorted(counts.items()):
            p = rate(count, total)
            entropy_contribution = -p * math.log2(p) if p > 0 else 0.0
            token_rows.append({
                "level": level,
                "token": token,
                "count": count,
                "rate": p,
                "entropy_contribution": entropy_contribution,
                "is_hotspot": count > hotspot_threshold,
                "num_tokens": summary["num_tokens"],
                "entropy": summary["entropy"],
                "normalized_entropy": summary["normalized_entropy"],
                "perplexity": summary["perplexity"],
                "min_count": summary["min_count"],
                "max_count": summary["max_count"],
                "mean_count": summary["mean_count"],
                "p95_count": summary["p95_count"],
                "gini": summary["gini"],
            })

    return token_rows, {
        "levels": level_summaries,
        "normalized_entropy_by_level": [
            summary["normalized_entropy"] for summary in level_summaries
        ],
    }


def compute_prefix_stats(
    item2sid: dict[str, str],
    item_bucket_sizes: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prefix_items: dict[tuple[int, str], list[str]] = defaultdict(list)
    prefix_children: dict[tuple[int, str], set[str]] = defaultdict(set)
    item_tokens: dict[str, list[str]] = {}

    for item_id, sid in item2sid.items():
        tokens = parse_sid_tokens(sid)
        item_tokens[item_id] = tokens
        prefixes = sid_prefixes(sid)
        for index, prefix in enumerate(prefixes, start=1):
            key = (index, prefix)
            prefix_items[key].append(item_id)
            if index < len(tokens):
                prefix_children[key].add(tokens[index])

    rows_by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (level, prefix), items in prefix_items.items():
        bucket_values = [item_bucket_sizes.get(item_id, 0) for item_id in items]
        collided_count = sum(1 for bucket_size in bucket_values if bucket_size > 1)
        rows_by_level[level].append({
            "level": level,
            "prefix": prefix,
            "item_count": len(items),
            "unique_children": len(prefix_children.get((level, prefix), set())),
            "fanout": len(prefix_children.get((level, prefix), set())),
            "bucket_size_mean": safe_mean(bucket_values),
            "bucket_size_max": max(bucket_values) if bucket_values else 0,
            "collision_pressure": rate(collided_count, len(items)),
        })

    prefix_rows: list[dict[str, Any]] = []
    prefix_summary_levels: list[dict[str, Any]] = []
    for level in sorted(rows_by_level):
        level_rows = rows_by_level[level]
        item_counts = [row["item_count"] for row in level_rows]
        mean_item_count = safe_mean(item_counts)
        std_item_count = safe_std(item_counts)
        hotspot_threshold = mean_item_count + 2.0 * std_item_count
        for row in sorted(level_rows, key=lambda value: value["prefix"]):
            row["is_hotspot"] = row["item_count"] > hotspot_threshold
            prefix_rows.append(row)

        prefix_summary_levels.append({
            "level": level,
            "num_prefixes": len(level_rows),
            "mean_item_count": mean_item_count,
            "max_item_count": max(item_counts) if item_counts else 0,
            "p95_item_count": percentile(item_counts, 95),
            "mean_fanout": safe_mean([row["fanout"] for row in level_rows]),
            "max_fanout": max([row["fanout"] for row in level_rows]) if level_rows else 0,
            "max_collision_pressure": max([row["collision_pressure"] for row in level_rows]) if level_rows else 0.0,
            "num_hotspots": sum(1 for row in level_rows if row["item_count"] > hotspot_threshold),
        })

    return prefix_rows, {"levels": prefix_summary_levels}


def compute_train_popularity(
    train_csv: Path,
    item2sid: dict[str, str],
    item_bucket_sizes: dict[str, int],
    head_ratio: float,
    tail_ratio: float,
) -> dict[str, Any]:
    target_counts: Counter[str] = Counter()
    history_counts: Counter[str] = Counter()
    parse_error_samples: list[dict[str, str]] = []

    rows, read_errors = read_csv_rows(train_csv)
    for row_index, row in enumerate(rows, start=2):
        item_id = str(row.get("item_id", "")).strip()
        if item_id:
            target_counts[item_id] += 1

        history_ids, history_error = parse_list(row.get("history_item_id", "[]"))
        if history_error:
            if len(parse_error_samples) < 20:
                parse_error_samples.append({
                    "row": str(row_index),
                    "history_item_id_error": history_error,
                })
            continue
        for history_id in history_ids:
            history_id = str(history_id).strip()
            if history_id:
                history_counts[history_id] += 1

    item_counts: dict[str, dict[str, int]] = {}
    for item_id in item2sid:
        target_count = int(target_counts.get(item_id, 0))
        history_count = int(history_counts.get(item_id, 0))
        item_counts[item_id] = {
            "train_target_count": target_count,
            "train_history_count": history_count,
            "train_total_interaction_count": target_count + history_count,
        }

    sorted_items = sorted(
        item2sid.keys(),
        key=lambda item_id: (
            -item_counts[item_id]["train_total_interaction_count"],
            item_sort_key(item_id),
        ),
    )
    total_items = len(sorted_items)
    head_n = int(total_items * head_ratio)
    tail_n = int(total_items * tail_ratio)
    if head_ratio > 0 and total_items > 0:
        head_n = max(1, head_n)
    if tail_ratio > 0 and total_items > 0:
        tail_n = max(1, tail_n)
    if head_n + tail_n > total_items:
        tail_n = max(0, total_items - head_n)

    head_items = set(sorted_items[:head_n])
    tail_items = set(sorted_items[total_items - tail_n:]) if tail_n else set()
    popularity_group_by_item: dict[str, str] = {}
    for item_id in item2sid:
        if item_id in head_items:
            popularity_group_by_item[item_id] = "head"
        elif item_id in tail_items:
            popularity_group_by_item[item_id] = "tail"
        else:
            popularity_group_by_item[item_id] = "mid"

    group_summary: dict[str, dict[str, Any]] = {}
    for group_name in ["head", "mid", "tail"]:
        group_items = [
            item_id
            for item_id, group in popularity_group_by_item.items()
            if group == group_name
        ]
        collided_items = [
            item_id for item_id in group_items if item_bucket_sizes.get(item_id, 0) > 1
        ]
        unique_sid_count = len({item2sid[item_id] for item_id in group_items})
        group_summary[group_name] = {
            "num_items": len(group_items),
            "collision_rate": rate(len(collided_items), len(group_items)),
            "within_group_collision_rate": 1.0 - rate(unique_sid_count, len(group_items)),
            "train_target_count_sum": sum(item_counts[item_id]["train_target_count"] for item_id in group_items),
            "train_history_count_sum": sum(item_counts[item_id]["train_history_count"] for item_id in group_items),
            "train_total_interaction_count_sum": sum(item_counts[item_id]["train_total_interaction_count"] for item_id in group_items),
        }

    target_total = sum(target_counts.values())
    history_total = sum(history_counts.values())
    total_interaction = target_total + history_total
    collided_item_ids = {item_id for item_id, size in item_bucket_sizes.items() if size > 1}
    target_collided = sum(target_counts.get(item_id, 0) for item_id in collided_item_ids)
    history_collided = sum(history_counts.get(item_id, 0) for item_id in collided_item_ids)

    weighted_collision = {
        "target_interaction_weighted_collision_rate": rate(target_collided, target_total),
        "history_interaction_weighted_collision_rate": rate(history_collided, history_total),
        "total_interaction_weighted_collision_rate": rate(target_collided + history_collided, total_interaction),
        "target_interactions": int(target_total),
        "history_interactions": int(history_total),
        "total_interactions": int(total_interaction),
        "target_collided_interactions": int(target_collided),
        "history_collided_interactions": int(history_collided),
    }

    return {
        "item_counts": item_counts,
        "popularity_group_by_item": popularity_group_by_item,
        "popularity_summary": {
            "head_ratio": head_ratio,
            "tail_ratio": tail_ratio,
            "head_items": len(head_items),
            "tail_items": len(tail_items),
            "groups": group_summary,
            "errors": read_errors,
            "parse_error_samples": parse_error_samples,
        },
        "interaction_weighted_collision": weighted_collision,
    }


def build_collision_group_rows(
    sid2items: dict[str, list[str]],
    item_mapping: dict[str, Any],
    popularity: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    item_counts = popularity["item_counts"]
    popularity_group_by_item = popularity["popularity_group_by_item"]

    for sid, items in sorted(sid2items.items()):
        if len(items) <= 1:
            continue
        titles = [
            str(item_mapping.get(item_id, {}).get("title", ""))
            for item_id in items
        ]
        groups = [
            popularity_group_by_item.get(item_id, "mid")
            for item_id in items
        ]
        total_interactions = sum(
            item_counts.get(item_id, {}).get("train_total_interaction_count", 0)
            for item_id in items
        )
        rows.append({
            "sid": sid,
            "bucket_size": len(items),
            "item_ids": json_for_csv(items),
            "titles": json_for_csv(titles),
            "popularity_groups": json_for_csv(groups),
            "train_total_interaction_count_sum": total_interactions,
        })
    return rows


def build_item_diagnostics_rows(
    item2sid: dict[str, str],
    item_mapping: dict[str, Any],
    item_bucket_sizes: dict[str, int],
    popularity: dict[str, Any],
) -> list[dict[str, Any]]:
    item_counts = popularity["item_counts"]
    popularity_group_by_item = popularity["popularity_group_by_item"]
    rows: list[dict[str, Any]] = []
    for item_id in sorted(item2sid.keys(), key=item_sort_key):
        sid = item2sid[item_id]
        counts = item_counts.get(item_id, {})
        rows.append({
            "item_id": item_id,
            "sid": sid,
            "sid_tokens": json_for_csv(parse_sid_tokens(sid)),
            "title": str(item_mapping.get(item_id, {}).get("title", "")),
            "collision_group_size": item_bucket_sizes.get(item_id, 0),
            "is_collided": item_bucket_sizes.get(item_id, 0) > 1,
            "train_target_count": counts.get("train_target_count", 0),
            "train_history_count": counts.get("train_history_count", 0),
            "train_total_interaction_count": counts.get("train_total_interaction_count", 0),
            "popularity_group": popularity_group_by_item.get(item_id, "mid"),
        })
    return rows


def scalar_metrics(summary: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    category = summary["category"]
    sid_version = summary["sid_version"]

    def add(metric: str, value: Any) -> None:
        if isinstance(value, (dict, list)):
            value_str = json_for_csv(value)
        elif value is None:
            value_str = ""
        else:
            value_str = str(value)
        rows.append({
            "category": category,
            "sid_version": sid_version,
            "metric": metric,
            "value": value_str,
        })

    for key, value in summary["basic_uniqueness"].items():
        add(f"basic_uniqueness.{key}", value)
    for key, value in summary["legality"].items():
        if not key.endswith("_sample"):
            add(f"legality.{key}", value)
    for key, value in summary["source_consistency"].items():
        if not key.endswith("_sample"):
            add(f"source_consistency.{key}", value)
    for split, stats in summary["split_stats"].items():
        for key, value in stats.items():
            if key.endswith("_sample") or key == "errors":
                continue
            add(f"split_stats.{split}.{key}", value)
    for level_summary in summary["level_summary"]["levels"]:
        level = level_summary["level"]
        for key, value in level_summary.items():
            if key != "level":
                add(f"level_summary.level_{level}.{key}", value)
    for level_summary in summary["prefix_summary"]["levels"]:
        level = level_summary["level"]
        for key, value in level_summary.items():
            if key != "level":
                add(f"prefix_summary.level_{level}.{key}", value)
    for group, stats in summary["popularity_summary"]["groups"].items():
        for key, value in stats.items():
            add(f"popularity_summary.{group}.{key}", value)
    for key, value in summary["interaction_weighted_collision"].items():
        add(f"interaction_weighted_collision.{key}", value)
    for key, value in summary["reserved_metrics"].items():
        add(f"reserved_metrics.{key}", value)
    return rows


def overall_row(summary: dict[str, Any]) -> dict[str, Any]:
    basic = summary["basic_uniqueness"]
    weighted = summary["interaction_weighted_collision"]
    return {
        "category": summary["category"],
        "sid_version": summary["sid_version"],
        "num_items": basic["num_items"],
        "num_unique_sid": basic["num_unique_sid"],
        "collision_rate": basic["collision_rate"],
        "collided_item_rate": basic["collided_item_rate"],
        "num_collision_groups": basic["num_collision_groups"],
        "max_bucket_size": basic["max_bucket_size"],
        "avg_bucket_size": basic["avg_bucket_size"],
        "p95_bucket_size": basic["p95_bucket_size"],
        "level_normalized_entropy": json_for_csv(summary["level_summary"]["normalized_entropy_by_level"]),
        "target_interaction_weighted_collision_rate": weighted["target_interaction_weighted_collision_rate"],
        "history_interaction_weighted_collision_rate": weighted["history_interaction_weighted_collision_rate"],
        "total_interaction_weighted_collision_rate": weighted["total_interaction_weighted_collision_rate"],
    }


def print_category_summary(summary: dict[str, Any]) -> None:
    basic = summary["basic_uniqueness"]
    weighted = summary["interaction_weighted_collision"]
    print(
        f"[{summary['category']}] "
        f"num_items={basic['num_items']} "
        f"num_unique_sid={basic['num_unique_sid']} "
        f"collision_rate={basic['collision_rate']:.6f} "
        f"collided_item_rate={basic['collided_item_rate']:.6f} "
        f"max_bucket_size={basic['max_bucket_size']} "
        f"level_normalized_entropy={json_for_csv(summary['level_summary']['normalized_entropy_by_level'])} "
        f"target_iw_collision={weighted['target_interaction_weighted_collision_rate']:.6f} "
        f"history_iw_collision={weighted['history_interaction_weighted_collision_rate']:.6f} "
        f"total_iw_collision={weighted['total_interaction_weighted_collision_rate']:.6f}"
    )


def load_category_inputs(entry: dict[str, str]) -> dict[str, Path]:
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in entry]
    if missing:
        raise KeyError(f"Manifest entry missing required keys: {missing}")
    return {key: Path(entry[key]) for key in REQUIRED_MANIFEST_KEYS}


def process_category(
    category: str,
    sid_version: str,
    manifest_entry: dict[str, str],
    output_root: Path,
    head_ratio: float,
    tail_ratio: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_paths = load_category_inputs(manifest_entry)
    errors: list[str] = []

    item2sid = normalize_item2sid(load_json(input_paths["item2sid"]))
    sid2items = normalize_sid2items(load_json(input_paths["sid2items"]))
    valid_sid_set = {normalize_sid(sid) for sid in load_json_set(input_paths["valid_sid_set"])}
    raw_item_mapping = load_json(input_paths["item_mapping"])
    item_mapping = {str(item_id): value for item_id, value in raw_item_mapping.items()}
    index_map = load_index_mapping(input_paths["index"])
    info_map, malformed_info_rows = load_info_mapping(input_paths["info"])

    basic = compute_basic_uniqueness(item2sid, sid2items)
    legality = compute_legality(item2sid, sid2items, valid_sid_set)
    source_consistency = compute_source_consistency(
        item2sid,
        index_map,
        info_map,
        malformed_info_rows,
    )
    item_bucket_sizes = sid_bucket_sizes(item2sid, sid2items)
    split_stats = {
        split: compute_split_stats(input_paths[f"{split}_csv"], item2sid)
        for split in ["train", "valid", "test"]
    }
    for split, stats in split_stats.items():
        for error in stats.get("errors", []):
            errors.append(f"{split}: {error}")

    level_token_rows, level_summary = compute_level_stats(item2sid)
    prefix_rows, prefix_summary = compute_prefix_stats(item2sid, item_bucket_sizes)
    popularity = compute_train_popularity(
        input_paths["train_csv"],
        item2sid,
        item_bucket_sizes,
        head_ratio,
        tail_ratio,
    )
    for error in popularity["popularity_summary"].get("errors", []):
        errors.append(f"train_popularity: {error}")

    category_output_dir = output_root / sid_version / category
    output_paths = {
        "sid_quality_summary": category_output_dir / "sid_quality_summary.json",
        "metrics_summary": category_output_dir / "metrics_summary.csv",
        "sid_collision_groups": category_output_dir / "sid_collision_groups.csv",
        "level_token_stats": category_output_dir / "level_token_stats.csv",
        "prefix_stats": category_output_dir / "prefix_stats.csv",
        "item_sid_diagnostics": category_output_dir / "item_sid_diagnostics.csv",
    }

    summary = {
        "category": category,
        "sid_version": sid_version,
        "inputs": {key: path.as_posix() for key, path in input_paths.items()},
        "outputs": {key: path.as_posix() for key, path in output_paths.items()},
        "errors": errors,
        "basic_uniqueness": basic,
        "legality": legality,
        "source_consistency": source_consistency,
        "split_stats": split_stats,
        "level_summary": level_summary,
        "prefix_summary": prefix_summary,
        "popularity_summary": popularity["popularity_summary"],
        "interaction_weighted_collision": popularity["interaction_weighted_collision"],
        "reserved_metrics": dict(RESERVED_METRICS),
    }

    write_json(summary, output_paths["sid_quality_summary"])
    write_csv(
        output_paths["metrics_summary"],
        ["category", "sid_version", "metric", "value"],
        scalar_metrics(summary),
    )
    write_csv(
        output_paths["sid_collision_groups"],
        ["sid", "bucket_size", "item_ids", "titles", "popularity_groups", "train_total_interaction_count_sum"],
        build_collision_group_rows(sid2items, item_mapping, popularity),
    )
    write_csv(
        output_paths["level_token_stats"],
        [
            "level",
            "token",
            "count",
            "rate",
            "entropy_contribution",
            "is_hotspot",
            "num_tokens",
            "entropy",
            "normalized_entropy",
            "perplexity",
            "min_count",
            "max_count",
            "mean_count",
            "p95_count",
            "gini",
        ],
        level_token_rows,
    )
    write_csv(
        output_paths["prefix_stats"],
        [
            "level",
            "prefix",
            "item_count",
            "unique_children",
            "fanout",
            "bucket_size_mean",
            "bucket_size_max",
            "collision_pressure",
            "is_hotspot",
        ],
        prefix_rows,
    )
    write_csv(
        output_paths["item_sid_diagnostics"],
        [
            "item_id",
            "sid",
            "sid_tokens",
            "title",
            "collision_group_size",
            "is_collided",
            "train_target_count",
            "train_history_count",
            "train_total_interaction_count",
            "popularity_group",
        ],
        build_item_diagnostics_rows(item2sid, item_mapping, item_bucket_sizes, popularity),
    )

    return summary, overall_row(summary)


def error_summary(
    category: str,
    sid_version: str,
    manifest_entry: dict[str, str],
    output_root: Path,
    error: Exception,
) -> dict[str, Any]:
    category_output_dir = output_root / sid_version / category
    summary_path = category_output_dir / "sid_quality_summary.json"
    summary = {
        "category": category,
        "sid_version": sid_version,
        "inputs": manifest_entry,
        "outputs": {"sid_quality_summary": summary_path.as_posix()},
        "errors": [str(error)],
        "basic_uniqueness": "not_available",
        "legality": "not_available",
        "split_stats": "not_available",
        "level_summary": "not_available",
        "prefix_summary": "not_available",
        "popularity_summary": "not_available",
        "interaction_weighted_collision": "not_available",
        "reserved_metrics": dict(RESERVED_METRICS),
    }
    write_json(summary, summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze static SID quality diagnostics.")
    parser.add_argument("--manifest", type=Path, default=Path("data/Amazon/sid_maps/experiment_manifest.json"))
    parser.add_argument("--sid-version", default="text")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/Amazon/sid_maps/analysis"))
    parser.add_argument("--head-ratio", type=float, default=0.2)
    parser.add_argument("--tail-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    sid_versions = manifest.get("sid_versions", {})
    if args.sid_version not in sid_versions:
        raise KeyError(f"sid_version={args.sid_version!r} not found in {args.manifest}")

    manifest_categories = sid_versions[args.sid_version]
    categories = args.categories or sorted(manifest_categories)
    overall_rows: list[dict[str, Any]] = []

    for category in categories:
        if category not in manifest_categories:
            print(f"[{category}] skipped: not found under sid_version={args.sid_version}")
            continue
        try:
            summary, row = process_category(
                category=category,
                sid_version=args.sid_version,
                manifest_entry=manifest_categories[category],
                output_root=args.output_dir,
                head_ratio=args.head_ratio,
                tail_ratio=args.tail_ratio,
            )
            overall_rows.append(row)
            print_category_summary(summary)
        except Exception as exc:  # Keep other categories diagnosable.
            error_summary(category, args.sid_version, manifest_categories[category], args.output_dir, exc)
            print(f"[{category}] error: {exc}")

    overall_path = args.output_dir / args.sid_version / "sid_quality_overall_summary.csv"
    write_csv(
        overall_path,
        [
            "category",
            "sid_version",
            "num_items",
            "num_unique_sid",
            "collision_rate",
            "collided_item_rate",
            "num_collision_groups",
            "max_bucket_size",
            "avg_bucket_size",
            "p95_bucket_size",
            "level_normalized_entropy",
            "target_interaction_weighted_collision_rate",
            "history_interaction_weighted_collision_rate",
            "total_interaction_weighted_collision_rate",
        ],
        overall_rows,
    )
    print(f"Wrote overall summary: {overall_path}")


if __name__ == "__main__":
    main()
