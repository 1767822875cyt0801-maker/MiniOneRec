#!/usr/bin/env python3
"""Diagnose calc_plus per-sample SID smoke outputs by useful groups."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


NOT_AVAILABLE = "not_available"
SID_RE = re.compile(r"<[^<>]+>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a per_sample_eval.csv from calc_plus.py.")
    parser.add_argument("--per-sample-eval", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--focus-k", type=int, default=20)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["section", "group", "metric", "value"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sid_len(sid: str) -> int:
    return len(SID_RE.findall(str(sid)))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", NOT_AVAILABLE}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def hit_at(row: dict[str, str], rank_col: str, k: int) -> bool | None:
    rank = as_float(row.get(rank_col))
    if rank is None:
        return False
    return 0 <= rank < k


def ndcg_at(row: dict[str, str], rank_col: str, k: int) -> float:
    rank = as_float(row.get(rank_col))
    if rank is None or rank < 0 or rank >= k:
        return 0.0
    return 1.0 / math.log2(rank + 2.0)


def find_rank_col(rows: list[dict[str, str]], candidates: Iterable[str]) -> str | None:
    if not rows:
        return None
    columns = set(rows[0].keys())
    for col in candidates:
        if col in columns:
            return col
    return None


def exact_rank_col(rows: list[dict[str, str]]) -> str | None:
    return find_rank_col(rows, ["hit_rank_0_based", "sid_hit_rank_0_based", "exact_hit_rank_0_based"])


def prefix_rank_col(rows: list[dict[str, str]], level: int) -> str | None:
    return find_rank_col(
        rows,
        [
            f"prefix{level}_hit_rank_0_based",
            f"prefix_{level}_hit_rank_0_based",
            f"prefix@{level}_hit_rank_0_based",
        ],
    )


def target_len(row: dict[str, str]) -> int:
    return sid_len(row.get("target_sid", ""))


def group_key(row: dict[str, str], columns: list[str]) -> str:
    parts = []
    for col in columns:
        if col == "target_sid_len":
            value = str(target_len(row))
        else:
            value = str(row.get(col, "")).strip() or NOT_AVAILABLE
        parts.append(f"{col}={value}")
    return "|".join(parts)


def aggregate_exact(rows: list[dict[str, str]], rank_col: str | None, topk: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows)}
    if rank_col is None:
        for k in topk:
            out[f"hr@{k}"] = NOT_AVAILABLE
            out[f"ndcg@{k}"] = NOT_AVAILABLE
        return out
    for k in topk:
        hits = sum(1 for row in rows if hit_at(row, rank_col, k))
        out[f"hr@{k}"] = rate(hits, len(rows))
        out[f"ndcg@{k}"] = rate(sum(ndcg_at(row, rank_col, k) for row in rows), len(rows))
    return out


def aggregate_prefix(
    rows: list[dict[str, str]],
    prefix_cols: dict[int, str],
    k: int,
    required_target_len: int | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    eligible = rows
    if required_target_len is not None:
        eligible = [row for row in rows if target_len(row) >= required_target_len]
    for level, col in sorted(prefix_cols.items()):
        level_rows = eligible
        if required_target_len is None:
            level_rows = [row for row in rows if target_len(row) >= level]
        hits = sum(1 for row in level_rows if hit_at(row, col, k))
        out[f"prefix@{level}_eligible_count"] = len(level_rows)
        out[f"prefix@{level}_hit@{k}"] = rate(hits, len(level_rows))
    return out


def grouped(
    rows: list[dict[str, str]],
    columns: list[str],
    rank_col: str | None,
    topk: list[int],
    prefix_cols: dict[int, str],
    focus_k: int,
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[group_key(row, columns)].append(row)
    out: dict[str, Any] = {}
    for key, group_rows in sorted(buckets.items()):
        stats = aggregate_exact(group_rows, rank_col, topk)
        stats.update(aggregate_prefix(group_rows, prefix_cols, focus_k))
        out[key] = stats
    return out


def parse_jsonish_list(value: str) -> Any:
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validity_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    duplicate_values = [as_float(row.get("duplicate_count")) for row in rows if "duplicate_count" in row]
    duplicate_values = [value for value in duplicate_values if value is not None]
    invalid_predictions = 0
    total_predictions = 0
    if rows and "pred_valid_flags" in rows[0]:
        for row in rows:
            flags = parse_jsonish_list(row.get("pred_valid_flags", ""))
            if isinstance(flags, list):
                total_predictions += len(flags)
                invalid_predictions += sum(1 for flag in flags if not bool(flag))
    return {
        "duplicate_rows": sum(1 for value in duplicate_values if value > 0),
        "duplicate_rate": rate(sum(1 for value in duplicate_values if value > 0), len(duplicate_values)) if duplicate_values else NOT_AVAILABLE,
        "invalid_prediction_count": invalid_predictions,
        "total_prediction_count": total_predictions,
        "invalid_rate": rate(invalid_predictions, total_predictions) if total_predictions else NOT_AVAILABLE,
    }


def flatten(section: str, data: Any, group: str = "overall") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(data, dict):
        if data and all(not isinstance(value, (dict, list)) for value in data.values()):
            for key, value in data.items():
                rows.append({"section": section, "group": group, "metric": key, "value": value})
        else:
            for key, value in data.items():
                rows.extend(flatten(section, value, str(key)))
    else:
        rows.append({"section": section, "group": group, "metric": "value", "value": data})
    return rows


def compact_summary(
    rows: list[dict[str, str]],
    rank_col: str | None,
    prefix_cols: dict[int, str],
    focus_k: int,
) -> dict[str, Any]:
    exact = aggregate_exact(rows, rank_col, [focus_k])
    by_len = grouped(rows, ["target_sid_len"], rank_col, [focus_k], prefix_cols, focus_k)
    pop = grouped(rows, ["popularity_group"], rank_col, [focus_k], prefix_cols, focus_k) if rows and "popularity_group" in rows[0] else {}
    len3 = by_len.get("target_sid_len=3", {})
    len4 = by_len.get("target_sid_len=4", {})
    out = {
        "samples": len(rows),
        f"overall_hr@{focus_k}": exact.get(f"hr@{focus_k}"),
        f"overall_ndcg@{focus_k}": exact.get(f"ndcg@{focus_k}"),
        "sid_len3_count": len3.get("count", 0),
        "sid_len4_count": len4.get("count", 0),
        f"len3_hr@{focus_k}": len3.get(f"hr@{focus_k}", NOT_AVAILABLE),
        f"len4_hr@{focus_k}": len4.get(f"hr@{focus_k}", NOT_AVAILABLE),
        f"head_hr@{focus_k}": pop.get("popularity_group=head", {}).get(f"hr@{focus_k}", NOT_AVAILABLE),
        f"mid_hr@{focus_k}": pop.get("popularity_group=mid", {}).get(f"hr@{focus_k}", NOT_AVAILABLE),
        f"tail_hr@{focus_k}": pop.get("popularity_group=tail", {}).get(f"hr@{focus_k}", NOT_AVAILABLE),
    }
    for level in [1, 2, 3, 4]:
        out[f"prefix{level}_hr@{focus_k}_len4"] = len4.get(f"prefix@{level}_hit@{focus_k}", NOT_AVAILABLE)
    out.update(validity_stats(rows))
    return out


def main() -> None:
    args = parse_args()
    topk = sorted(set(k for k in args.topk if k > 0))
    rows = read_csv(args.per_sample_eval)
    rank_col = exact_rank_col(rows)
    prefix_cols = {level: col for level in range(1, 10) if (col := prefix_rank_col(rows, level))}
    target_lengths = Counter(target_len(row) for row in rows)

    report = {
        "inputs": {"per_sample_eval": args.per_sample_eval.as_posix()},
        "topk": topk,
        "focus_k": args.focus_k,
        "columns": {
            "exact_rank": rank_col,
            "prefix_rank": prefix_cols,
        },
        "target_sid_length_distribution": dict(sorted(target_lengths.items())),
        "overall": aggregate_exact(rows, rank_col, topk),
        "validity": validity_stats(rows),
        "by_target_sid_length": grouped(rows, ["target_sid_len"], rank_col, topk, prefix_cols, args.focus_k),
        "by_target_sid_length_and_popularity": grouped(
            rows, ["target_sid_len", "popularity_group"], rank_col, [args.focus_k], prefix_cols, args.focus_k
        ) if rows and "popularity_group" in rows[0] else NOT_AVAILABLE,
        "by_target_sid_length_and_history_length": grouped(
            rows, ["target_sid_len", "history_length_group"], rank_col, [args.focus_k], prefix_cols, args.focus_k
        ) if rows and "history_length_group" in rows[0] else NOT_AVAILABLE,
        "by_target_sid_length_and_bucket_size": grouped(
            rows, ["target_sid_len", "target_sid_bucket_size"], rank_col, [args.focus_k], prefix_cols, args.focus_k
        ) if rows and "target_sid_bucket_size" in rows[0] else NOT_AVAILABLE,
        "compact_summary": compact_summary(rows, rank_col, prefix_cols, args.focus_k),
    }

    csv_rows: list[dict[str, Any]] = []
    for section in [
        "overall",
        "validity",
        "target_sid_length_distribution",
        "by_target_sid_length",
        "by_target_sid_length_and_popularity",
        "by_target_sid_length_and_history_length",
        "by_target_sid_length_and_bucket_size",
        "compact_summary",
    ]:
        csv_rows.extend(flatten(section, report[section]))

    write_json(args.output_json, report)
    write_csv(args.output_csv, csv_rows)
    print(f"Wrote diagnostic JSON: {args.output_json}")
    print(f"Wrote diagnostic CSV: {args.output_csv}")
    summary = report["compact_summary"]
    print(
        "Compact summary: "
        f"samples={summary['samples']} "
        f"hr@{args.focus_k}={summary.get(f'overall_hr@{args.focus_k}')} "
        f"ndcg@{args.focus_k}={summary.get(f'overall_ndcg@{args.focus_k}')} "
        f"len3_hr@{args.focus_k}={summary.get(f'len3_hr@{args.focus_k}')} "
        f"len4_hr@{args.focus_k}={summary.get(f'len4_hr@{args.focus_k}')}"
    )


if __name__ == "__main__":
    main()
