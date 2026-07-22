#!/usr/bin/env python3
"""Merge frozen CF validation candidates with S6 direct SASRec candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_index", row.get("sample_id", "")))


def detail_by_item(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("item_id")): item for item in row.get("candidate_details", []) if isinstance(item, dict)}


def merge_rows(cf: dict[str, Any], direct: dict[str, Any]) -> dict[str, Any]:
    if str(cf.get("target_item_id")) != str(direct.get("target_item_id")):
        raise ValueError(f"target mismatch for row {row_id(cf)}")
    cf_items = [str(item) for item in cf.get("candidate_item_ids", [])]
    direct_items = [str(item) for item in direct.get("candidate_item_ids", [])]
    direct_details = detail_by_item(direct)
    out_items: list[str] = []
    details: list[dict[str, Any]] = []
    for item in cf_items + [item for item in direct_items if item not in set(cf_items)]:
        cf_rank = cf_items.index(item) if item in cf_items else None
        direct_rank = direct_items.index(item) if item in direct_items else None
        detail = {
            "item_id": item,
            "from_cf": cf_rank is not None,
            "from_sasrec_direct": direct_rank is not None,
            "cf_rank": cf_rank,
            "sasrec_direct_rank": direct_rank,
            "overlap": cf_rank is not None and direct_rank is not None,
        }
        if direct_rank is not None:
            detail["sasrec_direct_features"] = direct_details.get(item, {})
        out_items.append(item)
        details.append(detail)
    target = str(cf.get("target_item_id"))
    hit_rank = next((idx for idx, item in enumerate(out_items) if item == target), None)
    return {
        "row_index": row_id(cf),
        "target_item_id": target,
        "history_item_id": cf.get("history_item_id", direct.get("history_item_id", [])),
        "candidate_item_ids": out_items,
        "candidate_details": details,
        "candidate_hit_rank_0_based": hit_rank,
        "candidate_pool_hit": hit_rank is not None,
    }


def merge_files(cf_path: Path, direct_path: Path, output_jsonl: Path, report_path: Path) -> dict[str, Any]:
    existing = [path.as_posix() for path in [output_jsonl, report_path] if path.exists() and path.stat().st_size > 0]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing S6 union artifacts: {existing}")
    cf_rows = {row_id(row): row for row in read_jsonl(cf_path)}
    direct_rows = {row_id(row): row for row in read_jsonl(direct_path)}
    if set(cf_rows) != set(direct_rows):
        raise ValueError(f"row alignment mismatch: cf={len(cf_rows)} direct={len(direct_rows)}")
    merged = [merge_rows(cf_rows[key], direct_rows[key]) for key in sorted(cf_rows, key=lambda v: int(v) if v.isdigit() else v)]
    write_jsonl(output_jsonl, merged)
    counts = [len(row["candidate_item_ids"]) for row in merged]
    overlap = sum(1 for row in merged for detail in row["candidate_details"] if detail["overlap"])
    report = {
        "schema": "s6_cf_direct_sasrec_union_report.v1",
        "num_samples": len(merged),
        "candidate_count": {
            "min": min(counts) if counts else 0,
            "mean": sum(counts) / len(counts) if counts else 0.0,
            "max": max(counts) if counts else 0,
        },
        "overlap_candidate_count": overlap,
        "output_jsonl": output_jsonl.as_posix(),
        "test_read": False,
    }
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge CF and direct SASRec candidates.")
    parser.add_argument("--cf-candidates", type=Path, required=True)
    parser.add_argument("--direct-candidates", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "cf_candidates": args.cf_candidates.as_posix(),
            "direct_candidates": args.direct_candidates.as_posix(),
            "output_jsonl": args.output_jsonl.as_posix(),
            "report": args.report.as_posix(),
        }, indent=2, sort_keys=True))
        return
    report = merge_files(args.cf_candidates, args.direct_candidates, args.output_jsonl, args.report)
    print(json.dumps({"report": args.report.as_posix(), "num_samples": report["num_samples"]}, indent=2))


if __name__ == "__main__":
    main()
