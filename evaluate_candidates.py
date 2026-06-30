#!/usr/bin/env python3
"""Expand SID predictions into item candidates and evaluate item-level recall.

This script consumes the legacy ``evaluate.py`` prediction JSON format and a
version-matched SID mapping. It does not modify model/data artifacts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils_sid import load_json, load_json_set, normalize_sid, parse_sid_tokens


NOT_AVAILABLE = "not_available"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand predicted SIDs to item candidates.")
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--item2sid", type=Path, required=True)
    parser.add_argument("--sid2items", type=Path, required=True)
    parser.add_argument("--valid-sid-set", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    parser.add_argument(
        "--prefix-levels",
        type=int,
        nargs="*",
        default=[],
        help="Optional prefix expansion levels, e.g. --prefix-levels 3 2. Exact SID buckets are always included first.",
    )
    parser.add_argument("--max-pred-sids", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument(
        "--expand-invalid-prefixes",
        action="store_true",
        help="Allow prefix expansion for parsed but invalid predicted SIDs. Default keeps invalid predictions from expanding.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


def clean_sid(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return normalize_sid(str(value).strip(" \n\""))


def item_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def rate(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def rank_metrics(ranks: list[int | None], topk: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"num_samples": len(ranks)}
    for k in topk:
        hits = 0
        ndcg_sum = 0.0
        for rank in ranks:
            if rank is not None and rank < k:
                hits += 1
                ndcg_sum += 1.0 / math.log2(rank + 2)
        out[f"hr@{k}"] = rate(hits, len(ranks))
        out[f"ndcg@{k}"] = rate(ndcg_sum, len(ranks))
    return out


def load_predictions(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"Prediction file must be a JSON list: {path}")
    records: list[dict[str, Any]] = []
    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            sample = {}
        raw_preds = sample.get("predict", [])
        if isinstance(raw_preds, list):
            pred_sids = [clean_sid(pred) for pred in raw_preds]
        elif raw_preds is None:
            pred_sids = []
        else:
            pred_sids = [clean_sid(raw_preds)]
        records.append({
            "row_index": idx,
            "target_sid": clean_sid(sample.get("output", "")),
            "pred_sids": pred_sids,
            "raw": sample,
        })
    return records


def normalize_sid2items(path: Path) -> dict[str, list[str]]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected SID -> items JSON dict: {path}")
    sid2items: dict[str, list[str]] = {}
    for sid, items in raw.items():
        normalized_sid = normalize_sid(str(sid))
        if isinstance(items, list):
            item_ids = [str(item_id) for item_id in items]
        else:
            item_ids = [str(items)]
        sid2items[normalized_sid] = sorted(item_ids, key=item_sort_key)
    return sid2items


def sid_prefix(sid: str, level: int) -> str | None:
    tokens = parse_sid_tokens(sid)
    if len(tokens) < level:
        return None
    return "".join(tokens[:level])


def build_prefix_to_sids(sid2items: dict[str, list[str]], levels: list[int]) -> dict[int, dict[str, list[str]]]:
    prefix_to_sids: dict[int, dict[str, list[str]]] = {level: defaultdict(list) for level in levels}
    for sid in sorted(sid2items.keys()):
        for level in levels:
            prefix = sid_prefix(sid, level)
            if prefix:
                prefix_to_sids[level][prefix].append(sid)
    return {
        level: {prefix: sids for prefix, sids in mapping.items()}
        for level, mapping in prefix_to_sids.items()
    }


def candidate_rank(target_item_id: str, candidates: list[str]) -> int | None:
    for rank, item_id in enumerate(candidates):
        if str(item_id) == target_item_id:
            return rank
    return None


def target_candidate_sources(target_item_id: str, candidate_details: list[dict[str, Any]]) -> list[str]:
    sources: list[str] = []
    for detail in candidate_details:
        if str(detail.get("item_id", "")) == target_item_id:
            all_source_types = detail.get("all_source_types")
            if isinstance(all_source_types, list):
                for source_type in all_source_types:
                    source_type = str(source_type)
                    if source_type and source_type not in sources:
                        sources.append(source_type)
            else:
                source_type = str(detail.get("source_type", ""))
                if source_type:
                    sources.append(source_type)
    return sources


def mean_present(values: list[int | None]) -> float | str:
    present = [value for value in values if value is not None]
    if not present:
        return NOT_AVAILABLE
    return sum(present) / len(present)


def add_candidate(
    candidate_items: list[str],
    candidate_details: list[dict[str, Any]],
    seen_items: set[str],
    candidate_detail_index: dict[str, int],
    item_id: str,
    detail: dict[str, Any],
    max_candidates: int,
) -> None:
    item_id = str(item_id)
    source_type = str(detail.get("source_type", ""))
    if item_id in seen_items:
        detail_index = candidate_detail_index.get(item_id)
        if detail_index is not None:
            all_source_types = candidate_details[detail_index].setdefault("all_source_types", [])
            if source_type and source_type not in all_source_types:
                all_source_types.append(source_type)
        return
    if len(candidate_items) >= max_candidates:
        return
    seen_items.add(item_id)
    candidate_items.append(item_id)
    detail = dict(detail)
    detail["item_id"] = item_id
    detail["all_source_types"] = [source_type] if source_type else []
    candidate_detail_index[item_id] = len(candidate_details)
    candidate_details.append(detail)


def expand_candidates(
    pred_sids: list[str],
    valid_sid_set: set[str],
    sid2items: dict[str, list[str]],
    prefix_to_sids: dict[int, dict[str, list[str]]],
    prefix_levels: list[int],
    max_pred_sids: int,
    max_candidates: int,
    expand_invalid_prefixes: bool,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[bool]]:
    sid_bucket_item_ids: list[dict[str, Any]] = []
    candidate_items: list[str] = []
    candidate_details: list[dict[str, Any]] = []
    seen_items: set[str] = set()
    candidate_detail_index: dict[str, int] = {}
    pred_valid: list[bool] = []

    for sid_rank, pred_sid in enumerate(pred_sids[:max_pred_sids]):
        pred_sid = normalize_sid(pred_sid)
        is_valid = pred_sid in valid_sid_set
        pred_valid.append(is_valid)
        exact_items = sid2items.get(pred_sid, []) if is_valid else []
        sid_bucket_item_ids.append({
            "sid_rank_0_based": sid_rank,
            "sid": pred_sid,
            "valid": is_valid,
            "item_ids": exact_items,
        })

        for item_id in exact_items:
            add_candidate(
                candidate_items,
                candidate_details,
                seen_items,
                candidate_detail_index,
                item_id,
                {
                    "source_type": "exact",
                    "source_sid": pred_sid,
                    "matched_prefix": pred_sid,
                    "sid_rank_0_based": sid_rank,
                    "expansion_level": len(parse_sid_tokens(pred_sid)),
                    "bucket_size": len(exact_items),
                },
                max_candidates,
            )

        if not is_valid and not expand_invalid_prefixes:
            continue
        for level in prefix_levels:
            prefix = sid_prefix(pred_sid, level)
            if not prefix:
                continue
            for expanded_sid in prefix_to_sids.get(level, {}).get(prefix, []):
                for item_id in sid2items.get(expanded_sid, []):
                    add_candidate(
                        candidate_items,
                        candidate_details,
                        seen_items,
                        candidate_detail_index,
                        item_id,
                        {
                            "source_type": f"prefix@{level}",
                            "source_sid": expanded_sid,
                            "matched_prefix": prefix,
                            "sid_rank_0_based": sid_rank,
                            "expansion_level": level,
                            "bucket_size": len(sid2items.get(expanded_sid, [])),
                        },
                        max_candidates,
                    )
    return sid_bucket_item_ids, candidate_items, candidate_details, pred_valid


def main() -> None:
    args = parse_args()
    topk = sorted(set(k for k in args.topk if k > 0))
    prefix_levels = sorted(set(level for level in args.prefix_levels if level > 0), reverse=True)
    predictions = load_predictions(args.prediction_file)
    test_rows = read_csv_rows(args.test_csv)
    sid2items = normalize_sid2items(args.sid2items)
    valid_sid_set = {normalize_sid(sid) for sid in load_json_set(args.valid_sid_set)}
    item2sid_raw = load_json(args.item2sid)
    item2sid = {str(item_id): normalize_sid(str(sid)) for item_id, sid in item2sid_raw.items()}
    prefix_to_sids = build_prefix_to_sids(sid2items, prefix_levels)

    if len(predictions) != len(test_rows):
        raise ValueError(
            f"prediction/test row count mismatch: predictions={len(predictions)}, test_rows={len(test_rows)}"
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    item_ranks: list[int | None] = []
    candidate_counts: list[int] = []
    invalid_sid_count = 0
    total_pred_sid_count = 0
    target_sid_mismatch_count = 0
    target_source_counts: dict[str, int] = defaultdict(int)
    target_in_source_counts: dict[str, int] = defaultdict(int)
    target_in_pool_count = 0

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for idx, (pred, row) in enumerate(zip(predictions, test_rows)):
            target_item_id = str(row.get("item_id", "")).strip()
            history_item_ids = [str(item_id) for item_id in parse_list(row.get("history_item_id", ""))]
            history_item_sids = [clean_sid(sid) for sid in parse_list(row.get("history_item_sid", ""))]
            target_sid = clean_sid(row.get("item_sid", pred["target_sid"]))
            if pred["target_sid"] and target_sid and pred["target_sid"] != target_sid:
                target_sid_mismatch_count += 1

            sid_bucket_item_ids, candidate_items, candidate_details, pred_valid = expand_candidates(
                pred["pred_sids"],
                valid_sid_set,
                sid2items,
                prefix_to_sids,
                prefix_levels,
                args.max_pred_sids,
                args.max_candidates,
                args.expand_invalid_prefixes,
            )
            total_pred_sid_count += len(pred_valid)
            invalid_sid_count += sum(1 for valid in pred_valid if not valid)
            rank = candidate_rank(target_item_id, candidate_items)
            item_ranks.append(rank)
            candidate_counts.append(len(candidate_items))
            target_sources = target_candidate_sources(target_item_id, candidate_details)
            target_first_source = target_sources[0] if target_sources else None
            target_source_counts[target_first_source or "not_in_candidates"] += 1
            for source_type in set(target_sources):
                target_in_source_counts[source_type] += 1
            if rank is not None:
                target_in_pool_count += 1

            record = {
                "row_index": idx,
                "target_item_id": target_item_id,
                "target_sid": target_sid,
                "mapped_target_sid": item2sid.get(target_item_id, ""),
                "history_item_id": history_item_ids,
                "history_item_sid": history_item_sids,
                "pred_sids": pred["pred_sids"][: args.max_pred_sids],
                "pred_sid_valid": pred_valid,
                "sid_bucket_item_ids": sid_bucket_item_ids,
                "candidate_item_ids": candidate_items,
                "candidate_details": candidate_details,
                "candidate_hit_rank_0_based": rank,
                "candidate_pool_hit": rank is not None,
                "target_candidate_first_source_type": target_first_source,
                "target_candidate_source_types": target_sources,
                "target_in_exact_bucket": "exact" in target_sources,
                "target_in_prefix": {
                    f"prefix@{level}": f"prefix@{level}" in target_sources
                    for level in prefix_levels
                },
                "generation_scores": None,
                "latency_ms": None,
            }
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    metrics = rank_metrics(item_ranks, topk)
    candidate_recall = {
        key.replace("hr@", "candidate_recall@"): value
        for key, value in metrics.items()
        if key.startswith("hr@")
    }
    target_in_source = {
        source_type: {
            "count": count,
            "rate": rate(count, len(predictions)),
        }
        for source_type, count in sorted(target_in_source_counts.items())
    }
    for required_source in ["exact", "prefix@3", "prefix@2"]:
        target_in_source.setdefault(required_source, {"count": 0, "rate": 0.0})
    target_source_distribution = {
        source_type: {
            "count": count,
            "rate": rate(count, len(predictions)),
        }
        for source_type, count in sorted(target_source_counts.items())
    }
    report = {
        "inputs": {
            "prediction_file": args.prediction_file.as_posix(),
            "test_csv": args.test_csv.as_posix(),
            "item2sid": args.item2sid.as_posix(),
            "sid2items": args.sid2items.as_posix(),
            "valid_sid_set": args.valid_sid_set.as_posix(),
        },
        "output_jsonl": args.output_jsonl.as_posix(),
        "num_samples": len(predictions),
        "topk": topk,
        "prefix_levels": prefix_levels,
        "max_pred_sids": args.max_pred_sids,
        "max_candidates": args.max_candidates,
        "expand_invalid_prefixes": args.expand_invalid_prefixes,
        "candidate_count": {
            "min": min(candidate_counts) if candidate_counts else 0,
            "max": max(candidate_counts) if candidate_counts else 0,
            "mean": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
        },
        "sid_validity": {
            "invalid_sid_count": invalid_sid_count,
            "total_pred_sid_count": total_pred_sid_count,
            "invalid_sid_rate": rate(invalid_sid_count, total_pred_sid_count),
        },
        "target_sid_mismatch_count": target_sid_mismatch_count,
        "candidate_pool_recall": {
            "target_in_pool_count": target_in_pool_count,
            "target_in_pool_rate": rate(target_in_pool_count, len(predictions)),
            "candidate_recall_at_budget": rate(target_in_pool_count, len(predictions)),
            "budget": args.max_candidates,
            "avg_rank_before_rerank_if_hit": mean_present(item_ranks),
        },
        "candidate_recall": candidate_recall,
        "target_in_source": target_in_source,
        "target_source_distribution": target_source_distribution,
        "item_level_before_rerank": metrics,
        "notes": {
            "generation_scores": "not_available: legacy evaluate.py does not write beam scores",
            "latency_ms": "not_available: legacy evaluate.py does not write per-sample latency",
            "leakage_policy": "target item is used only for metrics, never for candidate construction",
            "target_source_distribution": "first source that inserted the target into the deduplicated candidate list; exact buckets are expanded before prefixes",
        },
    }
    write_json(args.output_report, report)
    print(f"Wrote candidate JSONL: {args.output_jsonl}")
    print(f"Wrote candidate report: {args.output_report}")
    print(
        "Item-level candidate summary: "
        f"num_samples={len(predictions)} "
        f"candidates_mean={report['candidate_count']['mean']:.2f} "
        f"HR@20={metrics.get('hr@20', NOT_AVAILABLE)} "
        f"NDCG@20={metrics.get('ndcg@20', NOT_AVAILABLE)}"
    )


if __name__ == "__main__":
    main()
