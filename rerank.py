#!/usr/bin/env python3
"""Lightweight non-leaky reranking for expanded item candidates."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from utils_sid import load_json


NOT_AVAILABLE = "not_available"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank candidate JSONL using train-only and history-only features.")
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-split", choices=["valid", "test"], default="test")
    parser.add_argument("--item-emb", type=Path, default=None)
    parser.add_argument("--row-index", type=Path, default=None)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--sid-rank-weight", type=float, default=1.0)
    parser.add_argument("--source-weight", type=float, default=0.5)
    parser.add_argument("--popularity-weight", type=float, default=0.2)
    parser.add_argument("--history-cosine-weight", type=float, default=0.8)
    parser.add_argument("--recent-cosine-weight", type=float, default=0.4)
    parser.add_argument("--bucket-penalty-weight", type=float, default=0.05)
    return parser.parse_args()


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
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


def train_popularity(train_csv: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in read_csv_rows(train_csv):
        target = str(row.get("item_id", "")).strip()
        if target:
            counts[target] += 1
        for item_id in parse_list(row.get("history_item_id", "")):
            counts[str(item_id)] += 1
    return counts


def rate(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def rank_of(target_item_id: str, candidates: list[str]) -> int | None:
    for rank, item_id in enumerate(candidates):
        if str(item_id) == target_item_id:
            return rank
    return None


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


def load_embeddings(item_emb: Path | None, row_index_path: Path | None) -> tuple[Any, dict[str, int], list[str]]:
    if item_emb is None or row_index_path is None:
        return None, {}, ["embedding rerank disabled: --item-emb and --row-index are both required"]
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, {}, [f"embedding rerank disabled: failed to import numpy: {exc}"]
    matrix = np.load(item_emb).astype("float32", copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    matrix = matrix / norms
    row_index_raw = load_json(row_index_path)
    row_index = {str(item_id): int(row) for item_id, row in row_index_raw.items()}
    return matrix, row_index, []


def cosine(matrix: Any, row_index: dict[str, int], item_a: str, item_b: str) -> float:
    if matrix is None:
        return 0.0
    row_a = row_index.get(str(item_a))
    row_b = row_index.get(str(item_b))
    if row_a is None or row_b is None:
        return 0.0
    return float(matrix[row_a].dot(matrix[row_b]))


def source_score(source_type: str, expansion_level: int | None) -> float:
    if source_type == "exact":
        return 1.0
    if expansion_level is None:
        return 0.0
    return max(0.0, min(1.0, float(expansion_level) / 4.0))


def detail_by_item(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for detail in record.get("candidate_details", []):
        item_id = str(detail.get("item_id", ""))
        if item_id and item_id not in out:
            out[item_id] = detail
    return out


def score_candidate(
    item_id: str,
    detail: dict[str, Any],
    history: list[str],
    popularity: Counter[str],
    max_log_pop: float,
    matrix: Any,
    row_index: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, float]:
    sid_rank = detail.get("sid_rank_0_based")
    try:
        sid_rank_value = int(sid_rank)
    except (TypeError, ValueError):
        sid_rank_value = 999999
    sid_rank_score = 1.0 / (sid_rank_value + 1.0)

    expansion_level = detail.get("expansion_level")
    try:
        expansion_level_value = int(expansion_level)
    except (TypeError, ValueError):
        expansion_level_value = None
    src_score = source_score(str(detail.get("source_type", "")), expansion_level_value)

    pop_log = math.log1p(popularity.get(str(item_id), 0))
    pop_score = 0.0 if max_log_pop <= 0 else pop_log / max_log_pop

    history_sims = [cosine(matrix, row_index, item_id, hist_item) for hist_item in history]
    history_cosine = max(history_sims) if history_sims else 0.0
    recent_cosine = cosine(matrix, row_index, item_id, history[-1]) if history else 0.0

    bucket_size = 1.0
    try:
        bucket_size = float(detail.get("bucket_size", 1.0))
    except (TypeError, ValueError):
        bucket_size = 1.0
    bucket_penalty = math.log1p(max(bucket_size, 1.0))

    total = (
        args.sid_rank_weight * sid_rank_score
        + args.source_weight * src_score
        + args.popularity_weight * pop_score
        + args.history_cosine_weight * history_cosine
        + args.recent_cosine_weight * recent_cosine
        - args.bucket_penalty_weight * bucket_penalty
    )
    return {
        "score": total,
        "sid_rank_score": sid_rank_score,
        "source_score": src_score,
        "popularity_score": pop_score,
        "history_cosine": history_cosine,
        "recent_cosine": recent_cosine,
        "bucket_penalty": bucket_penalty,
    }


def main() -> None:
    args = parse_args()
    topk = sorted(set(k for k in args.topk if k > 0))
    records = read_jsonl(args.candidate_jsonl)
    popularity = train_popularity(args.train_csv)
    max_log_pop = max([math.log1p(v) for v in popularity.values()], default=0.0)
    matrix, row_index, warnings = load_embeddings(args.item_emb, args.row_index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / "reranked_candidates.jsonl"
    per_sample_csv = args.output_dir / "per_sample_rerank.csv"
    report_json = args.output_dir / "rerank_report.json"
    report_csv = args.output_dir / "rerank_report.csv"

    before_ranks: list[int | None] = []
    after_ranks: list[int | None] = []
    per_sample_rows: list[dict[str, Any]] = []

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            target_item_id = str(record.get("target_item_id", ""))
            original_candidates = [str(item_id) for item_id in record.get("candidate_item_ids", [])[: args.max_candidates]]
            history = [str(item_id) for item_id in record.get("history_item_id", [])]
            details = detail_by_item(record)

            scored: list[dict[str, Any]] = []
            for original_rank, item_id in enumerate(original_candidates):
                detail = details.get(item_id, {})
                features = score_candidate(
                    item_id,
                    detail,
                    history,
                    popularity,
                    max_log_pop,
                    matrix,
                    row_index,
                    args,
                )
                scored.append({
                    "item_id": item_id,
                    "original_rank_0_based": original_rank,
                    **features,
                })
            scored.sort(key=lambda item: (-item["score"], item["original_rank_0_based"], item["item_id"]))
            reranked_items = [item["item_id"] for item in scored]

            before_rank = rank_of(target_item_id, original_candidates)
            after_rank = rank_of(target_item_id, reranked_items)
            before_ranks.append(before_rank)
            after_ranks.append(after_rank)

            out_record = {
                "row_index": record.get("row_index"),
                "target_item_id": target_item_id,
                "target_sid": record.get("target_sid"),
                "history_item_id": history,
                "candidate_item_ids": original_candidates,
                "reranked_item_ids": reranked_items,
                "reranked_details": scored[: args.max_candidates],
                "candidate_hit_rank_0_based": before_rank,
                "reranked_hit_rank_0_based": after_rank,
            }
            f.write(json.dumps(out_record, ensure_ascii=False, sort_keys=True) + "\n")

            per_sample_rows.append({
                "row_index": record.get("row_index"),
                "target_item_id": target_item_id,
                "candidate_hit_rank_0_based": "" if before_rank is None else before_rank,
                "reranked_hit_rank_0_based": "" if after_rank is None else after_rank,
                "num_candidates": len(original_candidates),
            })

    before_metrics = rank_metrics(before_ranks, topk)
    after_metrics = rank_metrics(after_ranks, topk)
    report = {
        "inputs": {
            "candidate_jsonl": args.candidate_jsonl.as_posix(),
            "eval_split": args.eval_split,
            "train_csv": args.train_csv.as_posix(),
            "item_emb": args.item_emb.as_posix() if args.item_emb else NOT_AVAILABLE,
            "row_index": args.row_index.as_posix() if args.row_index else NOT_AVAILABLE,
        },
        "outputs": {
            "reranked_candidates_jsonl": output_jsonl.as_posix(),
            "per_sample_csv": per_sample_csv.as_posix(),
        },
        "topk": topk,
        "weights": {
            "sid_rank_weight": args.sid_rank_weight,
            "source_weight": args.source_weight,
            "popularity_weight": args.popularity_weight,
            "history_cosine_weight": args.history_cosine_weight,
            "recent_cosine_weight": args.recent_cosine_weight,
            "bucket_penalty_weight": args.bucket_penalty_weight,
        },
        "warnings": warnings,
        "leakage_policy": "target item is used only for metrics, not for rerank features",
        "before_rerank": before_metrics,
        "after_rerank": after_metrics,
        "delta_after_minus_before": {
            key: after_metrics.get(key, 0.0) - before_metrics.get(key, 0.0)
            for key in after_metrics
            if key.startswith("hr@") or key.startswith("ndcg@")
        },
    }
    write_json(report_json, report)
    flat_rows = []
    for section in ["before_rerank", "after_rerank", "delta_after_minus_before"]:
        for metric, value in report[section].items():
            flat_rows.append({"section": section, "metric": metric, "value": value})
    write_csv(report_csv, ["section", "metric", "value"], flat_rows)
    write_csv(
        per_sample_csv,
        ["row_index", "target_item_id", "candidate_hit_rank_0_based", "reranked_hit_rank_0_based", "num_candidates"],
        per_sample_rows,
    )

    print(f"Wrote reranked JSONL: {output_jsonl}")
    print(f"Wrote rerank report JSON: {report_json}")
    print(f"Wrote rerank report CSV: {report_csv}")
    print(
        "Rerank summary: "
        f"before_HR@20={before_metrics.get('hr@20', NOT_AVAILABLE)} "
        f"after_HR@20={after_metrics.get('hr@20', NOT_AVAILABLE)} "
        f"before_NDCG@20={before_metrics.get('ndcg@20', NOT_AVAILABLE)} "
        f"after_NDCG@20={after_metrics.get('ndcg@20', NOT_AVAILABLE)}"
    )


if __name__ == "__main__":
    main()
