#!/usr/bin/env python3
"""Apply a frozen Stage 7 P2 history-aware ranker to fused candidates.

This script is inference-only. It loads the frozen P2-3 model, recomputes the
same non-leaky features used during P2-3, ranks candidates, and writes metrics.
It does not train, update, or rewrite the frozen ranker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import train_stage7_p2_history_ranker as history
import train_stage7_p2_minimal_ranker as base


DEFAULT_KS = [1, 5, 10, 20, 50]
NOT_AVAILABLE = "not_available"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a frozen P2 history-aware ranker.")
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument(
        "--frozen-ranker",
        type=Path,
        required=True,
        help="Path to p2_history_ranker/model.json or p2_frozen_ranker/frozen_ranker_manifest.json.",
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--item-emb", type=Path, required=True)
    parser.add_argument("--row-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-split", choices=["valid"], default="valid")
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--detail-limit", type=int, default=100)
    parser.add_argument("--sid-rank-weight", type=float, default=None)
    parser.add_argument("--source-weight", type=float, default=None)
    parser.add_argument("--popularity-weight", type=float, default=None)
    parser.add_argument("--history-cosine-weight", type=float, default=None)
    parser.add_argument("--recent-cosine-weight", type=float, default=None)
    parser.add_argument("--bucket-penalty-weight", type=float, default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def no_test_path(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    return "test" not in lowered_parts and path.name.lower() != "test.csv"


def resolve_model_path(path: Path) -> Path:
    payload = read_json(path)
    if isinstance(payload, dict) and payload.get("model_type") == "history_aware_linear_pairwise_logistic":
        return path
    model_path = payload.get("frozen_ranker", {}).get("model_path") if isinstance(payload, dict) else None
    if not model_path:
        raise ValueError(
            "--frozen-ranker must be a P2 history model.json or a frozen_ranker_manifest.json with frozen_ranker.model_path"
        )
    return Path(model_path)


def metric_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section in ["fusion_rrf", "history_aware_rerank", "delta_history_minus_fusion"]:
        values = report.get(section, {})
        if not isinstance(values, dict):
            continue
        for metric, value in values.items():
            if metric == "num_samples":
                continue
            rows.append({"section": section, "metric": metric, "value": value})
    return rows


def build_resource_namespace(args: argparse.Namespace, model: dict[str, Any]) -> argparse.Namespace:
    weights = dict(model.get("heuristic_component_weights", {}))
    overrides = {
        "sid_rank_weight": args.sid_rank_weight,
        "source_weight": args.source_weight,
        "popularity_weight": args.popularity_weight,
        "history_cosine_weight": args.history_cosine_weight,
        "recent_cosine_weight": args.recent_cosine_weight,
        "bucket_penalty_weight": args.bucket_penalty_weight,
    }
    for key, value in overrides.items():
        if value is not None:
            weights[key] = value
    defaults = {
        "sid_rank_weight": 1.0,
        "source_weight": 4.0,
        "popularity_weight": 0.2,
        "history_cosine_weight": 0.8,
        "recent_cosine_weight": 0.4,
        "bucket_penalty_weight": 0.05,
    }
    for key, value in defaults.items():
        weights.setdefault(key, value)
    return argparse.Namespace(
        train_csv=args.train_csv,
        item_emb=args.item_emb,
        row_index=args.row_index,
        **weights,
    )


def apply_stats_to_vector(raw_vector: list[float], stats: dict[str, Any]) -> list[float]:
    means = stats["mean"]
    stds = stats["std"]
    if len(raw_vector) != len(means) or len(raw_vector) != len(stds):
        raise ValueError("Feature vector length does not match frozen feature stats.")
    return [(value - means[idx]) / stds[idx] for idx, value in enumerate(raw_vector)]


def dot(weights: list[float], features: list[float]) -> float:
    return sum(weight * feature for weight, feature in zip(weights, features))


def rank_metrics(ranks: list[int | None], ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"num_samples": len(ranks)}
    for k in ks:
        hits = 0
        ndcg_sum = 0.0
        for rank in ranks:
            if rank is not None and rank < k:
                hits += 1
                ndcg_sum += 1.0 / math.log2(rank + 2)
        out[f"hr@{k}"] = base.rate(hits, len(ranks))
        out[f"ndcg@{k}"] = base.rate(ndcg_sum, len(ranks))
    return out


def build_outputs(
    records: list[dict[str, Any]],
    resources: history.FeatureResources,
    model: dict[str, Any],
    max_candidates: int,
    detail_limit: int,
    ks: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    weights = [float(value) for value in model["weights"]]
    stats = model["feature_stats"]
    output_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    fusion_ranks: list[int | None] = []
    learned_ranks: list[int | None] = []
    candidate_counts: list[int] = []

    for idx, record in enumerate(records):
        row_id = base.normalize_id(record.get("row_index", record.get("sample_id")), idx)
        target_item_id = str(record.get("target_item_id", "")).strip()
        candidate_items = base.list_field(record, "candidate_item_ids")[:max_candidates]
        history_items = base.list_field(record, "history_item_id")
        details = base.detail_by_item(record)
        scored: list[dict[str, Any]] = []
        for rank_0, item_id in enumerate(candidate_items):
            detail = details.get(item_id, {})
            raw = history.candidate_raw_features(
                item_id,
                detail,
                history_items,
                rank_0 + 1,
                len(candidate_items),
                resources,
            )
            raw_vector = history.vector_from_raw(raw)
            features = apply_stats_to_vector(raw_vector, stats)
            scored.append(
                {
                    "item_id": item_id,
                    "score": dot(weights, features),
                    "original_rank_0_based": rank_0,
                    "features_raw": raw,
                }
            )
        scored.sort(key=lambda item: (-item["score"], item["original_rank_0_based"], base.item_sort_key(item["item_id"])))
        learned_items = [item["item_id"] for item in scored]
        fusion_rank = base.rank_of(target_item_id, candidate_items)
        learned_rank = base.rank_of(target_item_id, learned_items)
        fusion_ranks.append(fusion_rank)
        learned_ranks.append(learned_rank)
        candidate_counts.append(len(candidate_items))
        output_rows.append(
            {
                "row_index": row_id,
                "target_item_id": target_item_id,
                "history_item_id": history_items,
                "candidate_item_ids": candidate_items,
                "history_aware_reranked_item_ids": learned_items,
                "reranked_item_ids": learned_items,
                "history_aware_details": scored[: max(detail_limit, 0)],
                "candidate_hit_rank_0_based": fusion_rank,
                "history_aware_hit_rank_0_based": learned_rank,
            }
        )
        per_sample_rows.append(
            {
                "row_index": row_id,
                "target_item_id": target_item_id,
                "fusion_hit_rank_0_based": "" if fusion_rank is None else fusion_rank,
                "history_aware_hit_rank_0_based": "" if learned_rank is None else learned_rank,
                "num_candidates": len(candidate_items),
            }
        )

    fusion_metrics = rank_metrics(fusion_ranks, ks)
    learned_metrics = rank_metrics(learned_ranks, ks)
    delta = {
        key: learned_metrics.get(key, 0.0) - fusion_metrics.get(key, 0.0)
        for key in learned_metrics
        if key.startswith("hr@") or key.startswith("ndcg@")
    }
    candidate_summary = {
        "min": min(candidate_counts) if candidate_counts else 0,
        "max": max(candidate_counts) if candidate_counts else 0,
        "mean": (sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0,
    }
    metrics = {
        "fusion_rrf": fusion_metrics,
        "history_aware_rerank": learned_metrics,
        "delta_history_minus_fusion": delta,
        "candidate_count": candidate_summary,
    }
    return output_rows, per_sample_rows, metrics


def make_markdown(report: dict[str, Any]) -> str:
    history_metrics = report["history_aware_rerank"]
    fusion_metrics = report["fusion_rrf"]
    return "\n".join(
        [
            "# P2 Frozen History Ranker Apply Report",
            "",
            f"- split: `{report['eval_split']}`",
            f"- frozen_model: `{report['inputs']['frozen_model']}`",
            f"- frozen_model_sha256_before: `{report['frozen_ranker_integrity']['sha256_before']}`",
            f"- frozen_model_sha256_after: `{report['frozen_ranker_integrity']['sha256_after']}`",
            f"- frozen_ranker_unchanged: `{report['frozen_ranker_integrity']['unchanged']}`",
            f"- no_training: `{report['policy']['no_training']}`",
            "",
            "| Model | HR@20 | NDCG@20 |",
            "|---|---:|---:|",
            f"| Fusion RRF | {fusion_metrics.get('hr@20', 0):.6f} | {fusion_metrics.get('ndcg@20', 0):.6f} |",
            f"| Frozen history ranker | {history_metrics.get('hr@20', 0):.6f} | {history_metrics.get('ndcg@20', 0):.6f} |",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    for path in [args.candidate_jsonl, args.frozen_ranker, args.train_csv, args.item_emb, args.row_index]:
        base.assert_file(path)
    checked_paths = [args.candidate_jsonl, args.frozen_ranker, args.train_csv, args.item_emb, args.row_index, args.output_dir]
    bad_paths = [path.as_posix() for path in checked_paths if not no_test_path(path)]
    if bad_paths:
        raise SystemExit(f"P2 frozen ranker apply for valid split refuses test paths: {bad_paths}")

    model_path = resolve_model_path(args.frozen_ranker)
    base.assert_file(model_path)
    model = read_json(model_path)
    if model.get("model_type") != "history_aware_linear_pairwise_logistic":
        raise ValueError(f"Unsupported frozen ranker model_type: {model.get('model_type')}")
    if model.get("feature_names") != history.FEATURE_NAMES:
        raise ValueError("Frozen model feature_names do not match current P2 history feature contract.")
    model_hash_before = file_sha256(model_path)

    resource_args = build_resource_namespace(args, model)
    resources = history.build_feature_resources(resource_args)
    records = base.read_jsonl(args.candidate_jsonl)
    ks = sorted(set(k for k in args.ks if k > 0))
    if 20 not in ks:
        ks.append(20)
        ks = sorted(ks)
    output_rows, per_sample_rows, metrics = build_outputs(
        records,
        resources,
        model,
        max_candidates=args.max_candidates,
        detail_limit=args.detail_limit,
        ks=ks,
    )
    model_hash_after = file_sha256(model_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / "history_reranked_candidates.jsonl"
    per_sample_csv = args.output_dir / "per_sample_metrics.csv"
    report_json = args.output_dir / "ranker_apply_report.json"
    report_csv = args.output_dir / "ranker_apply_report.csv"
    report_md = args.output_dir / "ranker_apply_report.md"

    report = {
        "eval_split": args.eval_split,
        "inputs": {
            "candidate_jsonl": args.candidate_jsonl.as_posix(),
            "frozen_ranker_arg": args.frozen_ranker.as_posix(),
            "frozen_model": model_path.as_posix(),
            "train_csv": args.train_csv.as_posix(),
            "item_emb": args.item_emb.as_posix(),
            "row_index": args.row_index.as_posix(),
        },
        "outputs": {
            "history_reranked_candidates": output_jsonl.as_posix(),
            "per_sample_metrics": per_sample_csv.as_posix(),
            "ranker_apply_report": report_json.as_posix(),
        },
        "selected_config_id": model.get("selected_config_id", NOT_AVAILABLE),
        "model_type": model.get("model_type", NOT_AVAILABLE),
        "topk": ks,
        "heuristic_component_weights_used": {
            "sid_rank_weight": resources.sid_rank_weight,
            "source_weight": resources.source_weight,
            "popularity_weight": resources.popularity_weight,
            "history_cosine_weight": resources.history_cosine_weight,
            "recent_cosine_weight": resources.recent_cosine_weight,
            "bucket_penalty_weight": resources.bucket_penalty_weight,
        },
        "policy": {
            "no_training": True,
            "test_paths_refused": True,
            "target_item_used_only_for_metrics": True,
            "popularity_source": "train_csv_only",
            "history_features_source": "query_history_only",
        },
        "frozen_ranker_integrity": {
            "sha256_before": model_hash_before,
            "sha256_after": model_hash_after,
            "unchanged": model_hash_before == model_hash_after,
        },
        **metrics,
    }
    write_jsonl(output_jsonl, output_rows)
    write_csv(
        per_sample_csv,
        ["row_index", "target_item_id", "fusion_hit_rank_0_based", "history_aware_hit_rank_0_based", "num_candidates"],
        per_sample_rows,
    )
    write_json(report_json, report)
    write_csv(report_csv, ["section", "metric", "value"], metric_rows(report))
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(make_markdown(report))

    print(f"Wrote frozen ranker outputs: {args.output_dir}")
    print(
        "P2 frozen apply summary: "
        f"fusion_hr20={metrics['fusion_rrf'].get('hr@20')} "
        f"history_hr20={metrics['history_aware_rerank'].get('hr@20')} "
        f"history_ndcg20={metrics['history_aware_rerank'].get('ndcg@20')} "
        f"ranker_unchanged={model_hash_before == model_hash_after}"
    )


if __name__ == "__main__":
    main()
