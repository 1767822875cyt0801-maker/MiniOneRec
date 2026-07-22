#!/usr/bin/env python3
"""Train the P2 history-aware learned reranker.

This extends the P2-2 minimal ranker with the same non-leaky context features
used by the current heuristic reranker: train-only popularity, history/recent
embedding cosine, bucket penalty, and source/SID-rank components.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import train_stage7_p2_minimal_ranker as base


DEFAULT_ROOT = Path("results/stage7_validation_protocol/valid/Industrial_and_Scientific")
DEFAULT_CATEGORY = "Industrial_and_Scientific"
DEFAULT_KS = [1, 5, 10, 20, 50]
MINIMAL_FEATURE_NAMES = list(base.FEATURE_NAMES)
HISTORY_FEATURE_NAMES = [
    "sid_rank_score",
    "source_score",
    "popularity_score",
    "history_cosine",
    "recent_cosine",
    "bucket_penalty",
    "bucket_size_log",
    "expansion_level_score",
    "exact_source",
    "prefix_source",
    "heuristic_score",
]
FEATURE_NAMES = MINIMAL_FEATURE_NAMES + HISTORY_FEATURE_NAMES


@dataclass
class FeatureResources:
    popularity: Counter[str]
    max_log_pop: float
    matrix: Any
    row_index: dict[str, int]
    sid_rank_weight: float
    source_weight: float
    popularity_weight: float
    history_cosine_weight: float
    recent_cosine_weight: float
    bucket_penalty_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P2 history-aware learned ranker.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--split", choices=["valid"], default="valid")
    parser.add_argument("--candidate-mode", choices=["exact"], default="exact")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path(f"data/Amazon/sid_versions/cf_k512_dedup/{DEFAULT_CATEGORY}/train.csv"),
    )
    parser.add_argument(
        "--item-emb",
        type=Path,
        default=Path(f"data/Amazon/cs_embeddings/{DEFAULT_CATEGORY}/{DEFAULT_CATEGORY}.cf_emb.npy"),
    )
    parser.add_argument(
        "--row-index",
        type=Path,
        default=Path(f"data/Amazon/cs_embeddings/{DEFAULT_CATEGORY}/{DEFAULT_CATEGORY}.row_index.json"),
    )
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--learning-rate-grid", type=float, nargs="+", default=[0.003, 0.01, 0.03, 0.1])
    parser.add_argument("--l2-grid", type=float, nargs="+", default=[0.0, 0.0001, 0.001, 0.01, 0.1])
    parser.add_argument("--max-negatives-per-positive", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--detail-limit", type=int, default=100)
    parser.add_argument("--min-select-improvement", type=float, default=0.0)
    parser.add_argument("--sid-rank-weight", type=float, default=1.0)
    parser.add_argument("--source-weight", type=float, default=4.0)
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


def train_popularity(train_csv: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in read_csv_rows(train_csv):
        target = str(row.get("item_id", "")).strip()
        if target:
            counts[target] += 1
        for item_id in parse_list(row.get("history_item_id", "")):
            counts[str(item_id)] += 1
    return counts


def load_embeddings(item_emb: Path, row_index_path: Path) -> tuple[Any, dict[str, int]]:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"Failed to import numpy for history-aware features: {exc}") from exc

    matrix = np.load(item_emb).astype("float32", copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    matrix = matrix / norms
    row_index_raw = base.read_json(row_index_path)
    row_index = {str(item_id): int(row) for item_id, row in row_index_raw.items()}
    return matrix, row_index


def cosine(matrix: Any, row_index: dict[str, int], item_a: str, item_b: str) -> float:
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


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def history_raw_features(
    item_id: str,
    detail: dict[str, Any],
    history: list[str],
    resources: FeatureResources,
) -> dict[str, float]:
    sid_rank_value = parse_int(detail.get("sid_rank_0_based"))
    if sid_rank_value is None:
        sid_rank_value = 999999
    sid_rank_score = 1.0 / (sid_rank_value + 1.0)

    expansion_level = parse_int(detail.get("expansion_level"))
    src_type = str(detail.get("source_type", ""))
    src_score = source_score(src_type, expansion_level)

    pop_log = math.log1p(resources.popularity.get(str(item_id), 0))
    pop_score = 0.0 if resources.max_log_pop <= 0 else pop_log / resources.max_log_pop

    history_sims = [cosine(resources.matrix, resources.row_index, item_id, hist_item) for hist_item in history]
    history_cosine = max(history_sims) if history_sims else 0.0
    recent_cosine = cosine(resources.matrix, resources.row_index, item_id, history[-1]) if history else 0.0

    bucket_size = base.parse_float(detail.get("bucket_size"))
    if bucket_size is None:
        bucket_size = 1.0
    bucket_penalty = math.log1p(max(bucket_size, 1.0))

    expansion_score = 0.0 if expansion_level is None else max(0.0, min(1.0, float(expansion_level) / 4.0))
    exact_source = 1.0 if src_type == "exact" else 0.0
    prefix_source = 1.0 if src_type.startswith("prefix@") else 0.0

    heuristic_score = (
        resources.sid_rank_weight * sid_rank_score
        + resources.source_weight * src_score
        + resources.popularity_weight * pop_score
        + resources.history_cosine_weight * history_cosine
        + resources.recent_cosine_weight * recent_cosine
        - resources.bucket_penalty_weight * bucket_penalty
    )
    return {
        "sid_rank_score": sid_rank_score,
        "source_score": src_score,
        "popularity_score": pop_score,
        "history_cosine": history_cosine,
        "recent_cosine": recent_cosine,
        "bucket_penalty": bucket_penalty,
        "bucket_size_log": bucket_penalty,
        "expansion_level_score": expansion_score,
        "exact_source": exact_source,
        "prefix_source": prefix_source,
        "heuristic_score": heuristic_score,
    }


def candidate_raw_features(
    item_id: str,
    detail: dict[str, Any],
    history: list[str],
    fusion_rank_1_based: int,
    num_candidates: int,
    resources: FeatureResources,
) -> dict[str, float]:
    raw = base.candidate_raw_features(detail, fusion_rank_1_based, num_candidates)
    raw.update(history_raw_features(item_id, detail, history, resources))
    return raw


def vector_from_raw(raw: dict[str, float]) -> list[float]:
    return [float(raw[name]) for name in FEATURE_NAMES]


def fit_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sums = [0.0 for _ in FEATURE_NAMES]
    sums_sq = [0.0 for _ in FEATURE_NAMES]
    count = 0
    for row in rows:
        if row["p2_split"] != "valid_fit":
            continue
        for candidate in row["candidates"]:
            values = candidate["features_raw_vector"]
            for idx, value in enumerate(values):
                sums[idx] += value
                sums_sq[idx] += value * value
            count += 1
    if count == 0:
        raise ValueError("No valid_fit candidate feature rows found.")
    means = [value / count for value in sums]
    stds: list[float] = []
    for idx, mean in enumerate(means):
        variance = max(sums_sq[idx] / count - mean * mean, 0.0)
        std = math.sqrt(variance)
        stds.append(std if std > 1e-12 else 1.0)
    return {"feature_names": FEATURE_NAMES, "mean": means, "std": stds, "count": count}


def apply_stats(rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    means = stats["mean"]
    stds = stats["std"]
    for row in rows:
        for candidate in row["candidates"]:
            raw = candidate["features_raw_vector"]
            candidate["features"] = [(value - means[idx]) / stds[idx] for idx, value in enumerate(raw)]


def build_feature_resources(args: argparse.Namespace) -> FeatureResources:
    base.assert_file(args.train_csv)
    base.assert_file(args.item_emb)
    base.assert_file(args.row_index)
    popularity = train_popularity(args.train_csv)
    max_log_pop = max([math.log1p(value) for value in popularity.values()], default=0.0)
    matrix, row_index = load_embeddings(args.item_emb, args.row_index)
    return FeatureResources(
        popularity=popularity,
        max_log_pop=max_log_pop,
        matrix=matrix,
        row_index=row_index,
        sid_rank_weight=args.sid_rank_weight,
        source_weight=args.source_weight,
        popularity_weight=args.popularity_weight,
        history_cosine_weight=args.history_cosine_weight,
        recent_cosine_weight=args.recent_cosine_weight,
        bucket_penalty_weight=args.bucket_penalty_weight,
    )


def build_dataset(root: Path, resources: FeatureResources) -> list[dict[str, Any]]:
    manifest_path = root / "p2_valid_split/valid_split_manifest.jsonl"
    fusion_path = root / "fusion/dual_fused_candidates.jsonl"
    heuristic_path = root / "rerank/reranked_candidates.jsonl"
    for path in [manifest_path, fusion_path, heuristic_path]:
        base.assert_file(path)
    manifest = base.load_manifest(manifest_path)
    fusion_rows = base.index_rows(base.read_jsonl(fusion_path), "fusion candidates")
    heuristic_rows = base.index_rows(base.read_jsonl(heuristic_path), "heuristic rerank")
    if set(manifest) != set(fusion_rows) or set(manifest) != set(heuristic_rows):
        raise ValueError(
            "P2 manifest, fusion rows, and heuristic rows are not aligned: "
            f"manifest={len(manifest)} fusion={len(fusion_rows)} heuristic={len(heuristic_rows)}"
        )

    dataset: list[dict[str, Any]] = []
    for row_id in sorted(manifest, key=base.item_sort_key):
        manifest_row = manifest[row_id]
        fusion_row = fusion_rows[row_id]
        heuristic_row = heuristic_rows[row_id]
        target_item_id = str(fusion_row.get("target_item_id", "")).strip()
        if not target_item_id:
            raise ValueError(f"Missing target_item_id for row_index={row_id}")
        if str(heuristic_row.get("target_item_id", "")).strip() != target_item_id:
            raise ValueError(f"Target mismatch for row_index={row_id}")

        candidate_items = base.list_field(fusion_row, "candidate_item_ids")
        heuristic_items = base.list_field(heuristic_row, "reranked_item_ids")
        history = base.list_field(fusion_row, "history_item_id")
        details = base.detail_by_item(fusion_row)
        candidates: list[dict[str, Any]] = []
        for idx, item_id in enumerate(candidate_items):
            detail = details.get(item_id, {})
            raw = candidate_raw_features(item_id, detail, history, idx + 1, len(candidate_items), resources)
            candidates.append(
                {
                    "item_id": item_id,
                    "original_rank_0_based": idx,
                    "features_raw": raw,
                    "features_raw_vector": vector_from_raw(raw),
                }
            )
        dataset.append(
            {
                "row_index": row_id,
                "p2_split": str(manifest_row.get("p2_split", "")),
                "target_item_id": target_item_id,
                "history_item_id": history,
                "candidate_item_ids": candidate_items,
                "heuristic_item_ids": heuristic_items,
                "candidates": candidates,
                "fusion_rank_0_based": base.rank_of(target_item_id, candidate_items),
                "heuristic_rank_0_based": base.rank_of(target_item_id, heuristic_items),
            }
        )
    return dataset


def selected_negative_indices(row: dict[str, Any], max_negatives: int) -> list[int]:
    target = row["target_item_id"]
    negatives = [
        idx
        for idx, candidate in enumerate(row["candidates"])
        if candidate["item_id"] != target
    ]
    if max_negatives <= 0 or len(negatives) <= max_negatives:
        return negatives
    return negatives[:max_negatives]


def train_pairwise_model(
    rows: list[dict[str, Any]],
    learning_rate: float,
    l2: float,
    epochs: int,
    max_negatives: int,
    seed: int,
) -> tuple[list[float], dict[str, Any]]:
    train_rows = [
        row
        for row in rows
        if row["p2_split"] == "valid_fit" and row["fusion_rank_0_based"] is not None
    ]
    if not train_rows:
        raise ValueError("No valid_fit rows with positive candidate found.")
    weights = [0.0 for _ in FEATURE_NAMES]
    rng = random.Random(seed)
    updates = 0
    positives = len(train_rows)
    negatives_per_epoch = 0
    for epoch in range(epochs):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        lr = learning_rate / (1.0 + 0.08 * epoch)
        for row_idx in order:
            row = train_rows[row_idx]
            target = row["target_item_id"]
            pos_idx = base.rank_of(target, [candidate["item_id"] for candidate in row["candidates"]])
            if pos_idx is None:
                continue
            pos_features = row["candidates"][pos_idx]["features"]
            negatives = selected_negative_indices(row, max_negatives)
            if epoch == 0:
                negatives_per_epoch += len(negatives)
            rng.shuffle(negatives)
            for neg_idx in negatives:
                neg_features = row["candidates"][neg_idx]["features"]
                diff = [pos - neg for pos, neg in zip(pos_features, neg_features)]
                margin = base.dot(weights, diff)
                scale = base.sigmoid_neg_margin(margin)
                for idx, value in enumerate(diff):
                    weights[idx] = weights[idx] * (1.0 - lr * l2) + lr * scale * value
                updates += 1
    train_info = {
        "positive_queries": positives,
        "pair_updates": updates,
        "negative_pairs_per_epoch": negatives_per_epoch,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "max_negatives_per_positive": max_negatives,
    }
    return weights, train_info


def score_candidate(candidate: dict[str, Any], weights: list[float]) -> float:
    return base.dot(weights, candidate["features"])


def learned_order(row: dict[str, Any], weights: list[float]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for candidate in row["candidates"]:
        scored.append(
            {
                "item_id": candidate["item_id"],
                "score": score_candidate(candidate, weights),
                "original_rank_0_based": candidate["original_rank_0_based"],
                "features_raw": candidate["features_raw"],
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["original_rank_0_based"], base.item_sort_key(item["item_id"])))
    return scored


def evaluate_rows(rows: list[dict[str, Any]], weights: list[float], ks: list[int]) -> dict[str, Any]:
    fusion_ranks: list[int | None] = []
    heuristic_ranks: list[int | None] = []
    learned_ranks: list[int | None] = []
    for row in rows:
        target = row["target_item_id"]
        learned_items = [item["item_id"] for item in learned_order(row, weights)]
        fusion_ranks.append(row["fusion_rank_0_based"])
        heuristic_ranks.append(row["heuristic_rank_0_based"])
        learned_ranks.append(base.rank_of(target, learned_items))
    metrics = {
        "num_samples": len(rows),
        "candidate_coverage": base.rate(sum(rank is not None for rank in fusion_ranks), len(rows)),
        "fusion_rrf": base.rank_metrics(fusion_ranks, ks),
        "heuristic_rerank": base.rank_metrics(heuristic_ranks, ks),
        "learned_rerank": base.rank_metrics(learned_ranks, ks),
    }
    deltas: dict[str, Any] = {}
    for k in ks:
        deltas[f"learned_minus_heuristic_hr@{k}"] = (
            metrics["learned_rerank"][f"hr@{k}"] - metrics["heuristic_rerank"][f"hr@{k}"]
        )
        deltas[f"learned_minus_heuristic_ndcg@{k}"] = (
            metrics["learned_rerank"][f"ndcg@{k}"] - metrics["heuristic_rerank"][f"ndcg@{k}"]
        )
        deltas[f"learned_minus_fusion_hr@{k}"] = (
            metrics["learned_rerank"][f"hr@{k}"] - metrics["fusion_rrf"][f"hr@{k}"]
        )
        deltas[f"learned_minus_fusion_ndcg@{k}"] = (
            metrics["learned_rerank"][f"ndcg@{k}"] - metrics["fusion_rrf"][f"ndcg@{k}"]
        )
    metrics["delta"] = deltas
    return metrics


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "valid_fit": [row for row in rows if row["p2_split"] == "valid_fit"],
        "valid_select": [row for row in rows if row["p2_split"] == "valid_select"],
        "all_valid": rows,
    }


def flatten_metrics(config_id: str, split_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    learned = metrics["learned_rerank"]
    heuristic = metrics["heuristic_rerank"]
    fusion = metrics["fusion_rrf"]
    delta = metrics["delta"]
    return {
        "config_id": config_id,
        "split": split_name,
        "num_samples": metrics["num_samples"],
        "candidate_coverage": metrics["candidate_coverage"],
        "learned_hr20": learned.get("hr@20"),
        "learned_ndcg20": learned.get("ndcg@20"),
        "heuristic_hr20": heuristic.get("hr@20"),
        "heuristic_ndcg20": heuristic.get("ndcg@20"),
        "fusion_hr20": fusion.get("hr@20"),
        "fusion_ndcg20": fusion.get("ndcg@20"),
        "learned_minus_heuristic_hr20": delta.get("learned_minus_heuristic_hr@20"),
        "learned_minus_heuristic_ndcg20": delta.get("learned_minus_heuristic_ndcg@20"),
        "learned_minus_fusion_hr20": delta.get("learned_minus_fusion_hr@20"),
        "learned_minus_fusion_ndcg20": delta.get("learned_minus_fusion_ndcg@20"),
    }


def train_grid(rows: list[dict[str, Any]], args: argparse.Namespace, ks: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    splits = split_rows(rows)
    grid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    config_idx = 0
    for learning_rate in args.learning_rate_grid:
        for l2 in args.l2_grid:
            config_id = f"hist_lr{learning_rate:g}_l2{l2:g}_ep{args.epochs}_neg{args.max_negatives_per_positive}"
            weights, train_info = train_pairwise_model(
                rows,
                learning_rate=learning_rate,
                l2=l2,
                epochs=args.epochs,
                max_negatives=args.max_negatives_per_positive,
                seed=args.seed + config_idx,
            )
            config_metrics = {
                split_name: evaluate_rows(split_data, weights, ks)
                for split_name, split_data in splits.items()
            }
            flat_select = flatten_metrics(config_id, "valid_select", config_metrics["valid_select"])
            grid_record = {
                "config_id": config_id,
                "learning_rate": learning_rate,
                "l2": l2,
                "epochs": args.epochs,
                "max_negatives_per_positive": args.max_negatives_per_positive,
                "weights": weights,
                "train_info": train_info,
                "metrics": config_metrics,
                **flat_select,
            }
            grid_rows.append(grid_record)
            key = (
                flat_select["learned_hr20"],
                flat_select["learned_ndcg20"],
                -l2,
                -learning_rate,
            )
            if best is None or key > best["selection_key"]:
                best = {**grid_record, "selection_key": key}
            config_idx += 1
    if best is None:
        raise ValueError("No history-aware ranker config was trained.")
    return best, grid_rows


def make_reranked_outputs(
    rows: list[dict[str, Any]],
    weights: list[float],
    detail_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jsonl_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    for row in rows:
        scored = learned_order(row, weights)
        learned_items = [item["item_id"] for item in scored]
        learned_rank = base.rank_of(row["target_item_id"], learned_items)
        jsonl_rows.append(
            {
                "row_index": row["row_index"],
                "p2_split": row["p2_split"],
                "target_item_id": row["target_item_id"],
                "history_item_id": row["history_item_id"],
                "candidate_item_ids": row["candidate_item_ids"],
                "reranked_item_ids": learned_items,
                "history_aware_reranked_item_ids": learned_items,
                "history_aware_details": scored[: max(detail_limit, 0)],
                "candidate_hit_rank_0_based": row["fusion_rank_0_based"],
                "heuristic_hit_rank_0_based": row["heuristic_rank_0_based"],
                "history_aware_hit_rank_0_based": learned_rank,
            }
        )
        per_sample_rows.append(
            {
                "row_index": row["row_index"],
                "p2_split": row["p2_split"],
                "target_item_id": row["target_item_id"],
                "fusion_hit_rank_0_based": "" if row["fusion_rank_0_based"] is None else row["fusion_rank_0_based"],
                "heuristic_hit_rank_0_based": "" if row["heuristic_rank_0_based"] is None else row["heuristic_rank_0_based"],
                "history_aware_hit_rank_0_based": "" if learned_rank is None else learned_rank,
                "num_candidates": len(row["candidate_item_ids"]),
            }
        )
    return jsonl_rows, per_sample_rows


def report_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = report["selected_config_id"]
    for split_name, metrics in report["split_metrics"].items():
        for model_name in ["fusion_rrf", "heuristic_rerank", "history_aware_rerank"]:
            model_metrics = metrics["learned_rerank"] if model_name == "history_aware_rerank" else metrics[model_name]
            for metric, value in model_metrics.items():
                if metric == "num_samples":
                    continue
                rows.append(
                    {
                        "config_id": selected,
                        "split": split_name,
                        "model": model_name,
                        "metric": metric,
                        "value": value,
                    }
                )
        for metric, value in metrics["delta"].items():
            rows.append(
                {
                    "config_id": selected,
                    "split": split_name,
                    "model": "delta",
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def make_grid_csv_rows(grid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in grid_rows:
        for split_name in ["valid_fit", "valid_select", "all_valid"]:
            flat = flatten_metrics(row["config_id"], split_name, row["metrics"][split_name])
            flat.update(
                {
                    "learning_rate": row["learning_rate"],
                    "l2": row["l2"],
                    "epochs": row["epochs"],
                    "max_negatives_per_positive": row["max_negatives_per_positive"],
                }
            )
            rows.append(flat)
    return rows


def weight_rows(weights: list[float], stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(FEATURE_NAMES):
        rows.append(
            {
                "feature": name,
                "weight": weights[idx],
                "abs_weight": abs(weights[idx]),
                "mean": stats["mean"][idx],
                "std": stats["std"][idx],
            }
        )
    rows.sort(key=lambda row: (-row["abs_weight"], row["feature"]))
    return rows


def make_markdown(report: dict[str, Any]) -> str:
    selected = report["selected_config_id"]
    metrics = report["split_metrics"]
    freeze = report["selection"]["freeze_recommendation"]
    lines = [
        "# P2-3 History-aware Learned Ranker",
        "",
        f"- split: `{report['split']}`",
        f"- category: `{report['category']}`",
        f"- candidate_mode: `{report['candidate_mode']}`",
        f"- selected_config: `{selected}`",
        f"- overall_ok: `{report['overall_ok']}`",
        "",
        "## Selection",
        "",
        f"- decision: `{freeze['decision']}`",
        f"- reason: `{freeze['reason']}`",
        "",
        "## Metrics",
        "",
        "| Split | Model | HR@20 | NDCG@20 |",
        "|---|---|---:|---:|",
    ]
    for split_name in ["valid_fit", "valid_select", "all_valid"]:
        for model_name in ["fusion_rrf", "heuristic_rerank", "learned_rerank"]:
            values = metrics[split_name][model_name]
            label = "history_aware_rerank" if model_name == "learned_rerank" else model_name
            lines.append(
                f"| {split_name} | {label} | {values.get('hr@20', 0):.6f} | {values.get('ndcg@20', 0):.6f} |"
            )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- Model fitting uses `valid_fit` only.",
            "- Model selection uses `valid_select` only.",
            "- Popularity is computed from train.csv only.",
            "- History/recent cosine uses query history only.",
            "- No test path is read or written.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = args.out_dir or (root / "p2_history_ranker")
    ks = sorted(set(k for k in args.ks if k > 0))
    if 20 not in ks:
        ks.append(20)
        ks = sorted(ks)

    split_summary_path = root / "p2_valid_split/valid_split_summary.json"
    p1_consistency_path = root / "summary/consistency_report.json"
    p2_minimal_report_path = root / "p2_minimal_ranker/ranker_report.json"
    for path in [split_summary_path, p1_consistency_path, p2_minimal_report_path]:
        base.assert_file(path)
    split_summary = base.read_json(split_summary_path)
    p1_consistency = base.read_json(p1_consistency_path)
    p2_minimal_report = base.read_json(p2_minimal_report_path)

    resources = build_feature_resources(args)
    rows = build_dataset(root, resources)
    stats = fit_stats(rows)
    apply_stats(rows, stats)
    selected, grid_rows = train_grid(rows, args, ks)
    selected_weights = selected["weights"]
    split_metrics = {
        split_name: evaluate_rows(split_data, selected_weights, ks)
        for split_name, split_data in split_rows(rows).items()
    }
    output_rows, per_sample_rows = make_reranked_outputs(rows, selected_weights, args.detail_limit)

    select_metrics = split_metrics["valid_select"]
    learned_select = select_metrics["learned_rerank"]
    heuristic_select = select_metrics["heuristic_rerank"]
    hr_gain = learned_select["hr@20"] - heuristic_select["hr@20"]
    ndcg_gain = learned_select["ndcg@20"] - heuristic_select["ndcg@20"]
    should_freeze = hr_gain > args.min_select_improvement and ndcg_gain > args.min_select_improvement
    freeze_recommendation = {
        "decision": "freeze_p2_history_ranker" if should_freeze else "do_not_freeze_p2_history_ranker",
        "reason": (
            f"valid_select HR@20 gain={hr_gain:.6f}, NDCG@20 gain={ndcg_gain:.6f}; "
            f"minimum required improvement={args.min_select_improvement:.6f}"
        ),
        "valid_select_hr20_gain": hr_gain,
        "valid_select_ndcg20_gain": ndcg_gain,
    }

    input_paths = [
        root / "p2_valid_split/valid_split_manifest.jsonl",
        split_summary_path,
        root / "fusion/dual_fused_candidates.jsonl",
        root / "rerank/reranked_candidates.jsonl",
        p1_consistency_path,
        p2_minimal_report_path,
        args.train_csv,
        args.item_emb,
        args.row_index,
    ]
    output_paths = [
        out_dir / "history_reranked_candidates.jsonl",
        out_dir / "per_sample_metrics.csv",
        out_dir / "ranker_report.json",
        out_dir / "ranker_report.csv",
        out_dir / "ranker_report.md",
        out_dir / "model.json",
        out_dir / "model_grid.csv",
        out_dir / "feature_weights.csv",
    ]
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    for path in input_paths[:6]:
        add_check(f"input under root: {path.as_posix()}", base.is_under(path, root), path.as_posix())
        add_check(f"input has no test path: {path.as_posix()}", base.no_test_path(path), path.as_posix())
    for path in input_paths[6:]:
        add_check(f"feature input has no test path: {path.as_posix()}", base.no_test_path(path), path.as_posix())
    for path in output_paths:
        add_check(f"output under root: {path.as_posix()}", base.is_under(path, root), path.as_posix())
        add_check(f"output has no test path: {path.as_posix()}", base.no_test_path(path), path.as_posix())
    split_counts = Counter(row["p2_split"] for row in rows)
    add_check("P1 consistency overall_ok", p1_consistency.get("overall_ok") is True, p1_consistency.get("overall_ok"))
    add_check("P2-1 split overall_ok", split_summary.get("overall_ok") is True, split_summary.get("overall_ok"))
    add_check(
        "P2-2 minimal ranker not frozen",
        p2_minimal_report.get("selection", {}).get("freeze_recommendation", {}).get("decision")
        == "do_not_freeze_p2_minimal_ranker",
        p2_minimal_report.get("selection", {}).get("freeze_recommendation", {}).get("decision"),
    )
    add_check("category Industrial", split_summary.get("category") == args.category, split_summary.get("category"))
    add_check("split valid", split_summary.get("split") == args.split, split_summary.get("split"))
    add_check("candidate mode exact", split_summary.get("candidate_mode") == args.candidate_mode, split_summary.get("candidate_mode"))
    add_check("row count matches P2-1", len(rows) == int(split_summary.get("num_samples", -1)), len(rows))
    add_check("valid_fit count matches P2-1", split_counts["valid_fit"] == int(split_summary.get("valid_fit_count", -1)), split_counts["valid_fit"])
    add_check("valid_select count matches P2-1", split_counts["valid_select"] == int(split_summary.get("valid_select_count", -1)), split_counts["valid_select"])
    add_check("trained positive queries exist", selected["train_info"]["positive_queries"] > 0, selected["train_info"]["positive_queries"])
    add_check("selection split is valid_select", True, "valid_select")
    add_check("feature stats include history features", set(HISTORY_FEATURE_NAMES).issubset(set(stats["feature_names"])), HISTORY_FEATURE_NAMES)

    model = {
        "model_type": "history_aware_linear_pairwise_logistic",
        "feature_names": FEATURE_NAMES,
        "minimal_feature_names": MINIMAL_FEATURE_NAMES,
        "history_feature_names": HISTORY_FEATURE_NAMES,
        "weights": selected_weights,
        "feature_stats": stats,
        "selected_config_id": selected["config_id"],
        "selected_config": {
            "learning_rate": selected["learning_rate"],
            "l2": selected["l2"],
            "epochs": selected["epochs"],
            "max_negatives_per_positive": selected["max_negatives_per_positive"],
            "seed": args.seed,
        },
        "heuristic_component_weights": {
            "sid_rank_weight": args.sid_rank_weight,
            "source_weight": args.source_weight,
            "popularity_weight": args.popularity_weight,
            "history_cosine_weight": args.history_cosine_weight,
            "recent_cosine_weight": args.recent_cosine_weight,
            "bucket_penalty_weight": args.bucket_penalty_weight,
        },
        "sort_tie_breaker": ["score desc", "original_rank_0_based asc", "item_id asc"],
        "training_policy": "fit weights on valid_fit; select config by valid_select hr@20 then ndcg@20",
    }
    report = {
        "split": args.split,
        "category": args.category,
        "candidate_mode": args.candidate_mode,
        "input_root": root.as_posix(),
        "output_dir": out_dir.as_posix(),
        "inputs": {
            "manifest": (root / "p2_valid_split/valid_split_manifest.jsonl").as_posix(),
            "split_summary": split_summary_path.as_posix(),
            "fusion_candidates": (root / "fusion/dual_fused_candidates.jsonl").as_posix(),
            "heuristic_rerank": (root / "rerank/reranked_candidates.jsonl").as_posix(),
            "p2_minimal_ranker": p2_minimal_report_path.as_posix(),
            "train_csv": args.train_csv.as_posix(),
            "item_emb": args.item_emb.as_posix(),
            "row_index": args.row_index.as_posix(),
        },
        "outputs": {path.stem: path.as_posix() for path in output_paths},
        "topk": ks,
        "feature_names": FEATURE_NAMES,
        "history_feature_names": HISTORY_FEATURE_NAMES,
        "selected_config_id": selected["config_id"],
        "selected_config": model["selected_config"],
        "train_info": selected["train_info"],
        "selection": {
            "criterion": "max valid_select hr@20, tie valid_select ndcg@20, tie lower l2/lr",
            "freeze_recommendation": freeze_recommendation,
        },
        "split_counts": dict(split_counts),
        "split_metrics": split_metrics,
        "checks": checks,
        "overall_ok": all(check["ok"] for check in checks),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    base.write_jsonl(out_dir / "history_reranked_candidates.jsonl", output_rows)
    base.write_csv(
        out_dir / "per_sample_metrics.csv",
        [
            "row_index",
            "p2_split",
            "target_item_id",
            "fusion_hit_rank_0_based",
            "heuristic_hit_rank_0_based",
            "history_aware_hit_rank_0_based",
            "num_candidates",
        ],
        per_sample_rows,
    )
    base.write_json(out_dir / "model.json", model)
    base.write_json(out_dir / "ranker_report.json", report)
    base.write_csv(out_dir / "ranker_report.csv", ["config_id", "split", "model", "metric", "value"], report_csv_rows(report))
    base.write_csv(
        out_dir / "model_grid.csv",
        [
            "config_id",
            "learning_rate",
            "l2",
            "epochs",
            "max_negatives_per_positive",
            "split",
            "num_samples",
            "candidate_coverage",
            "learned_hr20",
            "learned_ndcg20",
            "heuristic_hr20",
            "heuristic_ndcg20",
            "fusion_hr20",
            "fusion_ndcg20",
            "learned_minus_heuristic_hr20",
            "learned_minus_heuristic_ndcg20",
            "learned_minus_fusion_hr20",
            "learned_minus_fusion_ndcg20",
        ],
        make_grid_csv_rows(grid_rows),
    )
    base.write_csv(out_dir / "feature_weights.csv", ["feature", "weight", "abs_weight", "mean", "std"], weight_rows(selected_weights, stats))
    with open(out_dir / "ranker_report.md", "w", encoding="utf-8") as f:
        f.write(make_markdown(report))

    if not report["overall_ok"]:
        failed = [check for check in checks if not check["ok"]]
        raise SystemExit(f"P2-3 history-aware ranker finished but checks failed: {failed}")

    print(f"Wrote P2 history-aware ranker report: {out_dir / 'ranker_report.md'}")
    print(
        "P2-3 summary: "
        f"selected={selected['config_id']} "
        f"valid_select_history_hr20={learned_select['hr@20']} "
        f"valid_select_heuristic_hr20={heuristic_select['hr@20']} "
        f"decision={freeze_recommendation['decision']}"
    )


if __name__ == "__main__":
    main()
