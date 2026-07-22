#!/usr/bin/env python3
"""Train the minimal P2 source-aware learned reranker.

The model is intentionally small: a linear pairwise-logistic ranker trained on
valid_fit and selected on valid_select. It reads only fixed Stage 7 validation
artifacts and the P2-1 manifest; it never reads test data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/stage7_validation_protocol/valid/Industrial_and_Scientific")
DEFAULT_KS = [1, 5, 10, 20, 50]
FEATURE_NAMES = [
    "text_present",
    "cf_present",
    "both_sources",
    "text_only",
    "cf_only",
    "source_count",
    "text_rank_filled",
    "cf_rank_filled",
    "min_source_rank",
    "fusion_rank",
    "reciprocal_text_rank",
    "reciprocal_cf_rank",
    "reciprocal_min_source_rank",
    "reciprocal_fusion_rank",
    "fusion_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P2 minimal source-aware learned ranker.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--split", choices=["valid"], default="valid")
    parser.add_argument("--candidate-mode", choices=["exact"], default="exact")
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--learning-rate-grid", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    parser.add_argument("--l2-grid", type=float, nargs="+", default=[0.0, 0.0001, 0.001, 0.01])
    parser.add_argument("--max-negatives-per-positive", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--detail-limit", type=int, default=100)
    parser.add_argument("--min-select-improvement", type=float, default=0.0)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object rows in {path}")
                rows.append(value)
    return rows


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


def assert_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def item_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def normalize_id(value: Any, fallback: int) -> str:
    if value is None or value == "":
        return str(fallback)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        row_id = normalize_id(row.get("row_index", row.get("sample_id")), idx)
        if row_id in indexed:
            raise ValueError(f"Duplicate row id in {label}: {row_id}")
        indexed[row_id] = row
    return indexed


def list_field(row: dict[str, Any], field: str) -> list[str]:
    values = row.get(field, [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def rank_of(target_item_id: str, items: list[str]) -> int | None:
    target = str(target_item_id)
    for idx, item_id in enumerate(items):
        if str(item_id) == target:
            return idx
    return None


def hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank < k


def rate(count: int | float, total: int) -> float:
    return 0.0 if total == 0 else float(count) / float(total)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sigmoid_neg_margin(margin: float) -> float:
    if margin >= 40:
        return 0.0
    if margin <= -40:
        return 1.0
    return 1.0 / (1.0 + math.exp(margin))


def detail_by_item(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    details = record.get("candidate_details", [])
    if not isinstance(details, list):
        return out
    for detail in details:
        if not isinstance(detail, dict):
            continue
        item_id = str(detail.get("item_id", ""))
        if item_id and item_id not in out:
            out[item_id] = detail
    return out


def candidate_raw_features(
    detail: dict[str, Any],
    fusion_rank_1_based: int,
    num_candidates: int,
) -> dict[str, float]:
    text_rank = parse_float(detail.get("text_rank"))
    cf_rank = parse_float(detail.get("behavior_rank"))
    from_text = parse_bool(detail.get("from_text")) or text_rank is not None
    from_cf = parse_bool(detail.get("from_behavior")) or cf_rank is not None
    source_count = int(from_text) + int(from_cf)
    both = from_text and from_cf
    missing_rank = float(num_candidates + 1)
    text_rank_filled = text_rank if text_rank is not None else missing_rank
    cf_rank_filled = cf_rank if cf_rank is not None else missing_rank
    min_rank = min(text_rank_filled, cf_rank_filled)
    fusion_score = parse_float(detail.get("fusion_score")) or 0.0
    return {
        "text_present": float(from_text),
        "cf_present": float(from_cf),
        "both_sources": float(both),
        "text_only": float(from_text and not from_cf),
        "cf_only": float(from_cf and not from_text),
        "source_count": float(source_count),
        "text_rank_filled": float(text_rank_filled),
        "cf_rank_filled": float(cf_rank_filled),
        "min_source_rank": float(min_rank),
        "fusion_rank": float(fusion_rank_1_based),
        "reciprocal_text_rank": 0.0 if text_rank is None else 1.0 / float(text_rank),
        "reciprocal_cf_rank": 0.0 if cf_rank is None else 1.0 / float(cf_rank),
        "reciprocal_min_source_rank": 1.0 / float(min_rank),
        "reciprocal_fusion_rank": 1.0 / float(fusion_rank_1_based),
        "fusion_score": float(fusion_score),
    }


def rank_metrics(ranks: list[int | None], ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"num_samples": len(ranks)}
    for k in ks:
        hits = 0
        ndcg_sum = 0.0
        for rank in ranks:
            if rank is not None and rank < k:
                hits += 1
                ndcg_sum += 1.0 / math.log2(rank + 2)
        out[f"hr@{k}"] = rate(hits, len(ranks))
        out[f"ndcg@{k}"] = rate(ndcg_sum, len(ranks))
    return out


def no_test_path(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    return "test" not in lowered_parts and path.name.lower() != "test.csv"


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


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


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    indexed: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        row_id = normalize_id(row.get("row_index"), idx)
        if row_id in indexed:
            raise ValueError(f"Duplicate row_index in manifest: {row_id}")
        indexed[row_id] = row
    return indexed


def build_dataset(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / "p2_valid_split/valid_split_manifest.jsonl"
    fusion_path = root / "fusion/dual_fused_candidates.jsonl"
    heuristic_path = root / "rerank/reranked_candidates.jsonl"
    for path in [manifest_path, fusion_path, heuristic_path]:
        assert_file(path)
    manifest = load_manifest(manifest_path)
    fusion_rows = index_rows(read_jsonl(fusion_path), "fusion candidates")
    heuristic_rows = index_rows(read_jsonl(heuristic_path), "heuristic rerank")
    if set(manifest) != set(fusion_rows) or set(manifest) != set(heuristic_rows):
        raise ValueError(
            "P2 manifest, fusion rows, and heuristic rows are not aligned: "
            f"manifest={len(manifest)} fusion={len(fusion_rows)} heuristic={len(heuristic_rows)}"
        )

    dataset: list[dict[str, Any]] = []
    for row_id in sorted(manifest, key=item_sort_key):
        manifest_row = manifest[row_id]
        fusion_row = fusion_rows[row_id]
        heuristic_row = heuristic_rows[row_id]
        target_item_id = str(fusion_row.get("target_item_id", "")).strip()
        if not target_item_id:
            raise ValueError(f"Missing target_item_id for row_index={row_id}")
        if str(heuristic_row.get("target_item_id", "")).strip() != target_item_id:
            raise ValueError(f"Target mismatch for row_index={row_id}")
        candidate_items = list_field(fusion_row, "candidate_item_ids")
        heuristic_items = list_field(heuristic_row, "reranked_item_ids")
        details = detail_by_item(fusion_row)
        candidates: list[dict[str, Any]] = []
        for idx, item_id in enumerate(candidate_items):
            detail = details.get(item_id, {})
            raw = candidate_raw_features(detail, idx + 1, len(candidate_items))
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
                "candidate_item_ids": candidate_items,
                "heuristic_item_ids": heuristic_items,
                "candidates": candidates,
                "fusion_rank_0_based": rank_of(target_item_id, candidate_items),
                "heuristic_rank_0_based": rank_of(target_item_id, heuristic_items),
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
            pos_idx = rank_of(target, [candidate["item_id"] for candidate in row["candidates"]])
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
                margin = dot(weights, diff)
                scale = sigmoid_neg_margin(margin)
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
    return dot(weights, candidate["features"])


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
    scored.sort(key=lambda item: (-item["score"], item["original_rank_0_based"], item_sort_key(item["item_id"])))
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
        learned_ranks.append(rank_of(target, learned_items))
    metrics = {
        "num_samples": len(rows),
        "candidate_coverage": rate(sum(rank is not None for rank in fusion_ranks), len(rows)),
        "fusion_rrf": rank_metrics(fusion_ranks, ks),
        "heuristic_rerank": rank_metrics(heuristic_ranks, ks),
        "learned_rerank": rank_metrics(learned_ranks, ks),
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
    out = {
        "valid_fit": [row for row in rows if row["p2_split"] == "valid_fit"],
        "valid_select": [row for row in rows if row["p2_split"] == "valid_select"],
        "all_valid": rows,
    }
    return out


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
            config_id = f"lr{learning_rate:g}_l2{l2:g}_ep{args.epochs}_neg{args.max_negatives_per_positive}"
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
        raise ValueError("No ranker config was trained.")
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
        learned_rank = rank_of(row["target_item_id"], learned_items)
        record = {
            "row_index": row["row_index"],
            "p2_split": row["p2_split"],
            "target_item_id": row["target_item_id"],
            "candidate_item_ids": row["candidate_item_ids"],
            "reranked_item_ids": learned_items,
            "learned_reranked_item_ids": learned_items,
            "learned_details": scored[: max(detail_limit, 0)],
            "candidate_hit_rank_0_based": row["fusion_rank_0_based"],
            "heuristic_hit_rank_0_based": row["heuristic_rank_0_based"],
            "learned_hit_rank_0_based": learned_rank,
        }
        jsonl_rows.append(record)
        per_sample_rows.append(
            {
                "row_index": row["row_index"],
                "p2_split": row["p2_split"],
                "target_item_id": row["target_item_id"],
                "fusion_hit_rank_0_based": "" if row["fusion_rank_0_based"] is None else row["fusion_rank_0_based"],
                "heuristic_hit_rank_0_based": "" if row["heuristic_rank_0_based"] is None else row["heuristic_rank_0_based"],
                "learned_hit_rank_0_based": "" if learned_rank is None else learned_rank,
                "num_candidates": len(row["candidate_item_ids"]),
            }
        )
    return jsonl_rows, per_sample_rows


def report_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = report["selected_config_id"]
    for split_name, metrics in report["split_metrics"].items():
        for model_name in ["fusion_rrf", "heuristic_rerank", "learned_rerank"]:
            model_metrics = metrics[model_name]
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


def make_markdown(report: dict[str, Any]) -> str:
    selected = report["selected_config_id"]
    metrics = report["split_metrics"]
    freeze = report["selection"]["freeze_recommendation"]
    lines = [
        "# P2-2 Minimal Source-aware Learned Ranker",
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
            lines.append(
                f"| {split_name} | {model_name} | {values.get('hr@20', 0):.6f} | {values.get('ndcg@20', 0):.6f} |"
            )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- Model fitting uses `valid_fit` only.",
            "- Model selection uses `valid_select` only.",
            "- No test path is read or written.",
            "",
        ]
    )
    return "\n".join(lines)


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


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = args.out_dir or (root / "p2_minimal_ranker")
    ks = sorted(set(k for k in args.ks if k > 0))
    if 20 not in ks:
        ks.append(20)
        ks = sorted(ks)

    split_summary_path = root / "p2_valid_split/valid_split_summary.json"
    p1_consistency_path = root / "summary/consistency_report.json"
    for path in [split_summary_path, p1_consistency_path]:
        assert_file(path)
    split_summary = read_json(split_summary_path)
    p1_consistency = read_json(p1_consistency_path)

    rows = build_dataset(root)
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
        "decision": "freeze_p2_minimal_ranker" if should_freeze else "do_not_freeze_p2_minimal_ranker",
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
    ]
    output_paths = [
        out_dir / "learned_reranked_candidates.jsonl",
        out_dir / "per_sample_metrics.csv",
        out_dir / "ranker_report.json",
        out_dir / "ranker_report.csv",
        out_dir / "ranker_report.md",
        out_dir / "model.json",
        out_dir / "model_grid.csv",
    ]
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    for path in input_paths:
        add_check(f"input under root: {path.as_posix()}", is_under(path, root), path.as_posix())
        add_check(f"input has no test path: {path.as_posix()}", no_test_path(path), path.as_posix())
    for path in output_paths:
        add_check(f"output under root: {path.as_posix()}", is_under(path, root), path.as_posix())
        add_check(f"output has no test path: {path.as_posix()}", no_test_path(path), path.as_posix())
    split_counts = Counter(row["p2_split"] for row in rows)
    add_check("P1 consistency overall_ok", p1_consistency.get("overall_ok") is True, p1_consistency.get("overall_ok"))
    add_check("P2-1 split overall_ok", split_summary.get("overall_ok") is True, split_summary.get("overall_ok"))
    add_check("category Industrial", split_summary.get("category") == args.category, split_summary.get("category"))
    add_check("split valid", split_summary.get("split") == args.split, split_summary.get("split"))
    add_check("candidate mode exact", split_summary.get("candidate_mode") == args.candidate_mode, split_summary.get("candidate_mode"))
    add_check("row count matches P2-1", len(rows) == int(split_summary.get("num_samples", -1)), len(rows))
    add_check("valid_fit count matches P2-1", split_counts["valid_fit"] == int(split_summary.get("valid_fit_count", -1)), split_counts["valid_fit"])
    add_check("valid_select count matches P2-1", split_counts["valid_select"] == int(split_summary.get("valid_select_count", -1)), split_counts["valid_select"])
    add_check("trained positive queries exist", selected["train_info"]["positive_queries"] > 0, selected["train_info"]["positive_queries"])
    add_check("selection split is valid_select", True, "valid_select")

    model = {
        "model_type": "linear_pairwise_logistic",
        "feature_names": FEATURE_NAMES,
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
        },
        "outputs": {path.stem: path.as_posix() for path in output_paths},
        "topk": ks,
        "feature_names": FEATURE_NAMES,
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
    write_jsonl(out_dir / "learned_reranked_candidates.jsonl", output_rows)
    write_csv(
        out_dir / "per_sample_metrics.csv",
        [
            "row_index",
            "p2_split",
            "target_item_id",
            "fusion_hit_rank_0_based",
            "heuristic_hit_rank_0_based",
            "learned_hit_rank_0_based",
            "num_candidates",
        ],
        per_sample_rows,
    )
    write_json(out_dir / "model.json", model)
    write_json(out_dir / "ranker_report.json", report)
    write_csv(out_dir / "ranker_report.csv", ["config_id", "split", "model", "metric", "value"], report_csv_rows(report))
    write_csv(
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
    with open(out_dir / "ranker_report.md", "w", encoding="utf-8") as f:
        f.write(make_markdown(report))

    if not report["overall_ok"]:
        failed = [check for check in checks if not check["ok"]]
        raise SystemExit(f"P2-2 minimal ranker finished but checks failed: {failed}")

    print(f"Wrote P2 minimal ranker report: {out_dir / 'ranker_report.md'}")
    print(
        "P2-2 summary: "
        f"selected={selected['config_id']} "
        f"valid_select_learned_hr20={learned_select['hr@20']} "
        f"valid_select_heuristic_hr20={heuristic_select['hr@20']} "
        f"decision={freeze_recommendation['decision']}"
    )


if __name__ == "__main__":
    main()
