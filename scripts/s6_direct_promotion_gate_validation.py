#!/usr/bin/env python3
"""S6-6R validation-only CF-anchored direct-promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import s5_auxiliary_fusion_conversion as s5
import s6_frozen_ranker_validation as s6fr
import s6_lightweight_ranker_validation as s6lw


ROOT = Path(__file__).resolve().parents[1]
CATEGORY = "Industrial_and_Scientific"
RUN_ID = "promotion_v1"
SPLITS = ("valid_fit", "valid_select", "valid_gate")
FIT_SELECT_SPLITS = ("valid_fit", "valid_select")
KS = (1, 5, 10, 20)
DEFAULT_PROTOCOL = ROOT / "configs/s6_cost_aware_aux/direct_promotion_protocol.json"
DEFAULT_OUTPUT_ROOT = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific"
DEFAULT_REPORT = DEFAULT_OUTPUT_ROOT / "s6_6r_direct_promotion_gate_report.json"
FORBIDDEN_FEATURE_NAME_PARTS = (
    "target",
    "label",
    "is_correct",
    "post_ranking",
    "outcome",
)


def np_module():
    return s6lw.np_module()


def file_sha256(path: Path) -> str:
    return s5.file_sha256(path)


def read_json(path: Path) -> Any:
    return s5.read_json(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return s5.read_jsonl(path)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reject_existing(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        raise FileExistsError(f"refusing to overwrite existing S6-6R artifact: {path}")


def reject_test_path(path: Path) -> None:
    lowered = [part.lower() for part in path.parts]
    if "test" in lowered or "final_test" in lowered or path.name.lower() == "test.csv":
        raise ValueError(f"S6-6R refuses test path: {path}")


def write_json(path: Path, data: Any) -> None:
    reject_existing(path)
    s5.write_json(path, data)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    reject_existing(path)
    s5.write_jsonl(path, rows)


def output_dir(output_root: Path) -> Path:
    return output_root / "direct_promotion" / RUN_ID


def split_output_dir(output_root: Path, split: str) -> Path:
    return output_dir(output_root) / split


def candidate_rank(target: str, items: list[str]) -> int | None:
    return s6fr.candidate_rank(target, items)


def hit(rank: int | None, k: int) -> bool:
    return rank is not None and rank < k


def load_protocol(path: Path) -> dict[str, Any]:
    reject_test_path(path)
    protocol = read_json(path)
    if protocol["candidate_budget_k"] != 20:
        raise ValueError("S6-6R requires K=20")
    if protocol["train_split"] != "valid_fit" or protocol["selection_split"] != "valid_select":
        raise ValueError("S6-6R requires fit/select split isolation")
    if len(protocol["max_promotions"]) * int(protocol["threshold_count"]) > 12:
        raise ValueError("S6-6R policy set must contain at most 12 configurations")
    return protocol


def feature_names() -> list[str]:
    names = [
        "direct_rank",
        "direct_reciprocal_rank",
        "direct_score",
        "direct_zscore",
        "direct_top1_margin",
        "direct_score_minus_topk_mean",
        "direct_frozen_projected_score",
        "direct_history_cosine",
        "direct_recent_cosine",
        "direct_popularity_score",
        "direct_bucket_penalty",
        "boundary_cf_rank",
        "boundary_reciprocal_cf_rank",
        "boundary_frozen_projected_score",
        "boundary_history_cosine",
        "boundary_recent_cosine",
        "boundary_popularity_score",
        "boundary_bucket_penalty",
        "score_diff_direct_minus_boundary",
        "rank_confidence_diff",
        "boundary_position",
    ]
    bad = [name for name in names if any(part in name for part in FORBIDDEN_FEATURE_NAME_PARTS)]
    if bad:
        raise ValueError(f"forbidden feature names: {bad}")
    return names


def split_samples(output_root: Path, split: str) -> list[s5.Sample]:
    samples, _ = s6fr.build_samples_for_split(output_root, Path(), split)
    return samples


def load_context():
    config, model, matrix, row_index, popularity, max_log_pop, base_config = s6lw.load_context()
    return config, model, matrix, row_index, popularity, max_log_pop, base_config


def load_union_rows(output_root: Path, split: str) -> dict[str, dict[str, Any]]:
    return {str(row["row_index"]): row for row in read_jsonl(s6fr.union_path(output_root, split))}


def detail_map(union_row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(detail["item_id"]): detail for detail in union_row.get("candidate_details", [])}


def direct_detail(detail: dict[str, Any] | None) -> dict[str, Any]:
    if not detail:
        return {}
    return detail.get("sasrec_direct_detail") or {}


def fit_normalizer(rows: list[dict[str, Any]]):
    np = np_module()
    if not rows:
        raise ValueError("cannot fit promotion normalizer without training rows")
    x = np.asarray([row["features"] for row in rows], dtype="float64")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std <= 1e-12] = 1.0
    return mean, std


def normalize_features(rows: list[dict[str, Any]], mean: Any, std: Any):
    np = np_module()
    return np.asarray([(np.asarray(row["features"], dtype="float64") - mean) / std for row in rows], dtype="float64")


def item_side_features(
    sample: s5.Sample,
    item: str,
    detail: dict[str, Any] | None,
    frozen_score: float,
    matrix: Any,
    row_index: dict[str, int],
    popularity: dict[str, int],
    max_log_pop: float,
    fusion_rank0: int,
    pool_size: int,
) -> dict[str, float]:
    cand = sample.candidates[item]
    raw = s5.raw_features(
        sample,
        cand,
        fusion_rank0 + 1,
        pool_size,
        matrix,
        row_index,
        popularity,
        max_log_pop,
        "source_independent_projection",
    )
    dd = direct_detail(detail)
    direct_rank = cand.sasrec_rank
    cf_rank = cand.cf_rank
    return {
        "direct_rank": float(direct_rank + 1) if direct_rank is not None else float(pool_size + 1),
        "direct_reciprocal_rank": 0.0 if direct_rank is None else 1.0 / float(direct_rank + 1),
        "direct_score": float(dd.get("sasrec_direct_score", 0.0) or 0.0),
        "direct_zscore": float(dd.get("sasrec_direct_zscore", 0.0) or 0.0),
        "direct_top1_margin": float(dd.get("sasrec_direct_top1_margin", 0.0) or 0.0),
        "direct_score_minus_topk_mean": float(dd.get("sasrec_direct_score_minus_topk_mean", 0.0) or 0.0),
        "frozen_projected_score": float(frozen_score),
        "history_cosine": float(raw["history_cosine"]),
        "recent_cosine": float(raw["recent_cosine"]),
        "popularity_score": float(raw["popularity_score"]),
        "bucket_penalty": float(raw["bucket_penalty"]),
        "cf_rank": float(cf_rank + 1) if cf_rank is not None else float(pool_size + 1),
        "reciprocal_cf_rank": 0.0 if cf_rank is None else 1.0 / float(cf_rank + 1),
    }


def pair_features(direct: dict[str, float], boundary: dict[str, float], boundary_position: int) -> list[float]:
    return [
        direct["direct_rank"],
        direct["direct_reciprocal_rank"],
        direct["direct_score"],
        direct["direct_zscore"],
        direct["direct_top1_margin"],
        direct["direct_score_minus_topk_mean"],
        direct["frozen_projected_score"],
        direct["history_cosine"],
        direct["recent_cosine"],
        direct["popularity_score"],
        direct["bucket_penalty"],
        boundary["cf_rank"],
        boundary["reciprocal_cf_rank"],
        boundary["frozen_projected_score"],
        boundary["history_cosine"],
        boundary["recent_cosine"],
        boundary["popularity_score"],
        boundary["bucket_penalty"],
        direct["frozen_projected_score"] - boundary["frozen_projected_score"],
        direct["direct_reciprocal_rank"] - boundary["reciprocal_cf_rank"],
        float(boundary_position + 1),
    ]


def build_split_cache(
    output_root: Path,
    split: str,
    samples: list[s5.Sample],
    base_config: dict[str, Any],
    model: dict[str, Any],
    matrix: Any,
    row_index: dict[str, int],
    popularity: dict[str, int],
    max_log_pop: float,
) -> dict[str, Any]:
    union_rows = load_union_rows(output_root, split)
    frozen_scores = s6lw.frozen_scores_for_samples(samples, base_config, model, matrix, row_index, popularity, max_log_pop)
    cache: dict[str, Any] = {}
    for sample in samples:
        union_row = union_rows[sample.sample_id]
        details = detail_map(union_row)
        base_items = s5.policy_items(sample, base_config)
        side = {}
        for rank0, item in enumerate(base_items):
            side[item] = item_side_features(
                sample,
                item,
                details.get(item),
                frozen_scores[sample.sample_id][item],
                matrix,
                row_index,
                popularity,
                max_log_pop,
                rank0,
                len(base_items),
            )
        cache[sample.sample_id] = {
            "union_row": union_row,
            "details": details,
            "base_items": base_items,
            "side_features": side,
            "input_hash": canonical_hash(
                {
                    "row_index": sample.sample_id,
                    "target": sample.target_item_id,
                    "history": sample.history_item_id,
                    "cf_items": sample.cf_items,
                    "direct_items": sample.sasrec_items,
                    "union_items": list(sample.candidates),
                }
            ),
        }
    return cache


def direct_exclusive_items(sample: s5.Sample) -> list[str]:
    return [
        item
        for item in sample.sasrec_items
        if item in sample.candidates
        and sample.candidates[item].source_sasrec
        and not sample.candidates[item].source_cf
    ]


def boundary_items(sample: s5.Sample, protocol: dict[str, Any]) -> list[tuple[int, str]]:
    out = []
    for pos in protocol["boundary_positions_0_based"]:
        if pos < len(sample.cf_items) and sample.cf_items[pos] in sample.candidates:
            out.append((int(pos), sample.cf_items[pos]))
    return out


def build_training_pairs(
    samples: list[s5.Sample],
    split_cache: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pos_queries = set()
    neg_queries = set()
    positive_pairs = 0
    negative_pairs = 0
    for sample in samples:
        cache = split_cache[sample.sample_id]
        side = cache["side_features"]
        direct_only = direct_exclusive_items(sample)
        boundaries = boundary_items(sample, protocol)
        cf_rank = candidate_rank(sample.target_item_id, sample.cf_items)
        direct_rank = candidate_rank(sample.target_item_id, direct_only)
        if direct_rank is not None and not hit(cf_rank, 20):
            pos_queries.add(sample.sample_id)
            direct_item = sample.target_item_id
            for pos, boundary in boundaries:
                rows.append(
                    {
                        "row_index": sample.sample_id,
                        "direct_item_id": direct_item,
                        "boundary_item_id": boundary,
                        "boundary_position": pos,
                        "features": pair_features(side[direct_item], side[boundary], pos),
                        "label": 1,
                        "example_type": "positive_promotion",
                    }
                )
                positive_pairs += 1
        if hit(cf_rank, 20):
            neg_queries.add(sample.sample_id)
            protected = sample.target_item_id
            candidates = [item for item in direct_only if item != sample.target_item_id][:5]
            for direct_item in candidates:
                rows.append(
                    {
                        "row_index": sample.sample_id,
                        "direct_item_id": direct_item,
                        "boundary_item_id": protected,
                        "boundary_position": int(cf_rank if cf_rank is not None else 19),
                        "features": pair_features(side[direct_item], side[protected], int(cf_rank if cf_rank is not None else 19)),
                        "label": 0,
                        "example_type": "negative_protection",
                    }
                )
                negative_pairs += 1
    summary = {
        "positive_promotion_queries": len(pos_queries),
        "negative_protection_queries": len(neg_queries),
        "pair_count": len(rows),
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "query_count": len({row["row_index"] for row in rows}),
        "class_weighting": "inverse_class_frequency_in_gradient",
        "seed": 20260717,
    }
    return rows, summary


def train_binary_logistic(rows: list[dict[str, Any]], mean: Any, std: Any, l2: float = 0.001) -> tuple[Any, dict[str, Any]]:
    np = np_module()
    x = normalize_features(rows, mean, std)
    y = np.asarray([float(row["label"]) for row in rows], dtype="float64")
    pos_count = max(float((y == 1.0).sum()), 1.0)
    neg_count = max(float((y == 0.0).sum()), 1.0)
    sample_weight = np.where(y == 1.0, len(y) / (2.0 * pos_count), len(y) / (2.0 * neg_count))
    weights = np.zeros(x.shape[1], dtype="float64")
    bias = 0.0
    lr = 0.05
    epochs = 120
    start = time.perf_counter()
    for _ in range(epochs):
        logits = x.dot(weights) + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        err = (probs - y) * sample_weight
        weights -= lr * ((x.T.dot(err) / len(y)) + l2 * weights)
        bias -= lr * float(err.mean())
    elapsed = time.perf_counter() - start
    logits = x.dot(weights) + bias
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    loss = -float(np.mean(sample_weight * (y * np.log(probs + 1e-12) + (1.0 - y) * np.log(1.0 - probs + 1e-12))))
    return (weights, bias), {
        "epochs": epochs,
        "learning_rate": lr,
        "l2": l2,
        "training_wall_seconds": elapsed,
        "loss": loss,
        "positive_pairs": int((y == 1.0).sum()),
        "negative_pairs": int((y == 0.0).sum()),
    }


def score_features(features: list[float], weights: Any, bias: float, mean: Any, std: Any) -> tuple[float, float]:
    np = np_module()
    x = (np.asarray(features, dtype="float64") - mean) / std
    logit = float(x.dot(weights) + bias)
    prob = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit))))
    return prob, logit


def calibrate_thresholds(
    rows: list[dict[str, Any]],
    weights: Any,
    bias: float,
    mean: Any,
    std: Any,
    protocol: dict[str, Any],
) -> list[float]:
    np = np_module()
    positive_probs = [
        score_features(row["features"], weights, bias, mean, std)[0]
        for row in rows
        if int(row["label"]) == 1
    ]
    if not positive_probs:
        return [0.5]
    values = [float(np.quantile(np.asarray(positive_probs, dtype="float64"), q)) for q in protocol["threshold_quantiles"]]
    values = [max(0.0, min(1.0, value)) for value in values]
    deduped: list[float] = []
    for value in sorted(values):
        rounded = round(value, 6)
        if rounded not in deduped:
            deduped.append(rounded)
    return deduped[: int(protocol["threshold_count"])]


def policy_grid(protocol: dict[str, Any], thresholds: list[float]) -> list[dict[str, Any]]:
    configs = []
    for max_promotions in protocol["max_promotions"]:
        for threshold in thresholds:
            configs.append(
                {
                    "config_id": f"mp{int(max_promotions)}_thr{threshold:.6f}",
                    "max_promotions": int(max_promotions),
                    "promotion_threshold": float(threshold),
                    "margin_threshold": 0.0,
                }
            )
    if len(configs) > 12:
        raise ValueError("S6-6R policy grid exceeded 12 configurations")
    return configs


def promotion_scores_for_sample(
    sample: s5.Sample,
    cache: dict[str, Any],
    protocol: dict[str, Any],
    weights: Any,
    bias: float,
    mean: Any,
    std: Any,
) -> list[dict[str, Any]]:
    side = cache["side_features"]
    tail = boundary_items(sample, protocol)
    if not tail:
        return []
    boundary_pos, boundary_item = tail[-1]
    scored = []
    for direct_item in direct_exclusive_items(sample):
        prob, logit = score_features(pair_features(side[direct_item], side[boundary_item], boundary_pos), weights, bias, mean, std)
        direct_rank = sample.candidates[direct_item].sasrec_rank
        scored.append(
            {
                "item_id": direct_item,
                "promotion_probability": prob,
                "promotion_margin": logit,
                "direct_rank": 999999 if direct_rank is None else int(direct_rank),
                "boundary_item_id": boundary_item,
                "boundary_position": boundary_pos,
            }
        )
    scored.sort(key=lambda row: (-row["promotion_probability"], -row["promotion_margin"], row["direct_rank"], s5.item_sort_key(row["item_id"])))
    return scored


def apply_promotion_policy(
    samples: list[s5.Sample],
    split_cache: dict[str, Any],
    protocol: dict[str, Any],
    weights: Any,
    bias: float,
    mean: Any,
    std: Any,
    config: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any], list[dict[str, Any]]]:
    orders: dict[str, list[str]] = {}
    audit_rows: list[dict[str, Any]] = []
    promotions_per_query = []
    true_promotions = 0
    false_promotions = 0
    invalid = 0
    displacements = []
    zero = one = two = three = 0
    max_promotions = int(config["max_promotions"])
    threshold = float(config["promotion_threshold"])
    for sample in samples:
        cf_top = [item for item in sample.cf_items[:20] if item in sample.candidates]
        if len(cf_top) < 20:
            cf_top = cf_top + [item for item in sample.cf_items[20:] if item in sample.candidates][: 20 - len(cf_top)]
        selected_direct = []
        for scored in promotion_scores_for_sample(sample, split_cache[sample.sample_id], protocol, weights, bias, mean, std):
            if scored["promotion_probability"] < threshold or scored["promotion_margin"] < float(config["margin_threshold"]):
                continue
            if scored["item_id"] in cf_top or scored["item_id"] in selected_direct:
                invalid += 1
                continue
            selected_direct.append(scored)
            if len(selected_direct) >= max_promotions:
                break
        n_promotions = min(len(selected_direct), len(cf_top))
        promotions_per_query.append(n_promotions)
        if n_promotions == 0:
            zero += 1
        elif n_promotions == 1:
            one += 1
        elif n_promotions == 2:
            two += 1
        else:
            three += 1
        retained_cf = cf_top[: max(0, 20 - n_promotions)]
        promoted_items = [row["item_id"] for row in selected_direct[:n_promotions]]
        top20 = retained_cf + promoted_items
        for item in promoted_items:
            if item == sample.target_item_id:
                true_promotions += 1
            else:
                false_promotions += 1
        for idx, item in enumerate(cf_top):
            if item in top20:
                displacements.append(abs(top20.index(item) - idx))
            else:
                displacements.append(20 - idx)
        remainder = [
            item
            for item in sample.cf_items + sample.sasrec_items + list(sample.candidates)
            if item not in set(top20)
        ]
        seen = set()
        full = []
        for item in top20 + remainder:
            if item in sample.candidates and item not in seen:
                seen.add(item)
                full.append(item)
        orders[sample.sample_id] = full
        audit_rows.append(
            {
                "row_index": sample.sample_id,
                "target_item_id": sample.target_item_id,
                "promoted_item_ids": promoted_items,
                "num_promotions": n_promotions,
                "ranked_top20": full[:20],
                "target_rank_0_based": candidate_rank(sample.target_item_id, full),
            }
        )
    sorted_disp = sorted(displacements)
    audit = {
        "invalid_candidate_count": invalid,
        "average_promotions_per_query": sum(promotions_per_query) / len(promotions_per_query) if promotions_per_query else 0.0,
        "queries_with_zero_promotions": zero,
        "queries_with_one_promotion": one,
        "queries_with_two_promotions": two,
        "queries_with_three_promotions": three,
        "true_promotions": true_promotions,
        "false_promotions": false_promotions,
        "average_cf_displacement": sum(displacements) / len(displacements) if displacements else 0.0,
        "p95_cf_displacement": sorted_disp[min(len(sorted_disp) - 1, int(math.ceil(0.95 * len(sorted_disp)) - 1))] if sorted_disp else 0,
    }
    return orders, audit, audit_rows


def metric_definition_reconciliation(samples_by_split: dict[str, list[s5.Sample]], frozen_report: dict[str, Any] | None = None) -> dict[str, Any]:
    per_split = {}
    total_pool_exclusive = 0
    total_non_cf_top20_direct = 0
    total_non_cf_top20_union = 0
    total_direct_source_exclusive_targets = 0
    for split, samples in samples_by_split.items():
        pool_exclusive = 0
        non_cf_top20_direct = 0
        non_cf_top20_union = 0
        direct_source_exclusive_targets = 0
        for sample in samples:
            target = sample.target_item_id
            cf_pool = set(sample.cf_items)
            direct_pool = set(sample.sasrec_items)
            union_pool = set(sample.candidates)
            cf_rank = candidate_rank(target, sample.cf_items)
            direct_rank = candidate_rank(target, sample.sasrec_items)
            if target in direct_pool and target not in cf_pool:
                pool_exclusive += 1
            if not hit(cf_rank, 20) and hit(direct_rank, 20):
                non_cf_top20_direct += 1
            if not hit(cf_rank, 20) and target in union_pool:
                non_cf_top20_union += 1
            if target in union_pool:
                cand = sample.candidates[target]
                if cand.source_sasrec and not cand.source_cf:
                    direct_source_exclusive_targets += 1
        per_split[split] = {
            "cf_candidate_pool_exclusive_target": pool_exclusive,
            "non_cf_top20_target_available_in_direct_top20": non_cf_top20_direct,
            "non_cf_top20_target_available_in_union": non_cf_top20_union,
            "direct_source_exclusive_target": direct_source_exclusive_targets,
        }
        total_pool_exclusive += pool_exclusive
        total_non_cf_top20_direct += non_cf_top20_direct
        total_non_cf_top20_union += non_cf_top20_union
        total_direct_source_exclusive_targets += direct_source_exclusive_targets
    frozen_aggregate = None
    aggregate_reconciles = False
    if frozen_report is not None:
        cf_pool = int(frozen_report["aggregate"]["cf_only"]["target_in_pool_count"])
        union_pool = int(frozen_report["aggregate"]["cf_only"]["union_target_in_pool_count"])
        direct_recoverable = int(frozen_report["aggregate"]["cf_only"]["direct_only_targets_available_at20"])
        frozen_aggregate = {
            "cf_candidate_pool_exclusive_target": union_pool - cf_pool,
            "non_cf_top20_target_available_in_direct_top20": direct_recoverable,
            "source": "S6-5 frozen aggregate metrics; avoids opening valid_gate raw rows before fit/select selection",
        }
        aggregate_reconciles = (
            frozen_aggregate["cf_candidate_pool_exclusive_target"] == 236
            and frozen_aggregate["non_cf_top20_target_available_in_direct_top20"] == 268
        )
    reconciles = aggregate_reconciles if frozen_report is not None else (total_pool_exclusive == 236 and total_non_cf_top20_direct == 268)
    return {
        "definitions": {
            "cf_candidate_pool_exclusive_target": "target exists in direct pool but not in CF candidate pool",
            "non_cf_top20_target_available_in_direct_top20": "target is not hit by CF top20 but is present in direct top20",
            "non_cf_top20_target_available_in_union": "target is not hit by CF top20 but is present in union",
            "direct_source_exclusive_candidate": "candidate has from_sasrec_direct=true and from_cf=false",
            "promotion_recoverable_target": "non-CF-top20 target that can enter top20 by promoting a direct-source-exclusive candidate",
            "cf_hit_preservation": "target hit by CF top20 remains hit after promotion",
        },
        "per_split": per_split,
        "totals": {
            "cf_candidate_pool_exclusive_target": total_pool_exclusive,
            "non_cf_top20_target_available_in_direct_top20": total_non_cf_top20_direct,
            "non_cf_top20_target_available_in_union": total_non_cf_top20_union,
            "direct_source_exclusive_target": total_direct_source_exclusive_targets,
        },
        "expected_s6_4_value": 236,
        "expected_s6_5_s6_6_denominator": 268,
        "fit_select_raw_totals": {
            "cf_candidate_pool_exclusive_target": total_pool_exclusive,
            "non_cf_top20_target_available_in_direct_top20": total_non_cf_top20_direct,
        },
        "frozen_full_validation_aggregate": frozen_aggregate,
        "reconciles": reconciles,
    }


def metric_dict(
    samples: list[s5.Sample],
    orders: dict[str, list[str]],
    audit: dict[str, Any] | None,
    baseline_frozen: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = s6fr.metrics_for_orders(samples, orders)
    cf_hits = metrics["cf_hits_at20"]
    metrics["cf_preservation_rate_at20"] = metrics["cf_hits_preserved_at20"] / cf_hits if cf_hits else 1.0
    if audit:
        metrics.update(audit)
    else:
        metrics.update(
            {
                "invalid_candidate_count": 0,
                "average_promotions_per_query": 0.0,
                "queries_with_zero_promotions": len(samples),
                "queries_with_one_promotion": 0,
                "queries_with_two_promotions": 0,
                "queries_with_three_promotions": 0,
                "true_promotions": 0,
                "false_promotions": 0,
                "average_cf_displacement": 0.0,
                "p95_cf_displacement": 0,
            }
        )
    metrics["pool_to_top20_conversion"] = (
        metrics["hits@20"] / metrics["union_target_in_pool_count"]
        if metrics["union_target_in_pool_count"]
        else 0.0
    )
    if baseline_frozen is not None:
        cf_base = baseline_frozen["cf_only"]
        frozen = baseline_frozen["projected_ranker"]
        hr_denom = frozen["hr@20"] - cf_base["hr@20"]
        ndcg_denom = frozen["ndcg@20"] - cf_base["ndcg@20"]
        metrics["hr20_uplift_retention_vs_frozen"] = None if hr_denom <= 0 else (metrics["hr@20"] - cf_base["hr@20"]) / hr_denom
        metrics["ndcg20_uplift_retention_vs_frozen"] = None if ndcg_denom <= 0 else (metrics["ndcg@20"] - cf_base["ndcg@20"]) / ndcg_denom
        denom = frozen["direct_only_targets_recovered_at20"]
        metrics["direct_recovery_retention_vs_frozen"] = metrics["direct_only_targets_recovered_at20"] / denom if denom else 1.0
    return metrics


def passes_fit_select(metrics: dict[str, Any]) -> bool:
    for split in FIT_SELECT_SPLITS:
        item = metrics[split]
        if item["cf_preservation_rate_at20"] < 0.97:
            return False
        if item["invalid_candidate_count"] != 0:
            return False
        if item["hr20_uplift_retention_vs_frozen"] is None or item["hr20_uplift_retention_vs_frozen"] < 0.95:
            return False
        if item["ndcg20_uplift_retention_vs_frozen"] is None or item["ndcg20_uplift_retention_vs_frozen"] < 0.95:
            return False
        if item["direct_recovery_retention_vs_frozen"] < 0.90:
            return False
    return True


def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    min_uplift = min(
        metrics["valid_fit"]["hr20_uplift_retention_vs_frozen"],
        metrics["valid_fit"]["ndcg20_uplift_retention_vs_frozen"],
        metrics["valid_select"]["hr20_uplift_retention_vs_frozen"],
        metrics["valid_select"]["ndcg20_uplift_retention_vs_frozen"],
    )
    min_recovery = min(
        metrics["valid_fit"]["direct_recovery_retention_vs_frozen"],
        metrics["valid_select"]["direct_recovery_retention_vs_frozen"],
    )
    avg_promotions = (
        metrics["valid_fit"]["average_promotions_per_query"]
        + metrics["valid_select"]["average_promotions_per_query"]
    ) / 2.0
    cfg = row["config"]
    return (min_uplift, min_recovery, -avg_promotions, -cfg["max_promotions"], -cfg["promotion_threshold"], cfg["config_id"])


def aggregate_metrics(per_split: dict[str, dict[str, Any]], splits: tuple[str, ...]) -> dict[str, Any]:
    total = sum(per_split[split]["num_samples"] for split in splits)
    out: dict[str, Any] = {"num_samples": total}
    count_keys = [
        "hits@1",
        "hits@5",
        "hits@10",
        "hits@20",
        "target_in_pool_count",
        "union_target_in_pool_count",
        "cf_hits_at20",
        "cf_hits_preserved_at20",
        "cf_hits_lost_at20",
        "direct_only_targets_available_at20",
        "direct_only_targets_recovered_at20",
        "true_promotions",
        "false_promotions",
        "invalid_candidate_count",
        "queries_with_zero_promotions",
        "queries_with_one_promotion",
        "queries_with_two_promotions",
        "queries_with_three_promotions",
    ]
    for key in count_keys:
        out[key] = sum(per_split[split].get(key, 0) for split in splits)
    for k in KS:
        out[f"hr@{k}"] = out[f"hits@{k}"] / total if total else 0.0
        out[f"ndcg@{k}"] = sum(per_split[split][f"ndcg@{k}"] * per_split[split]["num_samples"] for split in splits) / total if total else 0.0
    out["mrr"] = sum(per_split[split]["mrr"] * per_split[split]["num_samples"] for split in splits) / total if total else 0.0
    out["cf_preservation_rate_at20"] = out["cf_hits_preserved_at20"] / out["cf_hits_at20"] if out["cf_hits_at20"] else 1.0
    out["direct_only_recovery_rate_at20"] = (
        out["direct_only_targets_recovered_at20"] / out["direct_only_targets_available_at20"]
        if out["direct_only_targets_available_at20"]
        else 0.0
    )
    out["average_promotions_per_query"] = sum(
        per_split[split]["average_promotions_per_query"] * per_split[split]["num_samples"]
        for split in splits
    ) / total if total else 0.0
    return out


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "num_samples",
        "hr@1",
        "hr@5",
        "hr@10",
        "hr@20",
        "ndcg@1",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
        "mrr",
        "hits@20",
        "cf_hits_at20",
        "cf_hits_preserved_at20",
        "cf_hits_lost_at20",
        "cf_preservation_rate_at20",
        "direct_only_targets_available_at20",
        "direct_only_targets_recovered_at20",
        "direct_only_recovery_rate_at20",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in [args.protocol, args.output_root, args.report]:
        reject_test_path(path)
    protocol = load_protocol(args.protocol)
    start_total = time.perf_counter()
    feature_start = time.perf_counter()
    config, model, matrix, row_index, popularity, max_log_pop, base_config = load_context()
    frozen_report = read_json(args.frozen_ranker_report)
    residual_report = read_json(args.lightweight_report)
    samples = {split: split_samples(args.output_root, split) for split in FIT_SELECT_SPLITS}
    reconciliation = metric_definition_reconciliation(samples, frozen_report)
    if not reconciliation["reconciles"]:
        raise ValueError("S6-6R metric-definition reconciliation failed")
    split_cache = {
        split: build_split_cache(
            args.output_root,
            split,
            samples[split],
            base_config,
            model,
            matrix,
            row_index,
            popularity,
            max_log_pop,
        )
        for split in FIT_SELECT_SPLITS
    }
    feature_seconds = time.perf_counter() - feature_start
    training_rows, pair_summary = build_training_pairs(samples["valid_fit"], split_cache["valid_fit"], protocol)
    names = feature_names()
    if any(len(row["features"]) != len(names) for row in training_rows):
        raise ValueError("promotion feature dimension mismatch")
    mean, std = fit_normalizer(training_rows)
    train_start = time.perf_counter()
    (weights, bias), train_summary = train_binary_logistic(training_rows, mean, std)
    training_seconds = time.perf_counter() - train_start
    thresholds = calibrate_thresholds(training_rows, weights, bias, mean, std, protocol)
    configs = policy_grid(protocol, thresholds)

    grid_path = output_dir(args.output_root) / "predeclared_policy_grid.json"
    train_manifest_path = output_dir(args.output_root) / "train_manifest.json"
    write_json(grid_path, {"schema": "s6_direct_promotion_policy_grid.v1", "configs": configs, "count": len(configs), "threshold_source": "valid_fit_only"})
    input_manifest = {
        "schema": "s6_direct_promotion_input_manifest.v1",
        "run_id": args.run_id,
        "input_hashes": {
            "protocol": file_sha256(args.protocol),
            "frozen_ranker_report": file_sha256(args.frozen_ranker_report),
            "lightweight_negative_evidence_report": file_sha256(args.lightweight_report),
            "union_report": file_sha256(args.union_report),
            "validation_split_manifest": file_sha256(args.output_root / "s6_validation_split_manifest.json"),
            "frozen_ranker_model": file_sha256(ROOT / config["ranker"]["model_path"]),
        },
        "split_hashes": {
            split: {
                "cf_split_candidates": file_sha256(s6fr.cf_split_path(args.output_root, split)),
                "union_candidates": file_sha256(s6fr.union_path(args.output_root, split)),
                "feature_input_hash": canonical_hash([split_cache[split][sample.sample_id]["input_hash"] for sample in samples[split]]),
            }
            for split in FIT_SELECT_SPLITS
        },
        "feature_names": names,
        "feature_count": len(names),
        "normalization": {"source": "valid_fit", "mean": [float(v) for v in mean], "std": [float(v) for v in std]},
        "training_pairs": pair_summary,
        "training_summary": train_summary | {"training_wall_seconds": training_seconds},
        "thresholds": thresholds,
        "test_read": False,
    }
    write_json(train_manifest_path, input_manifest)

    eval_rows = []
    inference_seconds = 0.0
    for cfg in configs:
        metrics = {}
        hashes = {}
        for split in FIT_SELECT_SPLITS:
            infer_start = time.perf_counter()
            orders, audit, audit_rows = apply_promotion_policy(
                samples[split],
                split_cache[split],
                protocol,
                weights,
                bias,
                mean,
                std,
                cfg,
            )
            inference_seconds += time.perf_counter() - infer_start
            baseline = frozen_report["per_split"][split]["baselines"] | {"projected_ranker": frozen_report["per_split"][split]["projected_ranker"]}
            metrics[split] = metric_dict(samples[split], orders, audit, baseline)
            hashes[split] = canonical_hash({"orders": orders, "audit": audit_rows})
        eval_rows.append({"config": cfg, "metrics": metrics, "hashes": hashes, "passes": passes_fit_select(metrics)})
    passing = [row for row in eval_rows if row["passes"]]
    selected = max(passing, key=selection_key) if passing else None
    selected_report = None
    gate_opened = False
    verdict = "REVISE_STAGE_CLOSEOUT"
    if selected is not None:
        selected_cfg = selected["config"]
        cfg_hash = canonical_hash(selected_cfg)
        model_artifact = {
            "schema": "s6_direct_promotion_model.v1",
            "config": selected_cfg,
            "config_hash": cfg_hash,
            "feature_names": names,
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "weights": [float(v) for v in weights],
            "bias": float(bias),
            "training_summary": train_summary | pair_summary,
        }
        model_path = output_dir(args.output_root) / "model.json"
        write_json(model_path, model_artifact)
        samples["valid_gate"] = split_samples(args.output_root, "valid_gate")
        split_cache["valid_gate"] = build_split_cache(
            args.output_root,
            "valid_gate",
            samples["valid_gate"],
            base_config,
            model,
            matrix,
            row_index,
            popularity,
            max_log_pop,
        )
        gate_opened = True
        per_split = {}
        ranked_hashes = {}
        for split in SPLITS:
            orders, audit, audit_rows = apply_promotion_policy(
                samples[split],
                split_cache[split],
                protocol,
                weights,
                bias,
                mean,
                std,
                selected_cfg,
            )
            baseline = frozen_report["per_split"][split]["baselines"] | {"projected_ranker": frozen_report["per_split"][split]["projected_ranker"]}
            metrics = metric_dict(samples[split], orders, audit, baseline)
            per_split[split] = metrics
            ranked_rows = [
                {
                    "row_index": sample.sample_id,
                    "target_item_id": sample.target_item_id,
                    "history_item_id": sample.history_item_id,
                    "ranked_candidate_item_ids": orders[sample.sample_id],
                    "target_rank_0_based": candidate_rank(sample.target_item_id, orders[sample.sample_id]),
                }
                for sample in samples[split]
            ]
            ranked_path = split_output_dir(args.output_root, split) / "ranked_candidates.jsonl"
            metrics_path = split_output_dir(args.output_root, split) / "metrics.json"
            audit_path = split_output_dir(args.output_root, split) / "promotion_audit.jsonl"
            write_jsonl(ranked_path, ranked_rows)
            write_jsonl(audit_path, audit_rows)
            write_json(metrics_path, {"schema": "s6_direct_promotion_split_metrics.v1", "split": split, "metrics": metrics, "ranked_sha256": file_sha256(ranked_path), "audit_sha256": file_sha256(audit_path), "test_read": False})
            ranked_hashes[split] = file_sha256(ranked_path)
        gate_metrics = per_split["valid_gate"]
        gate_pass = (
            gate_metrics["cf_preservation_rate_at20"] >= 0.97
            and gate_metrics["hr@20"] >= frozen_report["per_split"]["valid_gate"]["baselines"]["cf_only"]["hr@20"]
            and gate_metrics["ndcg@20"] >= frozen_report["per_split"]["valid_gate"]["baselines"]["cf_only"]["ndcg@20"]
            and gate_metrics["direct_recovery_retention_vs_frozen"] >= 0.75
            and gate_metrics["invalid_candidate_count"] == 0
        )
        verdict = "GO_COST_EVIDENCE_CLOSEOUT" if gate_pass else "REVISE_STAGE_CLOSEOUT"
        selected_report = {
            "config": selected_cfg,
            "config_hash": cfg_hash,
            "model_path": model_path.as_posix(),
            "model_sha256": file_sha256(model_path),
            "per_split": per_split,
            "aggregate": aggregate_metrics(per_split, SPLITS),
            "ranked_hashes": ranked_hashes,
            "valid_gate_confirmation": {"opened_after_fit_select_pass": True, "pass": gate_pass, "metrics": gate_metrics},
        }

    total_seconds = time.perf_counter() - start_total
    comparison = {}
    for split in FIT_SELECT_SPLITS:
        comparison[split] = {
            "cf_only": compact_metrics(frozen_report["per_split"][split]["baselines"]["cf_only"]),
            "fixed_rrf": compact_metrics(frozen_report["per_split"][split]["baselines"]["fixed_rrf"]),
            "s6_5_frozen_projected": compact_metrics(frozen_report["per_split"][split]["projected_ranker"]),
        }
    report = {
        "schema": "s6_direct_promotion_gate_report.v1",
        "verdict": verdict,
        "run_id": args.run_id,
        "metric_definition_reconciliation": reconciliation,
        "input_manifest": train_manifest_path.as_posix(),
        "feature_contract": {
            "feature_names": names,
            "feature_count": len(names),
            "forbidden_feature_name_parts": FORBIDDEN_FEATURE_NAME_PARTS,
            "labels": "target equality is used only inside valid_fit loss and validation metrics",
            "normalization": "fit on valid_fit only",
        },
        "ranking_semantics": {
            "anchor": "CF-only candidate order",
            "cf_internal_order": "preserved for retained CF candidates",
            "direct_scope": "direct-source-exclusive candidates only",
            "replacement": "qualified direct candidates replace CF top20 tail candidates",
            "output_order": "retained CF candidates in original CF order, then promoted direct candidates by probability/margin/direct-rank/item-id tie break",
            "overlap_candidates": "kept at CF-anchored positions",
        },
        "policy_grid": {"path": grid_path.as_posix(), "count": len(configs), "configs": configs},
        "training": pair_summary | train_summary | {"training_wall_seconds": training_seconds},
        "fit_select_results": eval_rows,
        "selected": selected_report,
        "valid_gate": {"opened": gate_opened, "reason": "opened only after fit/select pass"},
        "comparisons": {
            "s6_5": comparison,
            "s6_6_best_aggressive_residual": "alpha1_l20.001_clip0.5_cfq0",
            "s6_6_best_preservation_residual": "alpha0.1_l20.001_clip0.25_cfq18",
            "s6_6_verdict": residual_report["verdict"],
        },
        "runtime": {
            "total_wall_seconds": total_seconds,
            "feature_generation_wall_seconds": feature_seconds,
            "training_wall_seconds": training_seconds,
            "inference_wall_seconds": inference_seconds,
            "samples_per_second_fit_select": sum(len(samples[split]) for split in FIT_SELECT_SPLITS) / inference_seconds if inference_seconds else None,
            "model_size_bytes": (output_dir(args.output_root) / "model.json").stat().st_size if (output_dir(args.output_root) / "model.json").exists() else 0,
            "peak_cpu_memory": "not_measured",
            "direct_sasrec_model_inference_seconds": 24.967018,
            "direct_sasrec_formal_wall_seconds": 34.233973,
            "direct_sasrec_peak_cuda_allocated_bytes": 13118464,
        },
        "test_read": False,
    }
    write_json(args.report, report)
    reject_existing(output_dir(args.output_root) / "exit_code.txt")
    (output_dir(args.output_root) / "exit_code.txt").write_text("0\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S6-6R CF-anchored direct promotion gate validation.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-ranker-report", type=Path, default=DEFAULT_OUTPUT_ROOT / "s6_5_frozen_ranker_validation_report.json")
    parser.add_argument("--lightweight-report", type=Path, default=DEFAULT_OUTPUT_ROOT / "s6_6_lightweight_ranker_validation_report.json")
    parser.add_argument("--union-report", type=Path, default=DEFAULT_OUTPUT_ROOT / "s6_4_union_validation_report.json")
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "protocol": args.protocol.as_posix(),
                    "report": args.report.as_posix(),
                    "run_id": args.run_id,
                    "test_read": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    report = run(args)
    print(json.dumps({"verdict": report["verdict"], "report": args.report.as_posix()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
