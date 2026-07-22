#!/usr/bin/env python3
"""S6-6 validation-only lightweight CF-preserving residual ranker."""

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


ROOT = Path(__file__).resolve().parents[1]
CATEGORY = "Industrial_and_Scientific"
RUN_ID = "lightweight_v1"
SPLITS = ("valid_fit", "valid_select", "valid_gate")
KS = (1, 5, 10, 20, 50)
DEFAULT_PROTOCOL = ROOT / "configs/s6_cost_aware_aux/lightweight_ranker_protocol.json"
DEFAULT_OUTPUT_ROOT = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific"
DEFAULT_REPORT = DEFAULT_OUTPUT_ROOT / "s6_6_lightweight_ranker_validation_report.json"
FORBIDDEN_FEATURE_NAME_PARTS = ("target", "label", "is_correct", "post_ranking")


def np_module():
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"NumPy is required for S6 lightweight ranker validation: {exc}") from exc
    return np


def file_sha256(path: Path) -> str:
    return s5.file_sha256(path)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reject_existing(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        raise FileExistsError(f"refusing to overwrite existing S6-6 artifact: {path}")


def reject_test_path(path: Path) -> None:
    parts = [part.lower() for part in path.parts]
    if "test" in parts or "final_test" in parts or path.name.lower() == "test.csv":
        raise ValueError(f"S6-6 refuses test path: {path}")


def read_json(path: Path) -> Any:
    return s5.read_json(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return s5.read_jsonl(path)


def write_json(path: Path, data: Any) -> None:
    reject_existing(path)
    s5.write_json(path, data)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    reject_existing(path)
    s5.write_jsonl(path, rows)


def candidate_rank(target: str, items: list[str]) -> int | None:
    return s6fr.candidate_rank(target, items)


def hit(rank: int | None, k: int) -> bool:
    return rank is not None and rank < k


def output_dir(output_root: Path) -> Path:
    return output_root / "lightweight_ranker" / RUN_ID


def split_output_dir(output_root: Path, split: str) -> Path:
    return output_dir(output_root) / split


def load_protocol(path: Path) -> dict[str, Any]:
    reject_test_path(path)
    protocol = read_json(path)
    if protocol["candidate_budget_k"] != 20:
        raise ValueError("S6-6 requires K=20")
    if protocol["train_split"] != "valid_fit" or protocol["selection_split"] != "valid_select":
        raise ValueError("unexpected split protocol")
    return protocol


def feature_names() -> list[str]:
    names = [
        "frozen_ranker_score",
        "frozen_rank",
        "reciprocal_frozen_rank",
        "from_cf",
        "from_sasrec_direct",
        "overlap",
        "cf_rank_filled",
        "sasrec_direct_rank_filled",
        "reciprocal_cf_rank",
        "reciprocal_sasrec_direct_rank",
        "is_cf_top1",
        "is_cf_top5",
        "is_cf_top10",
        "is_cf_top20",
        "sasrec_direct_score",
        "sasrec_direct_zscore",
        "sasrec_direct_top1_margin",
        "sasrec_direct_score_minus_topk_mean",
        "min_source_rank",
        "fusion_rank",
        "reciprocal_min_source_rank",
        "reciprocal_fusion_rank",
        "fusion_score",
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
    bad = [name for name in names if any(part in name for part in FORBIDDEN_FEATURE_NAME_PARTS)]
    if bad:
        raise ValueError(f"forbidden feature names: {bad}")
    return names


def load_context():
    config = s6fr.load_frozen_config(s6fr.DEFAULT_CONFIG)
    model, _contract = s6fr.load_model_and_contract(config)
    matrix, row_index = s5.load_embeddings(
        ROOT / f"data/Amazon/cs_embeddings/{CATEGORY}/{CATEGORY}.cf_emb.npy",
        ROOT / f"data/Amazon/cs_embeddings/{CATEGORY}/{CATEGORY}.row_index.json",
    )
    popularity = s5.train_popularity(ROOT / f"data/Amazon/sid_versions/cf_k512_dedup/{CATEGORY}/train.csv")
    max_log_pop = max((math.log1p(value) for value in popularity.values()), default=1.0)
    base_config = s6fr.source_aware_rrf_config(config)
    return config, model, matrix, row_index, popularity, max_log_pop, base_config


def split_samples(output_root: Path, split: str) -> list[s5.Sample]:
    samples, _ = s6fr.build_samples_for_split(output_root, Path(), split)
    return samples


def frozen_scores_for_samples(samples: list[s5.Sample], base_config: dict[str, Any], model: dict[str, Any], matrix: Any, row_index: dict[str, int], popularity: Any, max_log_pop: float) -> dict[str, dict[str, float]]:
    feature_names_model = model["feature_names"]
    means = model["feature_stats"]["mean"]
    stds = model["feature_stats"]["std"]
    weights = [float(value) for value in model["weights"]]
    excluded = set(s6fr.s5final.SOURCE_SPECIFIC_FEATURES_ZEROED)
    scores: dict[str, dict[str, float]] = {}
    for sample in samples:
        base_items = s5.policy_items(sample, base_config)
        sample_scores: dict[str, float] = {}
        for rank0, item in enumerate(base_items):
            raw = s5.raw_features(
                sample,
                sample.candidates[item],
                rank0 + 1,
                len(base_items),
                matrix,
                row_index,
                popularity,
                max_log_pop,
                "source_independent_projection",
            )
            score = 0.0
            for idx, name in enumerate(feature_names_model):
                z = 0.0 if name in excluded else (float(raw[name]) - float(means[idx])) / float(stds[idx])
                score += weights[idx] * z
            sample_scores[item] = score
        scores[sample.sample_id] = sample_scores
    return scores


def detail_map(sample: s5.Sample, item: str, union_row: dict[str, Any] | None = None) -> dict[str, Any]:
    if union_row is None:
        return {}
    for detail in union_row.get("candidate_details", []):
        if str(detail.get("item_id")) == str(item):
            return detail
    return {}


def load_union_rows(output_root: Path, split: str) -> dict[str, dict[str, Any]]:
    return {str(row["row_index"]): row for row in read_jsonl(s6fr.union_path(output_root, split))}


def make_feature_rows(samples: list[s5.Sample], split: str, output_root: Path, base_config: dict[str, Any], model: dict[str, Any], matrix: Any, row_index: dict[str, int], popularity: Any, max_log_pop: float) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    names = feature_names()
    union_rows = load_union_rows(output_root, split)
    frozen_scores = frozen_scores_for_samples(samples, base_config, model, matrix, row_index, popularity, max_log_pop)
    rows: list[dict[str, Any]] = []
    frozen_orders: dict[str, list[str]] = {}
    for sample in samples:
        base_items = s5.policy_items(sample, base_config)
        ordered = sorted(base_items, key=lambda item: (-frozen_scores[sample.sample_id][item], base_items.index(item), s5.item_sort_key(item)))
        frozen_orders[sample.sample_id] = ordered
        frozen_rank = {item: idx for idx, item in enumerate(ordered)}
        union_row = union_rows[sample.sample_id]
        for fusion_rank0, item in enumerate(base_items):
            cand = sample.candidates[item]
            detail = detail_map(sample, item, union_row)
            direct_detail = detail.get("sasrec_direct_detail") or {}
            cf_rank = cand.cf_rank
            direct_rank = cand.sasrec_rank
            missing_rank = float(len(base_items) + 1)
            raw = s5.raw_features(sample, cand, fusion_rank0 + 1, len(base_items), matrix, row_index, popularity, max_log_pop, "source_independent_projection")
            values = {
                "frozen_ranker_score": frozen_scores[sample.sample_id][item],
                "frozen_rank": float(frozen_rank[item] + 1),
                "reciprocal_frozen_rank": 1.0 / float(frozen_rank[item] + 1),
                "from_cf": float(cand.source_cf),
                "from_sasrec_direct": float(cand.source_sasrec),
                "overlap": float(cand.source_cf and cand.source_sasrec),
                "cf_rank_filled": float(cf_rank + 1) if cf_rank is not None else missing_rank,
                "sasrec_direct_rank_filled": float(direct_rank + 1) if direct_rank is not None else missing_rank,
                "reciprocal_cf_rank": 0.0 if cf_rank is None else 1.0 / float(cf_rank + 1),
                "reciprocal_sasrec_direct_rank": 0.0 if direct_rank is None else 1.0 / float(direct_rank + 1),
                "is_cf_top1": float(cf_rank is not None and cf_rank < 1),
                "is_cf_top5": float(cf_rank is not None and cf_rank < 5),
                "is_cf_top10": float(cf_rank is not None and cf_rank < 10),
                "is_cf_top20": float(cf_rank is not None and cf_rank < 20),
                "sasrec_direct_score": float(direct_detail.get("sasrec_direct_score", 0.0) or 0.0),
                "sasrec_direct_zscore": float(direct_detail.get("sasrec_direct_zscore", 0.0) or 0.0),
                "sasrec_direct_top1_margin": float(direct_detail.get("sasrec_direct_top1_margin", 0.0) or 0.0),
                "sasrec_direct_score_minus_topk_mean": float(direct_detail.get("sasrec_direct_score_minus_topk_mean", 0.0) or 0.0),
            }
            for name in [
                "min_source_rank",
                "fusion_rank",
                "reciprocal_min_source_rank",
                "reciprocal_fusion_rank",
                "fusion_score",
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
            ]:
                values[name] = float(raw[name])
            rows.append(
                {
                    "split": split,
                    "row_index": sample.sample_id,
                    "item_id": item,
                    "label": int(item == sample.target_item_id),
                    "features": [float(values[name]) for name in names],
                    "frozen_ranker_score": frozen_scores[sample.sample_id][item],
                    "frozen_rank_0_based": frozen_rank[item],
                    "base_rank_0_based": fusion_rank0,
                    "from_cf": cand.source_cf,
                    "from_sasrec_direct": cand.source_sasrec,
                }
            )
    return rows, frozen_orders


def fit_normalizer(rows: list[dict[str, Any]]):
    np = np_module()
    x = np.asarray([row["features"] for row in rows], dtype="float64")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std <= 1e-12] = 1.0
    return mean, std


def normalize_rows(rows: list[dict[str, Any]], mean: Any, std: Any):
    np = np_module()
    return np.asarray([(np.asarray(row["features"], dtype="float64") - mean) / std for row in rows], dtype="float64")


def train_pairwise(rows: list[dict[str, Any]], mean: Any, std: Any, l2: float, seed: int = 20260717) -> tuple[Any, dict[str, Any]]:
    np = np_module()
    by_query: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_query.setdefault(row["row_index"], []).append(idx)
    x = normalize_rows(rows, mean, std)
    pairs = []
    for indices in by_query.values():
        positives = [idx for idx in indices if rows[idx]["label"] == 1]
        negatives = [idx for idx in indices if rows[idx]["label"] == 0]
        for pos in positives:
            for neg in negatives:
                pairs.append((pos, neg))
    weights = np.zeros(x.shape[1], dtype="float64")
    lr = 0.05
    epochs = 80
    start = time.perf_counter()
    for _epoch in range(epochs):
        grad = l2 * weights
        for pos, neg in pairs:
            diff = x[pos] - x[neg]
            margin = float(diff.dot(weights))
            coeff = -1.0 / (1.0 + math.exp(min(50.0, margin)))
            grad += coeff * diff / max(1, len(pairs))
        weights -= lr * grad
    elapsed = time.perf_counter() - start
    return weights, {"pairs": len(pairs), "epochs": epochs, "learning_rate": lr, "l2": l2, "training_wall_seconds": elapsed, "seed": seed}


def scores_by_query(rows: list[dict[str, Any]], weights: Any, mean: Any, std: Any, alpha: float, residual_clip: float) -> dict[str, dict[str, float]]:
    np = np_module()
    x = normalize_rows(rows, mean, std)
    out: dict[str, dict[str, float]] = {}
    residuals = x.dot(weights)
    residuals = np.clip(residuals, -float(residual_clip), float(residual_clip))
    for row, residual in zip(rows, residuals):
        score = float(row["frozen_ranker_score"]) + float(alpha) * float(residual)
        out.setdefault(row["row_index"], {})[row["item_id"]] = score
    return out


def apply_cf_quota(scored_items: list[tuple[str, float, int]], sample: s5.Sample, quota: int) -> list[str]:
    base_sorted = [item for item, _score, _rank in scored_items]
    if quota <= 0:
        return base_sorted
    cf_top = [item for item in sample.cf_items[:20] if item in sample.candidates]
    cf_ordered = [item for item in base_sorted if item in set(cf_top)]
    selected_cf = cf_ordered[: min(quota, len(cf_ordered))]
    rest = [item for item in base_sorted if item not in set(selected_cf)]
    return selected_cf + rest


def rank_config(samples: list[s5.Sample], rows: list[dict[str, Any]], weights: Any, mean: Any, std: Any, config: dict[str, Any]) -> dict[str, list[str]]:
    by_row = {row["row_index"]: [] for row in rows}
    score_map = scores_by_query(rows, weights, mean, std, config["alpha"], config["residual_clip"])
    for row in rows:
        by_row[row["row_index"]].append(row)
    sample_map = {sample.sample_id: sample for sample in samples}
    orders: dict[str, list[str]] = {}
    for rid, group in by_row.items():
        scored = sorted(
            [(row["item_id"], score_map[rid][row["item_id"]], row["base_rank_0_based"]) for row in group],
            key=lambda item: (-item[1], item[2], s5.item_sort_key(item[0])),
        )
        orders[rid] = apply_cf_quota(scored, sample_map[rid], int(config["cf_head_quota"]))
    return orders


def metric_dict(samples: list[s5.Sample], orders: dict[str, list[str]], baseline_frozen: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = s6fr.metrics_for_orders(samples, orders)
    cf_top20_displacements = []
    for sample in samples:
        ranked = orders[sample.sample_id]
        for item in sample.cf_items[:20]:
            if item in ranked:
                cf_top20_displacements.append(abs(ranked.index(item) - sample.cf_items.index(item)))
    cf_hits = metrics["cf_hits_at20"]
    metrics["cf_preservation_rate_at20"] = metrics["cf_hits_preserved_at20"] / cf_hits if cf_hits else 1.0
    metrics["avg_cf_top20_displacement"] = sum(cf_top20_displacements) / len(cf_top20_displacements) if cf_top20_displacements else 0.0
    if cf_top20_displacements:
        sorted_disp = sorted(cf_top20_displacements)
        metrics["p95_cf_top20_displacement"] = sorted_disp[min(len(sorted_disp) - 1, int(math.ceil(0.95 * len(sorted_disp)) - 1))]
    else:
        metrics["p95_cf_top20_displacement"] = 0
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


def config_grid(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    grid = protocol["config_grid"]
    configs = []
    for alpha in grid["alpha"]:
        for l2 in grid["l2"]:
            for residual_clip in grid["residual_clip"]:
                for cf_head_quota in grid["cf_head_quota"]:
                    configs.append(
                        {
                            "config_id": f"alpha{alpha:g}_l2{l2:g}_clip{residual_clip:g}_cfq{cf_head_quota}",
                            "alpha": float(alpha),
                            "l2": float(l2),
                            "residual_clip": float(residual_clip),
                            "cf_head_quota": int(cf_head_quota),
                        }
                    )
    return configs


def passes_fit_select(metrics: dict[str, Any]) -> bool:
    return (
        metrics["valid_fit"]["cf_preservation_rate_at20"] >= 0.97
        and metrics["valid_select"]["cf_preservation_rate_at20"] >= 0.97
        and metrics["valid_fit"]["hr20_uplift_retention_vs_frozen"] is not None
        and metrics["valid_select"]["hr20_uplift_retention_vs_frozen"] is not None
        and metrics["valid_fit"]["hr20_uplift_retention_vs_frozen"] >= 0.95
        and metrics["valid_select"]["hr20_uplift_retention_vs_frozen"] >= 0.95
        and metrics["valid_fit"]["ndcg20_uplift_retention_vs_frozen"] >= 0.95
        and metrics["valid_select"]["ndcg20_uplift_retention_vs_frozen"] >= 0.95
        and metrics["valid_fit"]["direct_recovery_retention_vs_frozen"] >= 0.90
        and metrics["valid_select"]["direct_recovery_retention_vs_frozen"] >= 0.90
    )


def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    min_retention = min(
        metrics["valid_fit"]["hr20_uplift_retention_vs_frozen"],
        metrics["valid_select"]["hr20_uplift_retention_vs_frozen"],
        metrics["valid_fit"]["ndcg20_uplift_retention_vs_frozen"],
        metrics["valid_select"]["ndcg20_uplift_retention_vs_frozen"],
    )
    config = row["config"]
    return (min_retention, -config["alpha"], -config["cf_head_quota"], -config["residual_clip"], -config["l2"], config["config_id"])


def aggregate_metrics(per_split: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = sum(per_split[split]["num_samples"] for split in SPLITS)
    out = {"num_samples": total}
    count_keys = [
        "hits@1",
        "hits@5",
        "hits@10",
        "hits@20",
        "hits@50",
        "target_in_pool_count",
        "union_target_in_pool_count",
        "cf_hits_at20",
        "cf_hits_preserved_at20",
        "cf_hits_lost_at20",
        "direct_only_targets_available_at20",
        "direct_only_targets_recovered_at20",
        "overlap_source_target_recovery_at20",
    ]
    for key in count_keys:
        out[key] = sum(per_split[split].get(key, 0) for split in SPLITS)
    for k in KS:
        out[f"hr@{k}"] = out[f"hits@{k}"] / total if total else 0.0
        out[f"ndcg@{k}"] = sum(per_split[split][f"ndcg@{k}"] * per_split[split]["num_samples"] for split in SPLITS) / total if total else 0.0
    out["mrr"] = sum(per_split[split]["mrr"] * per_split[split]["num_samples"] for split in SPLITS) / total if total else 0.0
    out["cf_preservation_rate_at20"] = out["cf_hits_preserved_at20"] / out["cf_hits_at20"] if out["cf_hits_at20"] else 1.0
    out["direct_only_recovery_rate_at20"] = out["direct_only_targets_recovered_at20"] / out["direct_only_targets_available_at20"] if out["direct_only_targets_available_at20"] else 0.0
    out["average_candidate_pool_size"] = sum(per_split[split]["average_candidate_pool_size"] * per_split[split]["num_samples"] for split in SPLITS) / total if total else 0.0
    out["avg_cf_top20_displacement"] = sum(per_split[split]["avg_cf_top20_displacement"] * per_split[split]["num_samples"] for split in SPLITS) / total if total else 0.0
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in [args.protocol, args.output_root, args.report]:
        reject_test_path(path)
    protocol = load_protocol(args.protocol)
    start_all = time.perf_counter()
    config, model, matrix, row_index, popularity, max_log_pop, base_config = load_context()
    model_path = ROOT / config["ranker"]["model_path"]
    frozen_report = read_json(args.frozen_ranker_report)
    fit_select_splits = ("valid_fit", "valid_select")
    samples = {split: split_samples(args.output_root, split) for split in fit_select_splits}
    feature_rows = {}
    frozen_orders = {}
    for split in fit_select_splits:
        feature_rows[split], frozen_orders[split] = make_feature_rows(
            samples[split],
            split,
            args.output_root,
            base_config,
            model,
            matrix,
            row_index,
            popularity,
            max_log_pop,
        )
    names = feature_names()
    mean, std = fit_normalizer(feature_rows["valid_fit"])
    configs = config_grid(protocol)
    predeclared_path = output_dir(args.output_root) / "predeclared_config_grid.json"
    write_json(predeclared_path, {"schema": "s6_lightweight_config_grid.v1", "configs": configs, "count": len(configs)})
    train_manifest = {
        "schema": "s6_lightweight_train_manifest.v1",
        "run_id": args.run_id,
        "train_split": "valid_fit",
        "selection_split": "valid_select",
        "gate_split": "valid_gate",
        "input_hashes": {
            "protocol": file_sha256(args.protocol),
            "frozen_ranker_report": file_sha256(args.frozen_ranker_report),
            "frozen_ranker_model": file_sha256(model_path),
            "s6_4_union_report": file_sha256(args.union_report),
        },
        "feature_names": names,
        "feature_hashes": {split: canonical_hash([{k: row[k] for k in ["row_index", "item_id", "features"]} for row in feature_rows[split]]) for split in fit_select_splits},
        "normalization": {"source": "valid_fit", "mean": [float(v) for v in mean], "std": [float(v) for v in std]},
        "forbidden_features": protocol["forbidden_features"],
        "test_read": False,
    }
    write_json(output_dir(args.output_root) / "train_manifest.json", train_manifest)
    model_by_l2 = {}
    training_summaries = {}
    for l2 in sorted({cfg["l2"] for cfg in configs}):
        weights, summary = train_pairwise(feature_rows["valid_fit"], mean, std, l2)
        model_by_l2[l2] = weights
        training_summaries[str(l2)] = summary
    eval_rows = []
    for cfg in configs:
        weights = model_by_l2[cfg["l2"]]
        metrics = {}
        for split in ("valid_fit", "valid_select"):
            orders = rank_config(samples[split], feature_rows[split], weights, mean, std, cfg)
            metrics[split] = metric_dict(samples[split], orders, frozen_report["per_split"][split]["baselines"] | {"projected_ranker": frozen_report["per_split"][split]["projected_ranker"]})
        eval_rows.append({"config": cfg, "metrics": metrics, "passes": passes_fit_select(metrics)})
    passing = [row for row in eval_rows if row["passes"]]
    selected = max(passing, key=selection_key) if passing else None
    if selected is None:
        verdict = "REVISE_LIGHTWEIGHT_RANKER"
        selected_report = None
        gate_metrics = None
    else:
        selected_cfg = selected["config"]
        cfg_hash = canonical_hash(selected_cfg)
        write_json(output_dir(args.output_root) / "selected_config.json", {"config": selected_cfg, "config_hash": cfg_hash})
        weights = model_by_l2[selected_cfg["l2"]]
        model_artifact = {
            "schema": "s6_lightweight_linear_residual_model.v1",
            "config": selected_cfg,
            "feature_names": names,
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "weights": [float(v) for v in weights],
            "training_summary": training_summaries[str(selected_cfg["l2"])],
        }
        model_path_out = output_dir(args.output_root) / "model.json"
        write_json(model_path_out, model_artifact)
        samples["valid_gate"] = split_samples(args.output_root, "valid_gate")
        feature_rows["valid_gate"], frozen_orders["valid_gate"] = make_feature_rows(
            samples["valid_gate"],
            "valid_gate",
            args.output_root,
            base_config,
            model,
            matrix,
            row_index,
            popularity,
            max_log_pop,
        )
        per_split_metrics = {}
        ranked_hashes = {}
        for split in SPLITS:
            orders = rank_config(samples[split], feature_rows[split], weights, mean, std, selected_cfg)
            metrics = metric_dict(samples[split], orders, frozen_report["per_split"][split]["baselines"] | {"projected_ranker": frozen_report["per_split"][split]["projected_ranker"]})
            per_split_metrics[split] = metrics
            rows = [
                {
                    "row_index": sample.sample_id,
                    "target_item_id": sample.target_item_id,
                    "ranked_candidate_item_ids": orders[sample.sample_id],
                    "target_rank_0_based": candidate_rank(sample.target_item_id, orders[sample.sample_id]),
                }
                for sample in samples[split]
            ]
            ranked_path = split_output_dir(args.output_root, split) / "ranked_candidates.jsonl"
            write_jsonl(ranked_path, rows)
            metrics_path = split_output_dir(args.output_root, split) / "metrics.json"
            write_json(metrics_path, {"schema": "s6_lightweight_split_metrics.v1", "split": split, "metrics": metrics, "ranked_sha256": file_sha256(ranked_path), "test_read": False})
            ranked_hashes[split] = file_sha256(ranked_path)
        aggregate = aggregate_metrics(per_split_metrics)
        gate_metrics = per_split_metrics["valid_gate"]
        gate_pass = (
            gate_metrics["cf_preservation_rate_at20"] >= 0.97
            and gate_metrics["hr@20"] >= frozen_report["per_split"]["valid_gate"]["baselines"]["cf_only"]["hr@20"]
            and gate_metrics["ndcg@20"] >= frozen_report["per_split"]["valid_gate"]["baselines"]["cf_only"]["ndcg@20"]
            and gate_metrics["direct_recovery_retention_vs_frozen"] >= 0.75
        )
        verdict = "GO_COST_EVIDENCE_CLOSEOUT" if gate_pass else "REVISE_LIGHTWEIGHT_RANKER"
        selected_report = {
            "config": selected_cfg,
            "config_hash": cfg_hash,
            "model_path": model_path_out.as_posix(),
            "model_sha256": file_sha256(model_path_out),
            "per_split": per_split_metrics,
            "aggregate": aggregate,
            "ranked_hashes": ranked_hashes,
            "valid_gate_confirmation": {"inspected_after_selection": True, "pass": gate_pass, "metrics": gate_metrics},
        }
    elapsed_all = time.perf_counter() - start_all
    report = {
        "schema": "s6_lightweight_ranker_validation_report.v1",
        "verdict": verdict,
        "run_id": args.run_id,
        "metric_definition_reconciliation": {
            "s6_4_direct_only_targets_beyond_cf_pool": "target absent from CF candidate pool but present in direct K20 pool",
            "s6_5_direct_only_recovered_denominator": "target missed by CF top20 baseline but present in direct top20 ordering",
            "non_cf_top20_recoverable_targets": "target not hit by CF top20 but available somewhere in union pool",
            "direct_source_only_pool_targets": "target in direct K20 and not in CF candidate pool",
        },
        "feature_contract": {
            "feature_names": names,
            "feature_count": len(names),
            "normalization": "mean/std fitted on valid_fit only",
            "missing_value_policy": "missing ranks use candidate_count+1; missing direct scores use 0",
            "forbidden_feature_name_parts": FORBIDDEN_FEATURE_NAME_PARTS,
            "target_leakage_audit": "labels are used only for valid_fit training loss and validation metrics, never as features",
        },
        "predeclared_config_count": len(configs),
        "training_summaries": training_summaries,
        "fit_select_results": eval_rows,
        "selected": selected_report,
        "runtime": {
            "total_wall_seconds": elapsed_all,
            "training_wall_seconds": sum(summary["training_wall_seconds"] for summary in training_summaries.values()),
            "model_size_bytes": (output_dir(args.output_root) / "model.json").stat().st_size if (output_dir(args.output_root) / "model.json").exists() else 0,
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
    parser = argparse.ArgumentParser(description="S6-6 lightweight CF-preserving ranker validation.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-ranker-report", type=Path, default=DEFAULT_OUTPUT_ROOT / "s6_5_frozen_ranker_validation_report.json")
    parser.add_argument("--union-report", type=Path, default=DEFAULT_OUTPUT_ROOT / "s6_4_union_validation_report.json")
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps({"dry_run": True, "protocol": args.protocol.as_posix(), "report": args.report.as_posix(), "run_id": args.run_id, "test_read": False}, indent=2, sort_keys=True))
        return
    report = run(args)
    print(json.dumps({"verdict": report["verdict"], "report": args.report.as_posix()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
