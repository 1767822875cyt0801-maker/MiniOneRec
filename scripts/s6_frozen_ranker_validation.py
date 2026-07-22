#!/usr/bin/env python3
"""Validate the frozen S5 projected ranker on S6 CF + direct-SASRec union."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import s5_auxiliary_fusion_conversion as s5
import s5_final_test_confirmation as s5final
import s6_cf_direct_union_validation as s6union


ROOT = Path(__file__).resolve().parents[1]
CATEGORY = "Industrial_and_Scientific"
SPLITS = ("valid_fit", "valid_select", "valid_gate")
KS = (1, 5, 10, 20, 50)
RUN_ID = "frozen_ranker_v1"
EXPECTED_MODEL_SHA256 = "6a3bca7cc714d295572ce4a3953d60719a1c72757895835554a83d747dca65ce"
FORBIDDEN_DIRECT_FEATURES = {
    "sasrec_direct_score",
    "sasrec_direct_zscore",
    "sasrec_direct_top1_margin",
    "sasrec_direct_score_minus_topk_mean",
}

DEFAULT_OUTPUT_ROOT = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific"
DEFAULT_REPORT = DEFAULT_OUTPUT_ROOT / "s6_5_frozen_ranker_validation_report.json"
DEFAULT_CONFIG = ROOT / "configs/s5_auxiliary_fusion/frozen_release_config.json"


def file_sha256(path: Path) -> str:
    return s5.file_sha256(path)


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


def reject_existing(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        raise FileExistsError(f"refusing to overwrite existing S6-5 artifact: {path}")


def reject_test_path(path: Path) -> None:
    parts = [part.lower() for part in path.parts]
    if "test" in parts or "final_test" in parts or path.name.lower() == "test.csv":
        raise ValueError(f"S6-5 refuses test path: {path}")


def candidate_rank(target: str, items: list[str]) -> int | None:
    for idx, item in enumerate(items):
        if str(item) == str(target):
            return idx
    return None


def hit(rank: int | None, k: int) -> bool:
    return rank is not None and rank < k


def rank_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    ranks = [candidate_rank(row["target_item_id"], row[key]) for row in rows]
    out: dict[str, Any] = {"num_samples": len(rows)}
    for k in KS:
        hits = 0
        ndcg = 0.0
        for rank in ranks:
            if hit(rank, k):
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
        out[f"hits@{k}"] = hits
        out[f"hr@{k}"] = hits / len(rows) if rows else 0.0
        out[f"ndcg@{k}"] = ndcg / len(rows) if rows else 0.0
    out["mrr"] = sum(0.0 if rank is None else 1.0 / (rank + 1) for rank in ranks) / len(rows) if rows else 0.0
    out["target_in_pool_count"] = sum(rank is not None for rank in ranks)
    out["target_in_pool_rate"] = out["target_in_pool_count"] / len(rows) if rows else 0.0
    return out


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_config(path: Path) -> dict[str, Any]:
    reject_test_path(path)
    config = read_json(path)
    if config["ranker_compatibility_mode"] != "source_independent_projection":
        raise ValueError("frozen config is not source_independent_projection")
    if config["selected_candidate_policy"] != "source_aware_rrf_lam0.75_bonus0.01":
        raise ValueError("unexpected frozen candidate policy")
    if float(config["lambda_sasrec"]) != 0.75 or float(config["source_bonus"]) != 0.01:
        raise ValueError("unexpected frozen RRF parameters")
    model_path = ROOT / config["ranker"]["model_path"]
    if file_sha256(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen ranker model SHA256 mismatch")
    if config["ranker"]["model_sha256"] != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen config model SHA256 mismatch")
    return config


def load_model_and_contract(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_path = ROOT / config["ranker"]["model_path"]
    model = s5.load_model(model_path)
    contract = s5final.feature_contract(model)
    if model["feature_names"] != config["projection_contract"]["ordered_feature_names"]:
        raise ValueError("model feature order does not match frozen config")
    if model["feature_names"] != contract["ordered_feature_names"]:
        raise ValueError("model feature order does not match feature contract")
    if len(model["weights"]) != len(model["feature_names"]):
        raise ValueError("model weight dimension mismatch")
    return model, contract


def cf_split_path(output_root: Path, split: str) -> Path:
    return output_root / "cf_split_views/union_v1" / split / "candidates.jsonl"


def direct_path_from_bundle_root(bundle_root: Path, split: str) -> Path:
    return bundle_root / f"results/s6_cost_aware_aux/{CATEGORY}/direct_sasrec/{split}/formal_v1/k20/candidates.jsonl"


def union_path(output_root: Path, split: str) -> Path:
    return output_root / "union/union_v1" / split / "k20/candidates.jsonl"


def output_paths(output_root: Path, split: str) -> dict[str, Path]:
    base = output_root / "frozen_ranker" / RUN_ID / split
    return {
        "ranked": base / "ranked_candidates.jsonl",
        "metrics": base / "metrics.json",
    }


def direct_rows_from_union(union_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in union_rows:
        details = [item for item in row.get("candidate_details", []) if item.get("from_sasrec_direct")]
        details.sort(key=lambda item: int(item["sasrec_direct_rank_0_based"]))
        candidate_items = [str(item["item_id"]) for item in details]
        direct_details = []
        for detail in details:
            copied = dict(detail.get("sasrec_direct_detail") or {})
            copied.setdefault("item_id", str(detail["item_id"]))
            copied.setdefault("source_type", "exact")
            copied.setdefault("sid_rank_0_based", detail["sasrec_direct_rank_0_based"])
            direct_details.append(copied)
        rows.append(
            {
                "row_index": row["row_index"],
                "target_item_id": row["target_item_id"],
                "history_item_id": row["history_item_id"],
                "candidate_item_ids": candidate_items,
                "candidate_details": direct_details,
            }
        )
    return rows


def split_map_for(rows: list[dict[str, Any]], split: str) -> dict[str, str]:
    return {str(row["row_index"]): split for row in rows}


def source_aware_rrf_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_id": config["selected_candidate_policy"],
        "policy": "source_aware_rrf",
        "lambda_sasrec": config["lambda_sasrec"],
        "source_bonus": config["source_bonus"],
    }


def ranked_with_feature_hash(
    samples: list[s5.Sample],
    base_config: dict[str, Any],
    model: dict[str, Any],
    matrix: Any,
    row_index: dict[str, int],
    popularity: Any,
    max_log_pop: float,
) -> tuple[dict[str, list[str]], str, list[dict[str, Any]]]:
    feature_names = model["feature_names"]
    means = model["feature_stats"]["mean"]
    stds = model["feature_stats"]["std"]
    weights = [float(value) for value in model["weights"]]
    excluded = set(s5final.SOURCE_SPECIFIC_FEATURES_ZEROED)
    feature_rows = []
    orders: dict[str, list[str]] = {}
    ranked_rows: list[dict[str, Any]] = []
    for sample in samples:
        base_items = s5.policy_items(sample, base_config)
        scored = []
        for rank0, item in enumerate(base_items):
            candidate = sample.candidates[item]
            raw = s5.raw_features(
                sample,
                candidate,
                rank0 + 1,
                len(base_items),
                matrix,
                row_index,
                popularity,
                max_log_pop,
                "source_independent_projection",
            )
            if FORBIDDEN_DIRECT_FEATURES & set(raw):
                raise ValueError("forbidden direct score feature entered frozen ranker projection")
            projected = []
            score = 0.0
            for idx, name in enumerate(feature_names):
                z = 0.0 if name in excluded else (float(raw[name]) - float(means[idx])) / float(stds[idx])
                projected.append(z)
                score += weights[idx] * z
            feature_rows.append(
                {
                    "row_index": sample.sample_id,
                    "item_id": item,
                    "base_rank_0_based": rank0,
                    "projected_features": [round(value, 12) for value in projected],
                }
            )
            scored.append((score, rank0, s5.item_sort_key(item), item))
        scored.sort(key=lambda row: (-row[0], row[1], row[2]))
        ranked = [row[-1] for row in scored]
        orders[sample.sample_id] = ranked
        ranked_rows.append(
            {
                "row_index": sample.sample_id,
                "target_item_id": sample.target_item_id,
                "history_item_id": sample.history_item_id,
                "ranked_candidate_item_ids": ranked,
                "target_rank_0_based": candidate_rank(sample.target_item_id, ranked),
            }
        )
    return orders, canonical_hash(feature_rows), ranked_rows


def metrics_for_orders(samples: list[s5.Sample], orders: dict[str, list[str]]) -> dict[str, Any]:
    rows = [
        {
            "row_index": sample.sample_id,
            "target_item_id": sample.target_item_id,
            "ranked": orders[sample.sample_id],
            "cf": sample.cf_items,
            "direct": sample.sasrec_items,
            "union": list(sample.candidates),
        }
        for sample in samples
    ]
    metrics = rank_metrics(rows, "ranked")
    cf_ranks = [candidate_rank(row["target_item_id"], row["cf"]) for row in rows]
    direct_ranks = [candidate_rank(row["target_item_id"], row["direct"]) for row in rows]
    ranked_ranks = [candidate_rank(row["target_item_id"], row["ranked"]) for row in rows]
    union_ranks = [candidate_rank(row["target_item_id"], row["union"]) for row in rows]
    cf_hit20 = [hit(rank, 20) for rank in cf_ranks]
    direct_hit20 = [hit(rank, 20) for rank in direct_ranks]
    ranked_hit20 = [hit(rank, 20) for rank in ranked_ranks]
    direct_only = [not cf and direct for cf, direct in zip(cf_hit20, direct_hit20)]
    overlap = [cf and direct for cf, direct in zip(cf_hit20, direct_hit20)]
    union_oracle = [rank is not None for rank in union_ranks]
    metrics.update(
        {
            "union_target_in_pool_count": sum(union_oracle),
            "union_target_in_pool_rate": sum(union_oracle) / len(rows) if rows else 0.0,
            "union_oracle_hits@1": sum(union_oracle),
            "union_oracle_hits@5": sum(union_oracle),
            "union_oracle_hits@10": sum(union_oracle),
            "union_oracle_hits@20": sum(union_oracle),
            "union_oracle_hits@50": sum(union_oracle),
            "cf_hits_at20": sum(cf_hit20),
            "projected_hits_at20": sum(ranked_hit20),
            "cf_hits_preserved_at20": sum(cf and ranked for cf, ranked in zip(cf_hit20, ranked_hit20)),
            "cf_hits_lost_at20": sum(cf and not ranked for cf, ranked in zip(cf_hit20, ranked_hit20)),
            "direct_only_targets_available_at20": sum(direct_only),
            "direct_only_targets_recovered_at20": sum(do and ranked for do, ranked in zip(direct_only, ranked_hit20)),
            "overlap_source_target_recovery_at20": sum(ov and ranked for ov, ranked in zip(overlap, ranked_hit20)),
            "candidate_to_top20_conversion_rate": (
                sum(u and ranked for u, ranked in zip(union_oracle, ranked_hit20)) / sum(union_oracle)
                if sum(union_oracle)
                else 0.0
            ),
            "average_candidate_pool_size": sum(len(row["union"]) for row in rows) / len(rows) if rows else 0.0,
        }
    )
    available = metrics["direct_only_targets_available_at20"]
    metrics["direct_only_recovery_rate_at20"] = (
        metrics["direct_only_targets_recovered_at20"] / available if available else 0.0
    )
    return metrics


def baseline_metrics(samples: list[s5.Sample], base_config: dict[str, Any]) -> dict[str, Any]:
    orders = {
        "cf_only": {sample.sample_id: s5.cf_only(sample) for sample in samples},
        "direct_order": {sample.sample_id: s5.sasrec_only(sample) for sample in samples},
        "fixed_rrf": {sample.sample_id: s5.policy_items(sample, base_config) for sample in samples},
    }
    return {name: metrics_for_orders(samples, order) for name, order in orders.items()}


def build_samples_for_split(output_root: Path, bundle_root: Path, split: str) -> tuple[list[s5.Sample], list[dict[str, Any]]]:
    cf_rows = read_jsonl(cf_split_path(output_root, split))
    union_rows = read_jsonl(union_path(output_root, split))
    direct_rows = direct_rows_from_union(union_rows)
    samples = s5.build_samples(cf_rows, direct_rows, split_map_for(cf_rows, split))
    return samples, union_rows


def build_s5_samples_for_split(split: str, split_row_ids: set[str]) -> list[s5.Sample]:
    cf_rows = [
        row for row in read_jsonl(s6union.DEFAULT_CF_FULL)
        if str(row.get("row_index", row.get("sample_id", ""))) in split_row_ids
    ]
    sasrec_rows = [
        row for row in read_jsonl(s6union.DEFAULT_SASREC_SID_FULL)
        if str(row.get("row_index", row.get("sample_id", ""))) in split_row_ids
    ]
    return s5.build_samples(cf_rows, sasrec_rows, {str(row.get("row_index")): split for row in cf_rows})


def aggregate_named_metrics(named: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in next(iter(named.values())):
        total = sum(named[split][name]["num_samples"] for split in SPLITS)
        item: dict[str, Any] = {"num_samples": total}
        for key in [
            "hits@1",
            "hits@5",
            "hits@10",
            "hits@20",
            "hits@50",
            "target_in_pool_count",
            "union_target_in_pool_count",
            "cf_hits_at20",
            "projected_hits_at20",
            "cf_hits_preserved_at20",
            "cf_hits_lost_at20",
            "direct_only_targets_available_at20",
            "direct_only_targets_recovered_at20",
            "overlap_source_target_recovery_at20",
        ]:
            if key in next(iter(named.values()))[name]:
                item[key] = sum(named[split][name].get(key, 0) for split in SPLITS)
        for k in KS:
            hits = item.get(f"hits@{k}")
            if hits is not None:
                item[f"hr@{k}"] = hits / total if total else 0.0
            ndcg_sum = sum(named[split][name].get(f"ndcg@{k}", 0.0) * named[split][name]["num_samples"] for split in SPLITS)
            item[f"ndcg@{k}"] = ndcg_sum / total if total else 0.0
        item["mrr"] = sum(named[split][name]["mrr"] * named[split][name]["num_samples"] for split in SPLITS) / total if total else 0.0
        if item.get("union_target_in_pool_count") is not None:
            item["union_target_in_pool_rate"] = item["union_target_in_pool_count"] / total if total else 0.0
        if item.get("target_in_pool_count") is not None:
            item["target_in_pool_rate"] = item["target_in_pool_count"] / total if total else 0.0
        if item.get("direct_only_targets_available_at20"):
            item["direct_only_recovery_rate_at20"] = item["direct_only_targets_recovered_at20"] / item["direct_only_targets_available_at20"]
        item["average_candidate_pool_size"] = sum(
            named[split][name].get("average_candidate_pool_size", 0.0) * named[split][name]["num_samples"]
            for split in SPLITS
        ) / total if total else 0.0
        out[name] = item
    return out


def uplift_retention(s6: dict[str, Any], s5m: dict[str, Any], cf: dict[str, Any]) -> dict[str, Any]:
    hr_denom = s5m["hr@20"] - cf["hr@20"]
    ndcg_denom = s5m["ndcg@20"] - cf["ndcg@20"]
    return {
        "hr20": None if hr_denom <= 0 else (s6["hr@20"] - cf["hr@20"]) / hr_denom,
        "ndcg20": None if ndcg_denom <= 0 else (s6["ndcg@20"] - cf["ndcg@20"]) / ndcg_denom,
        "hr20_denominator": hr_denom,
        "ndcg20_denominator": ndcg_denom,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in [args.config, args.output_root, args.report]:
        reject_test_path(path)
    config = load_frozen_config(args.config)
    model, contract = load_model_and_contract(config)
    base_config = source_aware_rrf_config(config)

    matrix, row_index = s5.load_embeddings(
        ROOT / f"data/Amazon/cs_embeddings/{CATEGORY}/{CATEGORY}.cf_emb.npy",
        ROOT / f"data/Amazon/cs_embeddings/{CATEGORY}/{CATEGORY}.row_index.json",
    )
    popularity = s5.train_popularity(ROOT / f"data/Amazon/sid_versions/cf_k512_dedup/{CATEGORY}/train.csv")
    max_log_pop = max((math.log1p(value) for value in popularity.values()), default=1.0)

    projection_manifest = {
        "schema": "s6_frozen_ranker_projection_manifest.v1",
        "run_id": args.run_id,
        "projection_name": "source_independent_projection",
        "ranker_model_path": config["ranker"]["model_path"],
        "ranker_model_sha256": file_sha256(ROOT / config["ranker"]["model_path"]),
        "feature_names": model["feature_names"],
        "source_specific_features_zeroed_after_normalization": s5final.SOURCE_SPECIFIC_FEATURES_ZEROED,
        "forbidden_direct_score_features": sorted(FORBIDDEN_DIRECT_FEATURES),
        "direct_auxiliary_mapping": {
            "direct_sasrec_rank": "auxiliary source rank used only through source-independent best-rank/fusion-rank features",
            "direct_sasrec_raw_scores": "excluded",
            "overlap": "preserved in provenance; source identity features are zeroed by projection",
        },
        "frozen_policy": {
            "selected_candidate_policy": config["selected_candidate_policy"],
            "lambda_sasrec": config["lambda_sasrec"],
            "source_bonus": config["source_bonus"],
        },
        "feature_contract": contract,
    }

    projection_manifest_path = args.output_root / "frozen_ranker" / args.run_id / "projection_manifest.json"
    execution_log_path = args.output_root / "frozen_ranker" / args.run_id / "execution_log.json"
    exit_code_path = args.output_root / "frozen_ranker" / args.run_id / "exit_code.txt"
    write_json(projection_manifest_path, projection_manifest)

    per_split: dict[str, dict[str, Any]] = {}
    s5_per_split: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        samples, _union_rows = build_samples_for_split(args.output_root, Path(), split)
        split_paths = output_paths(args.output_root, split)
        baselines = baseline_metrics(samples, base_config)
        orders, feature_hash, ranked_rows = ranked_with_feature_hash(
            samples,
            base_config,
            model,
            matrix,
            row_index,
            popularity,
            max_log_pop,
        )
        projected = metrics_for_orders(samples, orders)
        projected["feature_audit_hash"] = feature_hash
        write_jsonl(split_paths["ranked"], ranked_rows)
        metrics = {
            "schema": "s6_frozen_ranker_split_metrics.v1",
            "split": split,
            "baselines": baselines,
            "projected_ranker": projected,
            "input_hashes": {
                "cf_split_candidates": file_sha256(cf_split_path(args.output_root, split)),
                "union_candidates": file_sha256(union_path(args.output_root, split)),
            },
            "output_hashes": {
                "ranked_candidates": file_sha256(split_paths["ranked"]),
            },
            "test_read": False,
        }
        write_json(split_paths["metrics"], metrics)
        per_split[split] = metrics

        split_ids = {sample.sample_id for sample in samples}
        s5_samples = build_s5_samples_for_split(split, split_ids)
        s5_baselines = baseline_metrics(s5_samples, base_config)
        s5_orders, s5_feature_hash, _ = ranked_with_feature_hash(
            s5_samples,
            base_config,
            model,
            matrix,
            row_index,
            popularity,
            max_log_pop,
        )
        s5_projected = metrics_for_orders(s5_samples, s5_orders)
        s5_projected["feature_audit_hash"] = s5_feature_hash
        s5_per_split[split] = {"baselines": s5_baselines, "projected_ranker": s5_projected}

    aggregate = aggregate_named_metrics(
        {split: {"cf_only": per_split[split]["baselines"]["cf_only"],
                 "direct_order": per_split[split]["baselines"]["direct_order"],
                 "fixed_rrf": per_split[split]["baselines"]["fixed_rrf"],
                 "projected_ranker": per_split[split]["projected_ranker"]}
         for split in SPLITS}
    )
    s5_aggregate = aggregate_named_metrics(
        {split: {"cf_only": s5_per_split[split]["baselines"]["cf_only"],
                 "fixed_rrf": s5_per_split[split]["baselines"]["fixed_rrf"],
                 "projected_ranker": s5_per_split[split]["projected_ranker"]}
         for split in SPLITS}
    )
    comparisons = {}
    for split in SPLITS:
        comparisons[split] = {
            "s6_vs_s5_uplift_retention": uplift_retention(
                per_split[split]["projected_ranker"],
                s5_per_split[split]["projected_ranker"],
                per_split[split]["baselines"]["cf_only"],
            ),
            "ranking_gate": {},
        }
        retention = comparisons[split]["s6_vs_s5_uplift_retention"]
        projected = per_split[split]["projected_ranker"]
        cf = per_split[split]["baselines"]["cf_only"]
        comparisons[split]["ranking_gate"] = {
            "hr20_retention_pass": retention["hr20"] is not None and retention["hr20"] >= 0.8,
            "ndcg20_retention_pass": retention["ndcg20"] is not None and retention["ndcg20"] >= 0.8,
            "cf_hit_preservation_rate": (
                projected["cf_hits_preserved_at20"] / cf["hits@20"] if cf["hits@20"] else 1.0
            ),
            "cf_hit_preservation_pass": (projected["cf_hits_preserved_at20"] / cf["hits@20"] if cf["hits@20"] else 1.0) >= 0.97,
        }

    fit_select_pass = all(
        comparisons[split]["ranking_gate"]["hr20_retention_pass"]
        and comparisons[split]["ranking_gate"]["ndcg20_retention_pass"]
        and comparisons[split]["ranking_gate"]["cf_hit_preservation_pass"]
        for split in ("valid_fit", "valid_select")
    )
    gate_confirms = (
        comparisons["valid_gate"]["ranking_gate"]["hr20_retention_pass"]
        and comparisons["valid_gate"]["ranking_gate"]["ndcg20_retention_pass"]
        and comparisons["valid_gate"]["ranking_gate"]["cf_hit_preservation_pass"]
    )
    if fit_select_pass and gate_confirms:
        verdict = "GO_COST_EFFECTIVENESS_CLOSEOUT"
    elif aggregate["projected_ranker"]["union_target_in_pool_count"] > aggregate["cf_only"]["target_in_pool_count"]:
        verdict = "GO_LIGHTWEIGHT_RANKER_VALIDATION"
    else:
        verdict = "NO_GO_DIRECT_SASREC_RANKING"

    report = {
        "schema": "s6_frozen_ranker_validation_report.v1",
        "verdict": verdict,
        "run_id": args.run_id,
        "compatibility_verdict": "GO_FROZEN_RANKER_COMPATIBLE",
        "projection_manifest": projection_manifest_path.as_posix(),
        "frozen_ranker": {
            "model_path": config["ranker"]["model_path"],
            "model_sha256": file_sha256(ROOT / config["ranker"]["model_path"]),
            "feature_names": model["feature_names"],
            "model_type": model["model_type"],
            "sort_tie_breaker": model.get("sort_tie_breaker"),
        },
        "per_split": per_split,
        "aggregate": aggregate,
        "s5_dual_qwen_reference": {
            "per_split": s5_per_split,
            "aggregate": s5_aggregate,
        },
        "comparisons": comparisons,
        "valid_gate_confirmation": {
            "used_for_selection": False,
            "confirms": gate_confirms,
            "details": comparisons["valid_gate"],
        },
        "cost_status": {
            "direct_model_inference_seconds": 24.967018,
            "direct_formal_wall_seconds": 34.233973,
            "direct_peak_cuda_allocated_bytes": 13118464,
            "sasrec_sid_qwen_comparable_validation_runtime": None,
            "cost_ratio_gate": "pending validation-side comparable SASRec-SID Qwen runtime evidence",
        },
        "test_read": False,
    }
    write_json(args.report, report)
    write_json(execution_log_path, {"complete": True, "report": args.report.as_posix(), "verdict": verdict})
    reject_existing(exit_code_path)
    exit_code_path.write_text("0\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S6-5 frozen ranker validation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
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
                    "config": args.config.as_posix(),
                    "output_root": args.output_root.as_posix(),
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
