#!/usr/bin/env python3
"""Export validation-only direct SASRec item candidates for S6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_s6_validation_split as split_manifest  # noqa: E402
import build_sasrec_embeddings as sasrec  # noqa: E402


CATEGORY = "Industrial_and_Scientific"
DEFAULT_CONFIG = ROOT / "configs/s6_cost_aware_aux/dev_config.json"
DEFAULT_CHECKPOINT_REPORT = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific/s6_checkpoint_compatibility_report.json"
ALLOWED_K = {20, 50, 100}
DEFAULT_TOPK = [1, 5, 10, 20, 50]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def subset_rows(rows: list[dict[str, str]], split_name: str, seed: int, split_key: str = "user_id") -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        key = str(row.get(split_key, "")).strip()
        if not key:
            raise ValueError(f"missing split key {split_key!r} at row {idx}")
        if split_manifest.split_for_key(key, seed) == split_name:
            copied = dict(row)
            copied["source_row_index"] = str(idx)
            out.append(copied)
    return out


def stable_topk(scores: Any, k: int) -> list[int]:
    import numpy as np  # type: ignore

    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("non-finite SASRec scores")
    order = np.lexsort((np.arange(values.shape[0]), -values))
    return [int(idx) for idx in order[:k]]


def score_features(scores: Any, selected: list[int]) -> dict[int, dict[str, float]]:
    import numpy as np  # type: ignore

    values = np.asarray(scores, dtype=np.float64)
    selected_values = values[selected]
    mean = float(values.mean())
    std = float(values.std())
    if std <= 1e-12 or not math.isfinite(std):
        std = 1.0
    top1 = float(selected_values[0]) if len(selected_values) else 0.0
    topk_mean = float(selected_values.mean()) if len(selected_values) else 0.0
    return {
        idx: {
            "sasrec_direct_rank": float(rank + 1),
            "sasrec_direct_reciprocal_rank": 1.0 / float(rank + 1),
            "sasrec_direct_score": float(values[idx]),
            "sasrec_direct_zscore": (float(values[idx]) - mean) / std,
            "sasrec_direct_top1_margin": top1 - float(values[idx]),
            "sasrec_direct_score_minus_topk_mean": float(values[idx]) - topk_mean,
        }
        for rank, idx in enumerate(selected)
    }


def rank_metrics(ranks: list[int | None], ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"num_samples": len(ranks)}
    for k in sorted(set(ks)):
        hits = 0
        ndcg = 0.0
        for rank in ranks:
            if rank is not None and rank < k:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
        out[f"hr@{k}"] = 0.0 if not ranks else hits / len(ranks)
        out[f"ndcg@{k}"] = 0.0 if not ranks else ndcg / len(ranks)
    out["mrr"] = 0.0 if not ranks else sum(0.0 if rank is None else 1.0 / (rank + 1) for rank in ranks) / len(ranks)
    return out


def load_checked_config(config_path: Path, checkpoint_report_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = json.loads(checkpoint_report_path.read_text(encoding="utf-8"))
    if report.get("verdict") != "GO_CHECKPOINT_COMPATIBLE":
        raise ValueError(f"checkpoint report is not GO: {report.get('verdict')}")
    return config, report


def build_runtime_model(config: dict[str, Any], device: str):
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - local WSL may not have torch
        raise RuntimeError(f"PyTorch is required for direct SASRec export: {exc}") from exc

    ckpt_path = ROOT / config["sasrec_formal_v3"]["declared_checkpoint"]
    formal_config = sasrec.read_json(ROOT / config["sasrec_formal_v3"]["config"])
    item_order, row_index = sasrec.load_canonical_mapping(
        ROOT / "data/Amazon/behavior_embeddings/sasrec/Industrial_and_Scientific/formal_v3_finite_maskfix_seed42/Industrial_and_Scientific.item_order.json",
        ROOT / "data/Amazon/behavior_embeddings/sasrec/Industrial_and_Scientific/formal_v3_finite_maskfix_seed42/Industrial_and_Scientific.row_index.json",
    )
    model_args = argparse.Namespace(
        embedding_dim=formal_config["embedding_dim"],
        max_seq_len=formal_config["max_seq_len"],
        num_heads=formal_config["num_heads"],
        num_layers=formal_config["num_layers"],
        dropout=formal_config["dropout"],
        device=device,
    )
    model = sasrec.build_torch_model(len(row_index), model_args).to(device)
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return torch, model, formal_config, item_order, row_index


def candidate_paths(out_root: Path) -> dict[str, Path]:
    return {
        "candidates": out_root / "candidates.jsonl",
        "features": out_root / "candidate_features.jsonl",
        "report": out_root / "candidate_report.json",
    }


def reject_existing(paths: dict[str, Path]) -> None:
    existing = [path.as_posix() for path in paths.values() if path.exists() and path.stat().st_size > 0]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing S6 direct SASRec artifacts: {existing}")


def output_root_for(args: argparse.Namespace, split: str, k: int, multi_view: bool) -> Path:
    if args.output_dir is not None:
        return args.output_dir / f"k{k}" if multi_view else args.output_dir
    base = args.output_root or (ROOT / f"results/s6_cost_aware_aux/{CATEGORY}/direct_sasrec")
    if args.run_id:
        return base / split / args.run_id / f"k{k}"
    return base / split / f"k{k}"


def make_candidate_rows(scored_rows: list[dict[str, Any]], item_order: list[str], k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int | None]]:
    candidate_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    ranks: list[int | None] = []
    for scored in scored_rows:
        top_indices = scored["top_indices"][:k]
        features = {idx: scored["features"][idx] for idx in top_indices}
        candidate_items = [item_order[idx] for idx in top_indices]
        target_index = scored["target_index"]
        rank = None if target_index is None else next((rank_idx for rank_idx, item_idx in enumerate(top_indices) if item_idx == target_index), None)
        ranks.append(rank)
        details = []
        per_candidate_features = []
        for rank_idx, item_idx in enumerate(top_indices):
            item_id = item_order[item_idx]
            # Re-rank fields are view-local, so prefixes are exact standalone K views.
            view_features = dict(features[item_idx])
            view_features["sasrec_direct_rank"] = float(rank_idx + 1)
            view_features["sasrec_direct_reciprocal_rank"] = 1.0 / float(rank_idx + 1)
            detail = {
                "item_id": item_id,
                "source_type": "sasrec_direct",
                "source_sid": None,
                "matched_prefix": None,
                "sid_rank_0_based": rank_idx,
                "sasrec_direct_rank_0_based": rank_idx,
                "all_source_types": ["sasrec_direct"],
            }
            detail.update(view_features)
            details.append(detail)
            per_candidate_features.append({"item_id": item_id, **view_features})
        row = scored["row"]
        candidate_rows.append(
            {
                "row_index": str(row["source_row_index"]),
                "user_id": str(row.get("user_id", "")),
                "target_item_id": scored["target"],
                "history_item_id": scored["history_item_id"],
                "candidate_item_ids": candidate_items,
                "candidate_details": details,
                "candidate_hit_rank_0_based": rank,
                "candidate_pool_hit": rank is not None,
                "target_candidate_first_source_type": "sasrec_direct" if rank is not None else None,
                "target_candidate_source_types": ["sasrec_direct"] if rank is not None else [],
            }
        )
        feature_rows.append(
            {
                "row_index": str(row["source_row_index"]),
                "target_item_id": scored["target"],
                "candidate_features": per_candidate_features,
            }
        )
    return candidate_rows, feature_rows, ranks


def write_view(
    scored_rows: list[dict[str, Any]],
    item_order: list[str],
    k: int,
    paths: dict[str, Path],
    args: argparse.Namespace,
    config: dict[str, Any],
    checkpoint_report: dict[str, Any],
    source_csv: Path,
    elapsed_model: float,
    derivation_start: float,
    torch: Any,
) -> dict[str, Any]:
    candidate_rows, feature_rows, ranks = make_candidate_rows(scored_rows, item_order, k)
    counts = [len(row["candidate_item_ids"]) for row in candidate_rows]
    write_jsonl(paths["candidates"], candidate_rows)
    write_jsonl(paths["features"], feature_rows)
    metrics = rank_metrics(ranks, [topk for topk in DEFAULT_TOPK if topk <= k])
    report = {
        "schema": "s6_direct_sasrec_candidate_report.v1",
        "split": args.split,
        "num_samples": len(candidate_rows),
        "k": k,
        "candidate_count": {
            "min": min(counts) if counts else 0,
            "mean": sum(counts) / len(counts) if counts else 0.0,
            "max": max(counts) if counts else 0,
        },
        **metrics,
        "target_in_pool_count": sum(rank is not None for rank in ranks),
        "target_in_pool_rate": 0.0 if not ranks else sum(rank is not None for rank in ranks) / len(ranks),
        "invalid_item_count": 0,
        "invalid_item_rate": 0.0,
        "duplicate_count": 0,
        "nan_inf_count": 0,
        "hashes": {
            "checkpoint": checkpoint_report["checkpoint_sha256"],
            "config": file_sha256(args.config),
            "checkpoint_report": file_sha256(args.checkpoint_report),
            "source_csv": file_sha256(source_csv),
            "candidates": file_sha256(paths["candidates"]),
            "candidate_features": file_sha256(paths["features"]),
        },
        "runtime": {
            "wall_clock_seconds": elapsed_model + (time.perf_counter() - derivation_start),
            "model_inference_time_seconds": elapsed_model,
            "k_view_derivation_time_seconds": time.perf_counter() - derivation_start,
            "samples_per_second": len(candidate_rows) / elapsed_model if elapsed_model > 0 else None,
            "candidate_generation_time_seconds": elapsed_model,
            "fusion_time_seconds": None,
            "fusion_time_missing_reason": "not part of direct candidate export",
            "ranker_time_seconds": None,
            "ranker_time_missing_reason": "not part of direct candidate export",
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        },
        "outputs": {key: path.as_posix() for key, path in paths.items()},
        "test_read": False,
        "derived_from_max_k": max([k] + (args.derive_prefix_views or [])),
    }
    write_json(paths["report"], report)
    return report


def export_candidates(args: argparse.Namespace) -> dict[str, Any]:
    config, checkpoint_report = load_checked_config(args.config, args.checkpoint_report)
    if args.k not in ALLOWED_K:
        raise ValueError(f"K must be one of {sorted(ALLOWED_K)}, got {args.k}")
    views = sorted(set(args.derive_prefix_views or [args.k]))
    if any(view not in ALLOWED_K for view in views):
        raise ValueError(f"derive-prefix views must be in {sorted(ALLOWED_K)}, got {views}")
    if max(views) > args.k:
        raise ValueError("--k must be at least the largest derived prefix view")
    source_csv = ROOT / config["source_valid_csv"]
    rows = read_csv_rows(source_csv)
    split_rows = subset_rows(rows, args.split, int(config["seed"]))
    multi_view = len(views) > 1
    view_paths = {view: candidate_paths(output_root_for(args, args.split, view, multi_view)) for view in views}
    plan = {
        "config": args.config.as_posix(),
        "checkpoint_report": args.checkpoint_report.as_posix(),
        "checkpoint": config["sasrec_formal_v3"]["declared_checkpoint"],
        "source_csv": config["source_valid_csv"],
        "split": args.split,
        "k": args.k,
        "derive_prefix_views": views,
        "run_id": args.run_id,
        "num_rows": len(split_rows),
        "outputs": {
            str(view): {key: path.as_posix() for key, path in paths.items()}
            for view, paths in view_paths.items()
        },
    }
    if args.dry_run or os.environ.get("DRY_RUN") == "1":
        print("S6_DIRECT_SASREC_DRY_RUN")
        print(json.dumps(plan, indent=2, sort_keys=True))
        return {"dry_run": True, "plan": plan}
    for paths in view_paths.values():
        reject_existing(paths)

    import numpy as np  # type: ignore

    start = time.perf_counter()
    torch, model, formal_config, item_order, row_index = build_runtime_model(config, args.device)
    invalid_count = 0
    duplicate_count = 0
    nonfinite_count = 0
    scored_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for local_idx, row in enumerate(split_rows):
            history_seq = sasrec.row_sequence(row, row_index, include_target=False)
            target = str(row.get("item_id", "")).strip()
            target_internal = sasrec.to_internal_id(target, row_index)
            if not history_seq:
                raise ValueError(f"row {local_idx} has no valid history items")
            inputs = torch.tensor(
                [sasrec.valid_input(history_seq, int(formal_config["max_seq_len"]))],
                dtype=torch.long,
                device=args.device,
            )
            logits = model.last_valid_logits(inputs, check_finite=True, finite_context=sasrec.finite_context(inputs))[0]
            scores = logits.detach().cpu().numpy()
            if not np.isfinite(scores).all():
                nonfinite_count += 1
                raise ValueError(f"non-finite scores at row {local_idx}")
            top_indices = stable_topk(scores, args.k)
            if len(set(top_indices)) != len(top_indices):
                duplicate_count += 1
                raise ValueError(f"duplicate top-K indices at row {local_idx}")
            features = score_features(scores, top_indices)
            candidate_items = [item_order[idx] for idx in top_indices]
            invalid_count += sum(1 for item in candidate_items if item not in row_index)
            scored_rows.append(
                {
                    "row": row,
                    "target": target,
                    "target_index": None if target_internal is None else target_internal - 1,
                    "history_item_id": [str(item) for item in sasrec.parse_list(row.get("history_item_id", ""))],
                    "top_indices": top_indices,
                    "features": features,
                }
            )
    elapsed_model = time.perf_counter() - start
    if invalid_count or duplicate_count or nonfinite_count:
        raise ValueError(f"invalid direct SASRec output: invalid={invalid_count} duplicate={duplicate_count} nonfinite={nonfinite_count}")
    reports = {}
    for view in views:
        derivation_start = time.perf_counter()
        reports[str(view)] = write_view(
            scored_rows,
            item_order,
            view,
            view_paths[view],
            args,
            config,
            checkpoint_report,
            source_csv,
            elapsed_model,
            derivation_start,
            torch,
        )
    print(json.dumps({"candidate_reports": {k: v["outputs"]["report"] for k, v in reports.items()}, "num_samples": len(scored_rows)}, indent=2))
    return reports[str(args.k)] if str(args.k) in reports else reports[str(max(views))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export S6 direct SASRec validation candidates.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-report", type=Path, default=DEFAULT_CHECKPOINT_REPORT)
    parser.add_argument("--split", choices=["valid_fit", "valid_select", "valid_gate"], required=True)
    parser.add_argument("--k", type=int, choices=sorted(ALLOWED_K), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--derive-prefix-views", type=int, nargs="*", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    export_candidates(parse_args())


if __name__ == "__main__":
    main()
