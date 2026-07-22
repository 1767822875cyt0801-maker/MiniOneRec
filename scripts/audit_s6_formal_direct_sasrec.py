#!/usr/bin/env python3
"""Read-only audit for S6 formal direct-SASRec validation artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import ast
import json
import math
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import build_s6_validation_split as split_manifest


ROOT = Path(__file__).resolve().parents[1]
CATEGORY = "Industrial_and_Scientific"
DEFAULT_BUNDLE = ROOT / "incoming/s6_formal_direct_sasrec/s6_formal_direct_sasrec_formal_v1_bundle.tar.gz"
DEFAULT_SHA = DEFAULT_BUNDLE.with_suffix(DEFAULT_BUNDLE.suffix + ".sha256")
DEFAULT_OUTPUT = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific/s6_3_formal_direct_sasrec_audit_report.json"
DEFAULT_CF_CANDIDATES = (
    ROOT
    / "incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036"
    / "results/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/formal/valid/full_valid"
    / "candidates/candidates.jsonl"
)
SPLITS = ("valid_fit", "valid_select", "valid_gate")
KS = (20, 50, 100)
METRIC_CUTOFFS = (1, 5, 10, 20, 50, 100)
FINITE_FEATURES = (
    "sasrec_direct_score",
    "sasrec_direct_zscore",
    "sasrec_direct_top1_margin",
    "sasrec_direct_score_minus_topk_mean",
    "sasrec_direct_reciprocal_rank",
    "sasrec_direct_rank",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                row["_line_no"] = line_no
                rows.append(row)
    return rows


def read_valid_csv(path: Path) -> list[dict[str, str]]:
    if "test" in [part.lower() for part in path.parts]:
        raise ValueError(f"refusing test path in S6 formal audit: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def parse_sha_file(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    parts = text.split()
    if len(parts) < 2:
        raise ValueError(f"invalid sha256 file: {path}")
    return parts[0], parts[-1].lstrip("*")


def verify_outer_sha(bundle: Path, sha_file: Path) -> dict[str, Any]:
    expected, name = parse_sha_file(sha_file)
    actual = file_sha256(bundle)
    return {
        "bundle": bundle.as_posix(),
        "sha_file": sha_file.as_posix(),
        "sha_file_name": name,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "ok": expected == actual,
    }


def safe_extract(bundle: Path, destination: Path) -> None:
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(destination, filter="data")


def verify_internal_sha(extracted_root: Path) -> dict[str, Any]:
    sha_path = (
        extracted_root
        / "results/s6_cost_aware_aux/Industrial_and_Scientific/s6_3_formal_evidence/formal_v1/formal_v1_files.sha256"
    )
    checked = []
    for line_no, line in enumerate(sha_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        expected, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*")
        path = extracted_root / rel
        actual = file_sha256(path) if path.is_file() else None
        checked.append({"line": line_no, "path": rel, "expected": expected, "actual": actual, "ok": expected == actual})
    return {
        "sha_file": sha_path.relative_to(extracted_root).as_posix(),
        "checked_files": len(checked),
        "failures": [row for row in checked if not row["ok"]],
        "ok": all(row["ok"] for row in checked),
    }


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_index", row.get("sample_id", "")))


def normalize_history(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
    return [text]


def source_rows_by_split(valid_rows: list[dict[str, str]], seed: int) -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {split: {} for split in SPLITS}
    for idx, row in enumerate(valid_rows):
        split = split_manifest.split_for_key(str(row["user_id"]), seed)
        copied = dict(row)
        copied["source_row_index"] = str(idx)
        out[split][str(idx)] = copied
    return out


def rank_metrics(ranks: list[int | None], cutoffs: tuple[int, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {"num_samples": len(ranks)}
    for k in cutoffs:
        hits = 0
        ndcg = 0.0
        for rank in ranks:
            if rank is not None and rank < k:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
        out[f"hr@{k}"] = 0.0 if not ranks else hits / len(ranks)
        out[f"ndcg@{k}"] = 0.0 if not ranks else ndcg / len(ranks)
        out[f"hits@{k}"] = hits
    out["mrr"] = 0.0 if not ranks else sum(0.0 if rank is None else 1.0 / (rank + 1) for rank in ranks) / len(ranks)
    out["target_in_pool_count"] = sum(rank is not None for rank in ranks)
    out["target_in_pool_rate"] = 0.0 if not ranks else out["target_in_pool_count"] / len(ranks)
    return out


def assert_close(left: float, right: float, tol: float = 1e-12) -> bool:
    return abs(left - right) <= tol


def audit_candidate_view(
    root: Path,
    split: str,
    k: int,
    expected_rows: dict[str, dict[str, str]],
    item_universe: set[str],
) -> dict[str, Any]:
    view = root / f"results/s6_cost_aware_aux/{CATEGORY}/direct_sasrec/{split}/formal_v1/k{k}"
    candidates_path = view / "candidates.jsonl"
    features_path = view / "candidate_features.jsonl"
    report_path = view / "candidate_report.json"
    candidates = read_jsonl(candidates_path)
    features = read_jsonl(features_path)
    report = read_json(report_path)

    failures: list[str] = []
    if len(candidates) != len(expected_rows):
        failures.append(f"{split}/k{k}: candidate row count {len(candidates)} != expected {len(expected_rows)}")
    if len(features) != len(candidates):
        failures.append(f"{split}/k{k}: feature row count {len(features)} != candidate row count {len(candidates)}")

    ranks: list[int | None] = []
    invalid_items = 0
    duplicate_rows = 0
    duplicate_candidates = 0
    nonfinite_count = 0
    rank_violations = 0
    monotonic_violations = 0
    row_alignment_failures = 0
    feature_alignment_failures = 0

    seen_row_ids: set[str] = set()
    feature_by_row = {row_id(row): row for row in features}
    for row in candidates:
        rid = row_id(row)
        if rid in seen_row_ids:
            duplicate_rows += 1
        seen_row_ids.add(rid)
        expected = expected_rows.get(rid)
        if expected is None:
            row_alignment_failures += 1
            continue
        if str(row.get("target_item_id")) != str(expected.get("item_id")):
            row_alignment_failures += 1
        if normalize_history(row.get("history_item_id")) != normalize_history(expected.get("history_item_id")):
            row_alignment_failures += 1
        if str(row.get("user_id")) != str(expected.get("user_id")):
            row_alignment_failures += 1
        items = [str(item) for item in row.get("candidate_item_ids", [])]
        if len(items) != k:
            failures.append(f"{split}/k{k}/row{rid}: candidate count {len(items)} != {k}")
        duplicate_candidates += len(items) - len(set(items))
        invalid_items += sum(item not in item_universe for item in items)

        details = row.get("candidate_details", [])
        if len(details) != len(items):
            feature_alignment_failures += 1
        previous_score = None
        for rank, detail in enumerate(details):
            if str(detail.get("item_id")) != items[rank]:
                feature_alignment_failures += 1
            if int(detail.get("sid_rank_0_based", -1)) != rank:
                rank_violations += 1
            if int(detail.get("sasrec_direct_rank_0_based", -1)) != rank:
                rank_violations += 1
            if not assert_close(float(detail.get("sasrec_direct_rank", -1.0)), float(rank + 1)):
                rank_violations += 1
            if not assert_close(float(detail.get("sasrec_direct_reciprocal_rank", -1.0)), 1.0 / float(rank + 1)):
                rank_violations += 1
            score = detail.get("sasrec_direct_score")
            if score is not None:
                score = float(score)
                if previous_score is not None and score > previous_score + 1e-12:
                    monotonic_violations += 1
                previous_score = score
            for key in FINITE_FEATURES:
                value = detail.get(key)
                if value is None or not math.isfinite(float(value)):
                    nonfinite_count += 1

        feature_row = feature_by_row.get(rid)
        if feature_row is None:
            feature_alignment_failures += 1
        else:
            feature_items = [str(item.get("item_id")) for item in feature_row.get("candidate_features", [])]
            if feature_items != items:
                feature_alignment_failures += 1
            for feature in feature_row.get("candidate_features", []):
                for key in FINITE_FEATURES:
                    value = feature.get(key)
                    if value is None or not math.isfinite(float(value)):
                        nonfinite_count += 1

        target = str(row.get("target_item_id"))
        recomputed_rank = next((idx for idx, item in enumerate(items) if item == target), None)
        ranks.append(recomputed_rank)
        declared_rank = row.get("candidate_hit_rank_0_based")
        if declared_rank != recomputed_rank:
            rank_violations += 1
        if bool(row.get("candidate_pool_hit")) != (recomputed_rank is not None):
            rank_violations += 1

    missing_expected = set(expected_rows) - seen_row_ids
    row_alignment_failures += len(missing_expected)

    recomputed = rank_metrics(ranks, tuple(cutoff for cutoff in METRIC_CUTOFFS if cutoff <= k))
    metric_mismatches: list[str] = []
    for key, value in recomputed.items():
        if key.startswith("hits@"):
            continue
        if key in report and isinstance(value, float):
            if not assert_close(value, float(report[key])):
                metric_mismatches.append(f"{key}: report={report[key]} recomputed={value}")
        elif key in report and report[key] != value:
            metric_mismatches.append(f"{key}: report={report[key]} recomputed={value}")

    hash_mismatches = []
    hashes = report.get("hashes", {})
    for key, path in (("candidates", candidates_path), ("candidate_features", features_path)):
        actual = file_sha256(path)
        if hashes.get(key) != actual:
            hash_mismatches.append({"artifact": key, "report": hashes.get(key), "actual": actual})

    safety = {
        "row_count": len(candidates),
        "feature_row_count": len(features),
        "candidate_count_min": min((len(row.get("candidate_item_ids", [])) for row in candidates), default=0),
        "candidate_count_max": max((len(row.get("candidate_item_ids", [])) for row in candidates), default=0),
        "duplicate_rows": duplicate_rows,
        "duplicate_candidates": duplicate_candidates,
        "invalid_item_count": invalid_items,
        "nonfinite_feature_count": nonfinite_count,
        "rank_violations": rank_violations,
        "monotonic_violations": monotonic_violations,
        "row_alignment_failures": row_alignment_failures,
        "feature_alignment_failures": feature_alignment_failures,
        "metric_mismatches": metric_mismatches,
        "hash_mismatches": hash_mismatches,
        "test_read": report.get("test_read"),
    }
    ok = not failures and all(
        safety[key] == 0
        for key in (
            "duplicate_rows",
            "duplicate_candidates",
            "invalid_item_count",
            "nonfinite_feature_count",
            "rank_violations",
            "monotonic_violations",
            "row_alignment_failures",
            "feature_alignment_failures",
        )
    ) and not metric_mismatches and not hash_mismatches and report.get("test_read") is False

    return {
        "split": split,
        "k": k,
        "ok": ok,
        "failures": failures,
        "safety": safety,
        "metrics": recomputed,
        "report_metrics": {key: report.get(key) for key in report if key.startswith("hr@") or key.startswith("ndcg@") or key in {"mrr", "target_in_pool_count", "target_in_pool_rate", "num_samples"}},
        "runtime": report.get("runtime", {}),
        "derived_from_max_k": report.get("derived_from_max_k"),
        "hashes": hashes,
        "paths": {
            "candidates": candidates_path.relative_to(root).as_posix(),
            "features": features_path.relative_to(root).as_posix(),
            "report": report_path.relative_to(root).as_posix(),
        },
    }


def audit_prefix_consistency(root: Path, split: str) -> dict[str, Any]:
    base = root / f"results/s6_cost_aware_aux/{CATEGORY}/direct_sasrec/{split}/formal_v1"
    rows_by_k = {k: read_jsonl(base / f"k{k}/candidates.jsonl") for k in KS}
    features_by_k = {k: read_jsonl(base / f"k{k}/candidate_features.jsonl") for k in KS}
    failures = []
    rows_100 = {row_id(row): row for row in rows_by_k[100]}
    features_100 = {row_id(row): row for row in features_by_k[100]}
    for k in (20, 50):
        for row in rows_by_k[k]:
            rid = row_id(row)
            base_row = rows_100.get(rid)
            if base_row is None:
                failures.append(f"{split}/k{k}: row {rid} missing from k100")
                continue
            if row.get("candidate_item_ids") != base_row.get("candidate_item_ids", [])[:k]:
                failures.append(f"{split}/k{k}: candidate prefix mismatch at row {rid}")
            if row.get("candidate_details") != base_row.get("candidate_details", [])[:k]:
                failures.append(f"{split}/k{k}: detail prefix mismatch at row {rid}")
        for row in features_by_k[k]:
            rid = row_id(row)
            base_row = features_100.get(rid)
            if base_row is None:
                failures.append(f"{split}/k{k}: feature row {rid} missing from k100")
                continue
            if row.get("candidate_features") != base_row.get("candidate_features", [])[:k]:
                failures.append(f"{split}/k{k}: feature prefix mismatch at row {rid}")
    return {"split": split, "ok": not failures, "failures": failures[:20], "failure_count": len(failures)}


def audit_cf_alignment(cf_path: Path, expected_by_split: dict[str, dict[str, dict[str, str]]]) -> dict[str, Any]:
    rows = read_jsonl(cf_path)
    cf_by_row = {row_id(row): row for row in rows}
    duplicate_rows = len(rows) - len(cf_by_row)
    split_summary: dict[str, Any] = {}
    for split, expected_rows in expected_by_split.items():
        missing = sorted(set(expected_rows) - set(cf_by_row), key=int)
        target_mismatch = 0
        history_mismatch = 0
        for rid, expected in expected_rows.items():
            row = cf_by_row.get(rid)
            if row is None:
                continue
            if str(row.get("target_item_id")) != str(expected.get("item_id")):
                target_mismatch += 1
            if normalize_history(row.get("history_item_id")) != normalize_history(expected.get("history_item_id")):
                history_mismatch += 1
        split_summary[split] = {
            "expected_rows": len(expected_rows),
            "missing_rows": len(missing),
            "target_mismatch": target_mismatch,
            "history_mismatch": history_mismatch,
            "ok": not missing and target_mismatch == 0 and history_mismatch == 0,
        }
    return {
        "cf_candidates": cf_path.as_posix(),
        "exists": cf_path.is_file(),
        "sha256": file_sha256(cf_path) if cf_path.is_file() else None,
        "row_count": len(rows),
        "duplicate_rows": duplicate_rows,
        "split_alignment": split_summary,
        "ok": cf_path.is_file() and duplicate_rows == 0 and all(item["ok"] for item in split_summary.values()),
    }


def aggregate_runtime(view_audits: list[dict[str, Any]]) -> dict[str, Any]:
    k100 = [row for row in view_audits if row["k"] == 100]
    model_seconds = sum(float(row["runtime"].get("model_inference_time_seconds", 0.0)) for row in k100)
    wall_seconds = sum(float(row["runtime"].get("wall_clock_seconds", 0.0)) for row in k100)
    samples = sum(int(row["metrics"]["num_samples"]) for row in k100)
    return {
        "aggregation_policy": "sum each split's k100 report once; k20/k50 are derived views and are not double-counted",
        "model_inference_seconds": model_seconds,
        "formal_wall_seconds": wall_seconds,
        "view_derivation_and_write_seconds": wall_seconds - model_seconds,
        "samples": samples,
        "samples_per_model_second": samples / model_seconds if model_seconds else None,
        "samples_per_wall_second": samples / wall_seconds if wall_seconds else None,
        "cuda_peak_allocated_bytes_max": max(int(row["runtime"].get("cuda_peak_allocated_bytes", 0)) for row in k100),
        "cuda_peak_reserved_bytes_max": max(int(row["runtime"].get("cuda_peak_reserved_bytes", 0)) for row in k100),
    }


def split_stability(view_audits: list[dict[str, Any]]) -> dict[str, Any]:
    by_k: dict[int, list[dict[str, Any]]] = {k: [] for k in KS}
    for audit in view_audits:
        by_k[audit["k"]].append(audit)
    out: dict[str, Any] = {}
    for k, rows in by_k.items():
        hr20 = [row["metrics"].get("hr@20", 0.0) for row in rows]
        ndcg20 = [row["metrics"].get("ndcg@20", 0.0) for row in rows]
        pool = [row["metrics"]["target_in_pool_rate"] for row in rows]
        out[f"k{k}"] = {
            "hr20_min": min(hr20),
            "hr20_max": max(hr20),
            "hr20_range": max(hr20) - min(hr20),
            "ndcg20_min": min(ndcg20),
            "ndcg20_max": max(ndcg20),
            "ndcg20_range": max(ndcg20) - min(ndcg20),
            "target_in_pool_min": min(pool),
            "target_in_pool_max": max(pool),
            "target_in_pool_range": max(pool) - min(pool),
        }
    return out


def recommend_k(view_audits: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[int, dict[str, float]] = {}
    for k in KS:
        rows = [row for row in view_audits if row["k"] == k]
        samples = sum(row["metrics"]["num_samples"] for row in rows)
        hits = sum(row["metrics"].get(f"hits@{k}", row["metrics"]["target_in_pool_count"]) for row in rows)
        hits20 = sum(row["metrics"].get("hits@20", 0) for row in rows)
        totals[k] = {
            "samples": samples,
            "target_in_pool_count": hits,
            "target_in_pool_rate": hits / samples,
            "hits_at_20": hits20,
            "hr_at_20": hits20 / samples,
        }
    increments = {
        "k20_to_k50_added_targets": totals[50]["target_in_pool_count"] - totals[20]["target_in_pool_count"],
        "k50_to_k100_added_targets": totals[100]["target_in_pool_count"] - totals[50]["target_in_pool_count"],
    }
    return {
        "budget_metrics": totals,
        "increments": increments,
        "recommendation": "carry K20, K50, and K100 into union validation; use K100 as the recall ceiling and let validation-side union/ranking gates choose the budget",
        "reason": "K100 views are already derived from one K100 model pass per split, while K50 and K100 expose additional target-in-pool headroom that must be tested against downstream candidate noise.",
    }


def build_metric_table(view_audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for audit in sorted(view_audits, key=lambda row: (row["split"], row["k"])):
        metrics = audit["metrics"]
        rows.append(
            {
                "split": audit["split"],
                "k": audit["k"],
                "samples": metrics["num_samples"],
                "HR@1": metrics.get("hr@1"),
                "HR@5": metrics.get("hr@5"),
                "HR@10": metrics.get("hr@10"),
                "HR@20": metrics.get("hr@20"),
                "HR@50": metrics.get("hr@50"),
                "HR@100": metrics.get("hr@100"),
                "NDCG@1": metrics.get("ndcg@1"),
                "NDCG@5": metrics.get("ndcg@5"),
                "NDCG@10": metrics.get("ndcg@10"),
                "NDCG@20": metrics.get("ndcg@20"),
                "NDCG@50": metrics.get("ndcg@50"),
                "NDCG@100": metrics.get("ndcg@100"),
                "MRR": metrics["mrr"],
                "target_in_pool_count": metrics["target_in_pool_count"],
                "target_in_pool_rate": metrics["target_in_pool_rate"],
            }
        )
    return rows


def aggregate_metrics_by_k(view_audits: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in KS:
        rows = [row for row in view_audits if row["k"] == k]
        samples = sum(row["metrics"]["num_samples"] for row in rows)
        item: dict[str, Any] = {"samples": samples}
        for cutoff in (1, 5, 10, 20, 50, 100):
            if cutoff > k:
                continue
            hits = sum(row["metrics"].get(f"hits@{cutoff}", 0) for row in rows)
            ndcg_sum = sum(row["metrics"].get(f"ndcg@{cutoff}", 0.0) * row["metrics"]["num_samples"] for row in rows)
            item[f"hits@{cutoff}"] = hits
            item[f"HR@{cutoff}"] = hits / samples if samples else 0.0
            item[f"NDCG@{cutoff}"] = ndcg_sum / samples if samples else 0.0
        item["MRR"] = sum(row["metrics"]["mrr"] * row["metrics"]["num_samples"] for row in rows) / samples if samples else 0.0
        item["target_in_pool_count"] = sum(row["metrics"]["target_in_pool_count"] for row in rows)
        item["target_in_pool_rate"] = item["target_in_pool_count"] / samples if samples else 0.0
        out[f"k{k}"] = item
    return out


def audit(args: argparse.Namespace) -> dict[str, Any]:
    outer_sha = verify_outer_sha(args.bundle, args.sha256_file)
    with tempfile.TemporaryDirectory(prefix="s6_formal_audit_") as tmp:
        extracted_root = Path(tmp)
        safe_extract(args.bundle, extracted_root)
        internal_sha = verify_internal_sha(extracted_root)

        split_manifest_path = extracted_root / "results/s6_cost_aware_aux/Industrial_and_Scientific/s6_validation_split_manifest.json"
        manifest = read_json(split_manifest_path)
        seed = int(manifest["split_algorithm"]["seed"])
        source_valid_csv = ROOT / manifest["source_valid_csv"]
        valid_rows = read_valid_csv(source_valid_csv)
        if file_sha256(source_valid_csv) != manifest["source_valid_csv_sha256"]:
            raise ValueError("local source valid CSV hash does not match formal manifest")
        expected_by_split = source_rows_by_split(valid_rows, seed)

        row_index_path = ROOT / "data/Amazon/cs_embeddings/Industrial_and_Scientific/Industrial_and_Scientific.row_index.json"
        item_universe = set(read_json(row_index_path))

        exit_codes = {}
        for name in ("overall", *SPLITS):
            path = extracted_root / f"logs/s6_direct_sasrec/formal_v1/{name}_exit_code.txt"
            if name == "overall":
                path = extracted_root / "logs/s6_direct_sasrec/formal_v1/overall_exit_code.txt"
            exit_codes[name] = path.read_text(encoding="utf-8").strip()

        view_audits = [
            audit_candidate_view(extracted_root, split, k, expected_by_split[split], item_universe)
            for split in SPLITS
            for k in KS
        ]
        prefix = [audit_prefix_consistency(extracted_root, split) for split in SPLITS]
        cf_alignment = audit_cf_alignment(args.cf_candidates, expected_by_split)
        runtime = aggregate_runtime(view_audits)
        stability = split_stability(view_audits)
        recommendation = recommend_k(view_audits)

        safety_ok = all(row["ok"] for row in view_audits)
        prefix_ok = all(row["ok"] for row in prefix)
        exit_ok = all(value == "0" for value in exit_codes.values())
        evidence_ok = outer_sha["ok"] and internal_sha["ok"] and exit_ok
        cf_ok = cf_alignment["ok"]
        verdict = "GO_UNION_VALIDATION" if evidence_ok and safety_ok and prefix_ok and cf_ok else "NO_GO_UNION_VALIDATION"

        return {
            "schema": "s6_formal_direct_sasrec_audit.v1",
            "verdict": verdict,
            "evidence_integrity": {
                "outer_sha256": outer_sha,
                "internal_sha256": internal_sha,
                "exit_codes": exit_codes,
                "ok": evidence_ok,
            },
            "row_and_candidate_safety_ok": safety_ok,
            "prefix_consistency_ok": prefix_ok,
            "cf_alignment_ok": cf_ok,
            "view_audits": view_audits,
            "metric_table": build_metric_table(view_audits),
            "aggregate_metrics_by_k": aggregate_metrics_by_k(view_audits),
            "prefix_consistency": prefix,
            "split_stability": stability,
            "candidate_budget_recommendation": recommendation,
            "runtime_aggregation": runtime,
            "frozen_cf_alignment": cf_alignment,
            "validation_side_cost_evidence": {
                "direct_sasrec_runtime_available": True,
                "direct_runtime_aggregation": runtime,
                "sasrec_sid_qwen_validation_runtime_available": False,
                "cost_gate_status": "pending comparable frozen SASRec-SID Qwen validation runtime measurement",
            },
            "union_input_inventory": {
                "direct_sasrec_formal_v1_root": "results/s6_cost_aware_aux/Industrial_and_Scientific/direct_sasrec/<split>/formal_v1/k<K>/candidates.jsonl",
                "frozen_cf_full_valid_candidates": args.cf_candidates.as_posix(),
                "cf_alignment_policy": "filter frozen CF full-valid candidates by direct split row_index before merge; do not rerun CF generation",
                "splits": {split: {"rows": len(expected_by_split[split])} for split in SPLITS},
                "ks": list(KS),
            },
            "future_union_commands": [
                "python3 merge_direct_sasrec_candidates.py --cf-candidates <filtered_cf_split_candidates.jsonl> --direct-candidates results/s6_cost_aware_aux/Industrial_and_Scientific/direct_sasrec/<split>/formal_v1/k<K>/candidates.jsonl --output-jsonl results/s6_cost_aware_aux/Industrial_and_Scientific/union_cf_direct_sasrec/<split>/formal_v1/k<K>/candidates.jsonl --report results/s6_cost_aware_aux/Industrial_and_Scientific/union_cf_direct_sasrec/<split>/formal_v1/k<K>/union_report.json",
            ],
            "notes": [
                "Audit reads only validation artifacts and local validation CSV.",
                "No direct inference, union, ranker, training, or test access is performed.",
                "K20/K50 are audited as prefixes derived from the K100 model pass.",
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit S6 formal direct-SASRec validation bundle.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--sha256-file", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--cf-candidates", type=Path, default=DEFAULT_CF_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args)
    write_json(args.output, report)
    print(json.dumps({"verdict": report["verdict"], "output": args.output.as_posix()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
