#!/usr/bin/env python3
"""Audit S6 direct SASRec smoke evidence without rerunning inference."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_s6_validation_split as split_manifest  # noqa: E402

DEFAULT_BUNDLE = ROOT / "incoming/s6_direct_sasrec_smoke/s6_direct_sasrec_valid_fit_k20_smoke_bundle.tar.gz"
DEFAULT_BUNDLE_SHA = DEFAULT_BUNDLE.with_suffix(DEFAULT_BUNDLE.suffix + ".sha256")
DEFAULT_OUTPUT = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific/s6_2_smoke_audit_report.json"
SOURCE_VALID = ROOT / "data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv"
ROW_INDEX = ROOT / "data/Amazon/behavior_embeddings/sasrec/Industrial_and_Scientific/formal_v3_finite_maskfix_seed42/Industrial_and_Scientific.row_index.json"
EXPECTED_CHECKPOINT_SHA = "577f51a9a539b9cd6307eea16c04db303691e349c0f32e245e44b8710cee94bd"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


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


def rank_of(target: str, items: list[str]) -> int | None:
    for idx, item in enumerate(items):
        if str(item) == str(target):
            return idx
    return None


def metric_table(ranks: list[int | None], ks: list[int]) -> dict[str, Any]:
    out = {"num_samples": len(ranks)}
    for k in ks:
        hits = 0
        ndcg = 0.0
        for rank in ranks:
            if rank is not None and rank < k:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
        out[f"hr@{k}"] = 0.0 if not ranks else hits / len(ranks)
        out[f"ndcg@{k}"] = 0.0 if not ranks else ndcg / len(ranks)
    out["mrr"] = 0.0 if not ranks else sum(0.0 if r is None else 1.0 / (r + 1) for r in ranks) / len(ranks)
    out["target_in_pool_count"] = sum(r is not None for r in ranks)
    out["target_in_pool_rate"] = 0.0 if not ranks else out["target_in_pool_count"] / len(ranks)
    return out


def parse_sha_file(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        out[rel.strip()] = digest.strip()
    return out


def verify_bundle(bundle: Path, sha_file: Path) -> dict[str, Any]:
    expected = parse_sha_file(sha_file)
    actual = file_sha256(bundle)
    bundle_name = bundle.name
    expected_hash = expected.get(bundle_name)
    if expected_hash != actual:
        raise ValueError(f"bundle sha mismatch: expected {expected_hash}, actual {actual}")
    return {"bundle": bundle.as_posix(), "sha256": actual, "verified": True}


def audit_extracted(root: Path) -> dict[str, Any]:
    base = root / "results/s6_cost_aware_aux/Industrial_and_Scientific"
    direct = base / "direct_sasrec/valid_fit/k20"
    evidence = base / "s6_smoke_evidence/valid_fit_k20"
    required = [
        direct / "candidates.jsonl",
        direct / "candidate_features.jsonl",
        direct / "candidate_report.json",
        evidence / "execution.log",
        evidence / "exit_code.txt",
        evidence / "smoke_output_files.sha256",
        root / "configs/s6_cost_aware_aux/dev_config.json",
        base / "s6_checkpoint_compatibility_report.json",
        base / "s6_validation_split_manifest.json",
    ]
    missing = [path.as_posix() for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(missing)
    expected_hashes = parse_sha_file(evidence / "smoke_output_files.sha256")
    hash_results = {}
    for rel, expected in expected_hashes.items():
        actual = file_sha256(root / rel)
        if actual != expected:
            raise ValueError(f"hash mismatch for {rel}: {actual} != {expected}")
        hash_results[rel] = actual
    exit_code = (evidence / "exit_code.txt").read_text(encoding="utf-8").strip()
    if exit_code != "0":
        raise ValueError(f"non-zero exit code: {exit_code}")

    report = json.loads((direct / "candidate_report.json").read_text(encoding="utf-8"))
    checkpoint_report = json.loads((base / "s6_checkpoint_compatibility_report.json").read_text(encoding="utf-8"))
    split = json.loads((base / "s6_validation_split_manifest.json").read_text(encoding="utf-8"))
    if report.get("hashes", {}).get("checkpoint") != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("checkpoint hash mismatch in candidate report")
    if checkpoint_report.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("checkpoint hash mismatch in compatibility report")
    if report.get("test_read") is not False:
        raise ValueError("candidate report does not record test_read=false")
    all_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in required if path.suffix != ".pt")
    if "/test/" in all_text.lower() or "test.csv" in all_text.lower():
        raise ValueError("test path marker found in smoke evidence")

    source_rows = read_csv_rows(SOURCE_VALID)
    valid_fit = []
    for idx, row in enumerate(source_rows):
        if split_manifest.split_for_key(row["user_id"], 42) == "valid_fit":
            copied = dict(row)
            copied["source_row_index"] = str(idx)
            valid_fit.append(copied)
    item_universe = set(json.loads(ROW_INDEX.read_text(encoding="utf-8")))
    candidates = read_jsonl(direct / "candidates.jsonl")
    features = read_jsonl(direct / "candidate_features.jsonl")
    if len(candidates) != 2660 or len(features) != 2660 or len(valid_fit) != 2660:
        raise ValueError(f"row count mismatch: candidates={len(candidates)} features={len(features)} valid_fit={len(valid_fit)}")
    ranks = []
    duplicate_rows = 0
    invalid_items = 0
    nonfinite = 0
    monotonic_violations = 0
    rank_violations = 0
    for idx, (row, feat, expected_row) in enumerate(zip(candidates, features, valid_fit)):
        if str(row.get("row_index")) != expected_row["source_row_index"]:
            raise ValueError(f"row_index mismatch at {idx}")
        if str(row.get("user_id")) != expected_row["user_id"]:
            raise ValueError(f"user_id mismatch at {idx}")
        if str(row.get("target_item_id")) != expected_row["item_id"]:
            raise ValueError(f"target mismatch at {idx}")
        expected_history = [str(item) for item in parse_list(expected_row.get("history_item_id", ""))]
        if [str(item) for item in row.get("history_item_id", [])] != expected_history:
            raise ValueError(f"history mismatch at {idx}")
        items = [str(item) for item in row.get("candidate_item_ids", [])]
        if len(items) != 20:
            raise ValueError(f"candidate count mismatch at {idx}")
        if len(set(items)) != len(items):
            duplicate_rows += 1
        invalid_items += sum(1 for item in items if item not in item_universe)
        details = row.get("candidate_details", [])
        scores = []
        ranks_seen = []
        for detail_idx, detail in enumerate(details):
            rank = detail.get("sasrec_direct_rank")
            ranks_seen.append(rank)
            if rank != detail_idx + 1:
                rank_violations += 1
            for key in [
                "sasrec_direct_score",
                "sasrec_direct_zscore",
                "sasrec_direct_top1_margin",
                "sasrec_direct_score_minus_topk_mean",
                "sasrec_direct_reciprocal_rank",
            ]:
                value = detail.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    nonfinite += 1
            scores.append(float(detail["sasrec_direct_score"]))
        if ranks_seen != list(range(1, 21)):
            rank_violations += 1
        if any(left < right for left, right in zip(scores, scores[1:])):
            monotonic_violations += 1
        if feat.get("row_index") != row.get("row_index"):
            raise ValueError(f"feature row alignment mismatch at {idx}")
        ranks.append(rank_of(str(row.get("target_item_id")), items))
    recomputed = metric_table(ranks, [1, 5, 10, 20])
    for key, value in recomputed.items():
        if isinstance(value, float):
            if abs(value - float(report.get(key, -999))) > 1e-12:
                raise ValueError(f"metric mismatch {key}: {value} vs {report.get(key)}")
        elif value != report.get(key):
            raise ValueError(f"metric mismatch {key}: {value} vs {report.get(key)}")
    return {
        "schema": "s6_2_smoke_audit_report.v1",
        "verdict": "GO_SMOKE_AUDIT",
        "exit_code": int(exit_code),
        "required_files_present": True,
        "smoke_output_hashes": hash_results,
        "checkpoint_sha256": report["hashes"]["checkpoint"],
        "test_read": False,
        "split": "valid_fit",
        "k": 20,
        "row_count": len(candidates),
        "feature_row_count": len(features),
        "candidate_count": report["candidate_count"],
        "duplicate_candidate_rows": duplicate_rows,
        "invalid_item_count_recomputed": invalid_items,
        "nonfinite_feature_count": nonfinite,
        "rank_violations": rank_violations,
        "monotonic_score_violations": monotonic_violations,
        "metrics_recomputed": recomputed,
        "metrics_reported": {key: report[key] for key in ["hr@1", "hr@5", "hr@10", "hr@20", "ndcg@1", "ndcg@5", "ndcg@10", "ndcg@20", "mrr", "target_in_pool_count", "target_in_pool_rate"]},
        "runtime": report.get("runtime", {}),
    }


def audit_bundle(bundle: Path, sha_file: Path) -> dict[str, Any]:
    integrity = verify_bundle(bundle, sha_file)
    with tempfile.TemporaryDirectory(prefix="s6_smoke_audit_") as tmp:
        with tarfile.open(bundle, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        report = audit_extracted(Path(tmp))
    report["bundle_integrity"] = integrity
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit S6 direct SASRec smoke bundle.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--sha256-file", type=Path, default=DEFAULT_BUNDLE_SHA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_bundle(args.bundle, args.sha256_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "row_count": report["row_count"]}, indent=2))


if __name__ == "__main__":
    main()
