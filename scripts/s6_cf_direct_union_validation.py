#!/usr/bin/env python3
"""Build and audit S6 CF + direct-SASRec validation union candidates."""

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

import build_s6_validation_split as split_manifest


ROOT = Path(__file__).resolve().parents[1]
CATEGORY = "Industrial_and_Scientific"
SPLITS = ("valid_fit", "valid_select", "valid_gate")
KS = (20, 50, 100)
FROZEN_CF_SHA256 = "eaef029b5679e525c06e51fabbac36bc6f029c9da9165d641634cbf945cd5281"
FORMAL_CHECKPOINT_SHA256 = "577f51a9a539b9cd6307eea16c04db303691e349c0f32e245e44b8710cee94bd"

DEFAULT_SPLIT_MANIFEST = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific/s6_validation_split_manifest.json"
DEFAULT_DIRECT_BUNDLE = ROOT / "incoming/s6_formal_direct_sasrec/s6_formal_direct_sasrec_formal_v1_bundle.tar.gz"
DEFAULT_DIRECT_SHA = DEFAULT_DIRECT_BUNDLE.with_suffix(DEFAULT_DIRECT_BUNDLE.suffix + ".sha256")
DEFAULT_CF_FULL = (
    ROOT
    / "incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036"
    / "results/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/formal/valid/full_valid"
    / "candidates/candidates.jsonl"
)
DEFAULT_CF_REPORT = DEFAULT_CF_FULL.with_name("candidate_report.json")
DEFAULT_SASREC_SID_FULL = (
    ROOT
    / "incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036"
    / "results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/formal/valid/full_valid"
    / "candidates/candidates.jsonl"
)
DEFAULT_SASREC_SID_REPORT = DEFAULT_SASREC_SID_FULL.with_name("candidate_report.json")
DEFAULT_OUTPUT_ROOT = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific"
DEFAULT_RUN_ID = "union_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    reject_existing(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    reject_existing(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def reject_existing(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        raise FileExistsError(f"refusing to overwrite existing S6-4 artifact: {path}")


def reject_test_path(path: Path) -> None:
    lowered = [part.lower() for part in path.parts]
    if "test" in lowered or "final_test" in lowered or path.name.lower() == "test.csv":
        raise ValueError(f"S6-4 refuses test paths: {path}")


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
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
        except (SyntaxError, ValueError):
            return [item.strip().strip("'\"") for item in text[1:-1].split(",") if item.strip()]
    return [text]


def parse_sha_file(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    parts = text.split()
    if len(parts) < 2:
        raise ValueError(f"invalid sha256 file: {path}")
    return parts[0], parts[-1].lstrip("*")


def verify_bundle_sha(bundle: Path, sha_file: Path) -> dict[str, Any]:
    expected, name = parse_sha_file(sha_file)
    actual = file_sha256(bundle)
    return {"bundle": bundle.as_posix(), "sha_file_name": name, "expected": expected, "actual": actual, "ok": expected == actual}


def safe_extract(bundle: Path, destination: Path) -> None:
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(destination, filter="data")


def load_valid_rows(split_manifest_path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    manifest = read_json(split_manifest_path)
    source_csv = ROOT / manifest["source_valid_csv"]
    reject_test_path(source_csv)
    with open(source_csv, "r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    if file_sha256(source_csv) != manifest["source_valid_csv_sha256"]:
        raise ValueError("source valid CSV hash mismatch")
    seed = int(manifest["split_algorithm"]["seed"])
    by_split: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    for idx, row in enumerate(rows):
        split = split_manifest.split_for_key(str(row["user_id"]), seed)
        copied = dict(row)
        copied["source_row_index"] = str(idx)
        by_split[split].append(copied)
    return manifest, by_split


def index_by_row(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        rid = row_id(row)
        if rid in indexed:
            raise ValueError(f"duplicate row id {rid}")
        indexed[rid] = row
    return indexed


def validate_row_alignment(candidate: dict[str, Any], expected: dict[str, Any], *, require_user: bool = False) -> None:
    rid = row_id(candidate)
    if rid != str(expected["source_row_index"]):
        raise ValueError(f"row id mismatch: {rid} != {expected['source_row_index']}")
    if str(candidate.get("target_item_id")) != str(expected.get("item_id")):
        raise ValueError(f"target mismatch for row {rid}")
    if normalize_history(candidate.get("history_item_id")) != normalize_history(expected.get("history_item_id")):
        raise ValueError(f"history mismatch for row {rid}")
    if require_user and str(candidate.get("user_id")) != str(expected.get("user_id")):
        raise ValueError(f"user mismatch for row {rid}")


def filter_cf_split(
    cf_full_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, str]],
    output_path: Path,
    report_path: Path,
    *,
    source_cf_hash: str,
    run_id: str,
) -> dict[str, Any]:
    cf_by_row = index_by_row(cf_full_rows)
    out_rows: list[dict[str, Any]] = []
    alignment_failures = 0
    for expected in expected_rows:
        rid = str(expected["source_row_index"])
        row = cf_by_row.get(rid)
        if row is None:
            raise ValueError(f"CF full-valid missing row {rid}")
        try:
            validate_row_alignment(row, expected, require_user=False)
        except ValueError:
            alignment_failures += 1
            raise
        out_rows.append(row)
    write_jsonl(output_path, out_rows)
    report = {
        "schema": "s6_cf_split_view_report.v1",
        "run_id": run_id,
        "source_cf_full_valid": DEFAULT_CF_FULL.as_posix(),
        "source_cf_sha256": source_cf_hash,
        "split_row_count": len(out_rows),
        "output": output_path.as_posix(),
        "output_sha256": file_sha256(output_path),
        "alignment_failures": alignment_failures,
        "row_order": "S6 split source-row order",
        "test_read": False,
    }
    write_json(report_path, report)
    return report


def detail_by_item(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("item_id")): item for item in row.get("candidate_details", []) if isinstance(item, dict)}


def merge_union_row(cf: dict[str, Any], direct: dict[str, Any]) -> dict[str, Any]:
    rid = row_id(cf)
    if row_id(direct) != rid:
        raise ValueError(f"row mismatch: cf={rid} direct={row_id(direct)}")
    if str(cf.get("target_item_id")) != str(direct.get("target_item_id")):
        raise ValueError(f"target mismatch for row {rid}")
    if normalize_history(cf.get("history_item_id")) != normalize_history(direct.get("history_item_id")):
        raise ValueError(f"history mismatch for row {rid}")
    cf_items = [str(item) for item in cf.get("candidate_item_ids", [])]
    direct_items = [str(item) for item in direct.get("candidate_item_ids", [])]
    cf_details = detail_by_item(cf)
    direct_details = detail_by_item(direct)
    out_items = cf_items + [item for item in direct_items if item not in set(cf_items)]
    details: list[dict[str, Any]] = []
    for union_rank, item in enumerate(out_items):
        cf_rank = cf_items.index(item) if item in cf_items else None
        direct_rank = direct_items.index(item) if item in direct_items else None
        detail = {
            "item_id": item,
            "union_rank_0_based": union_rank,
            "from_cf": cf_rank is not None,
            "from_sasrec_direct": direct_rank is not None,
            "overlap": cf_rank is not None and direct_rank is not None,
            "cf_rank_0_based": cf_rank,
            "cf_reciprocal_rank": None if cf_rank is None else 1.0 / float(cf_rank + 1),
            "sasrec_direct_rank_0_based": direct_rank,
            "sasrec_direct_reciprocal_rank": None if direct_rank is None else 1.0 / float(direct_rank + 1),
            "cf_detail": cf_details.get(item),
            "sasrec_direct_detail": direct_details.get(item),
        }
        details.append(detail)
    target = str(cf.get("target_item_id"))
    hit_rank = next((idx for idx, item in enumerate(out_items) if item == target), None)
    return {
        "row_index": rid,
        "target_item_id": target,
        "history_item_id": normalize_history(cf.get("history_item_id")),
        "candidate_item_ids": out_items,
        "candidate_details": details,
        "candidate_hit_rank_0_based": hit_rank,
        "candidate_pool_hit": hit_rank is not None,
        "target_candidate_source_types": [
            name
            for name, present in (
                ("cf", target in cf_items),
                ("sasrec_direct", target in direct_items),
            )
            if present
        ],
    }


def candidate_hit(row: dict[str, Any]) -> bool:
    target = str(row.get("target_item_id"))
    return target in [str(item) for item in row.get("candidate_item_ids", [])]


def merge_union_rows(cf_rows: list[dict[str, Any]], direct_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cf_by_row = index_by_row(cf_rows)
    direct_by_row = index_by_row(direct_rows)
    if set(cf_by_row) != set(direct_by_row):
        raise ValueError(f"row alignment mismatch: cf={len(cf_by_row)} direct={len(direct_by_row)}")
    return [merge_union_row(cf_by_row[key], direct_by_row[key]) for key in sorted(cf_by_row, key=lambda value: int(value))]


def union_metrics(cf_rows: list[dict[str, Any]], direct_rows: list[dict[str, Any]], union_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cf_by_row = index_by_row(cf_rows)
    direct_by_row = index_by_row(direct_rows)
    counts = [len(row["candidate_item_ids"]) for row in union_rows]
    cf_hits = direct_hits = union_hits = cf_only = direct_only = both = 0
    cf_preserved = 0
    overlap_candidates = 0
    added_candidates = 0
    duplicates_after_union = 0
    provenance_failures = 0
    for union in union_rows:
        rid = row_id(union)
        cf = cf_by_row[rid]
        direct = direct_by_row[rid]
        target = str(union["target_item_id"])
        cf_items = [str(item) for item in cf["candidate_item_ids"]]
        direct_items = [str(item) for item in direct["candidate_item_ids"]]
        union_items = [str(item) for item in union["candidate_item_ids"]]
        cf_hit = target in cf_items
        direct_hit = target in direct_items
        union_hit = target in union_items
        cf_hits += int(cf_hit)
        direct_hits += int(direct_hit)
        union_hits += int(union_hit)
        cf_only += int(cf_hit and not direct_hit)
        direct_only += int(direct_hit and not cf_hit)
        both += int(cf_hit and direct_hit)
        cf_preserved += int((not cf_hit) or union_hit)
        overlap_candidates += len(set(cf_items) & set(direct_items))
        added_candidates += len([item for item in direct_items if item not in set(cf_items)])
        duplicates_after_union += len(union_items) - len(set(union_items))
        for detail in union.get("candidate_details", []):
            if detail.get("from_cf") and detail.get("cf_reciprocal_rank") is None:
                provenance_failures += 1
            if detail.get("from_sasrec_direct") and detail.get("sasrec_direct_reciprocal_rank") is None:
                provenance_failures += 1
            if detail.get("from_sasrec_direct") and detail.get("sasrec_direct_detail") is None:
                provenance_failures += 1
    rows = len(union_rows)
    return {
        "num_rows": rows,
        "cf_target_count": cf_hits,
        "cf_target_rate": cf_hits / rows if rows else 0.0,
        "direct_target_count": direct_hits,
        "direct_target_rate": direct_hits / rows if rows else 0.0,
        "union_target_count": union_hits,
        "union_target_rate": union_hits / rows if rows else 0.0,
        "cf_only_target_count": cf_only,
        "direct_only_target_count": direct_only,
        "both_source_target_count": both,
        "added_targets_beyond_cf": union_hits - cf_hits,
        "cf_target_preservation_count": cf_preserved,
        "cf_target_preservation_rate": cf_preserved / rows if rows else 0.0,
        "union_size": {
            "min": min(counts) if counts else 0,
            "mean": sum(counts) / rows if rows else 0.0,
            "max": max(counts) if counts else 0,
        },
        "candidate_overlap_count": overlap_candidates,
        "candidate_overlap_rate_vs_direct": overlap_candidates / sum(len(row["candidate_item_ids"]) for row in direct_rows) if direct_rows else 0.0,
        "added_candidate_count": added_candidates,
        "candidate_noise_ratio": ((added_candidates - max(union_hits - cf_hits, 0)) / added_candidates) if added_candidates else 0.0,
        "invalid_candidate_count": 0,
        "duplicate_candidate_count_after_union": duplicates_after_union,
        "row_target_history_alignment_failures": 0,
        "candidate_provenance_failures": provenance_failures,
    }


def dual_qwen_metrics(cf_rows: list[dict[str, Any]], sasrec_sid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    union_rows = merge_dual_rows(cf_rows, sasrec_sid_rows)
    rows = len(union_rows)
    cf_hits = sum(candidate_hit(row) for row in cf_rows)
    sasrec_hits = sum(candidate_hit(row) for row in sasrec_sid_rows)
    union_hits = sum(candidate_hit(row) for row in union_rows)
    return {
        "num_rows": rows,
        "cf_target_count": cf_hits,
        "sasrec_sid_target_count": sasrec_hits,
        "dual_qwen_union_target_count": union_hits,
        "dual_qwen_auxiliary_uplift": union_hits - cf_hits,
        "dual_qwen_union_target_rate": union_hits / rows if rows else 0.0,
    }


def merge_dual_rows(cf_rows: list[dict[str, Any]], sasrec_sid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cf_by_row = index_by_row(cf_rows)
    sid_by_row = index_by_row(sasrec_sid_rows)
    if set(cf_by_row) != set(sid_by_row):
        raise ValueError("CF and SASRec-SID row sets differ")
    out: list[dict[str, Any]] = []
    for rid in sorted(cf_by_row, key=lambda value: int(value)):
        cf = cf_by_row[rid]
        sid = sid_by_row[rid]
        if str(cf["target_item_id"]) != str(sid["target_item_id"]):
            raise ValueError(f"dual-Qwen target mismatch for row {rid}")
        if normalize_history(cf.get("history_item_id")) != normalize_history(sid.get("history_item_id")):
            raise ValueError(f"dual-Qwen history mismatch for row {rid}")
        cf_items = [str(item) for item in cf["candidate_item_ids"]]
        sid_items = [str(item) for item in sid["candidate_item_ids"]]
        items = cf_items + [item for item in sid_items if item not in set(cf_items)]
        target = str(cf["target_item_id"])
        out.append({"row_index": rid, "target_item_id": target, "candidate_item_ids": items})
    return out


def filter_rows_by_expected(full_rows: list[dict[str, Any]], expected_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_row = index_by_row(full_rows)
    rows = []
    for expected in expected_rows:
        row = by_row[str(expected["source_row_index"])]
        validate_row_alignment(row, expected, require_user=False)
        rows.append(row)
    return rows


def select_k(metrics_by_split_k: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    candidates = []
    for k in KS:
        fit = metrics_by_split_k["valid_fit"][k]
        select = metrics_by_split_k["valid_select"][k]
        fit_retention = fit["uplift_retention"]
        select_retention = select["uplift_retention"]
        passes = fit_retention is not None and select_retention is not None and fit_retention >= 0.9 and select_retention >= 0.9
        candidates.append({"k": k, "valid_fit_retention": fit_retention, "valid_select_retention": select_retention, "passes": passes})
    selected = next((row["k"] for row in candidates if row["passes"]), None)
    return {
        "policy": "smallest K with candidate uplift retention >= 90% on valid_fit and valid_select",
        "evaluated": candidates,
        "selected_k": selected,
        "ok": selected is not None,
    }


def path_for_cf_split(output_root: Path, run_id: str, split: str) -> tuple[Path, Path]:
    base = output_root / "cf_split_views" / run_id / split
    return base / "candidates.jsonl", base / "candidate_report.json"


def path_for_union(output_root: Path, run_id: str, split: str, k: int) -> tuple[Path, Path]:
    base = output_root / "union" / run_id / split / f"k{k}"
    return base / "candidates.jsonl", base / "candidate_report.json"


def audit_direct_artifacts(direct_root: Path) -> dict[str, Any]:
    views: dict[str, Any] = {}
    failures = []
    for split in SPLITS:
        views[split] = {}
        for k in KS:
            base = direct_root / f"results/s6_cost_aware_aux/{CATEGORY}/direct_sasrec/{split}/formal_v1/k{k}"
            candidates = base / "candidates.jsonl"
            report = base / "candidate_report.json"
            if not candidates.is_file() or not report.is_file():
                failures.append(f"missing direct artifact {split}/k{k}")
                continue
            report_data = read_json(report)
            actual = file_sha256(candidates)
            expected = report_data.get("hashes", {}).get("candidates")
            ok = actual == expected and report_data.get("test_read") is False
            if report_data.get("hashes", {}).get("checkpoint") != FORMAL_CHECKPOINT_SHA256:
                ok = False
            views[split][str(k)] = {
                "candidates": candidates.as_posix(),
                "report": report.as_posix(),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "test_read": report_data.get("test_read"),
                "checkpoint_sha256": report_data.get("hashes", {}).get("checkpoint"),
                "ok": ok,
            }
            if not ok:
                failures.append(f"direct artifact audit failed {split}/k{k}")
    return {"views": views, "failures": failures, "ok": not failures}


def run_validation(args: argparse.Namespace, direct_root: Path) -> dict[str, Any]:
    for path in [args.split_manifest, args.cf_full_candidates, args.sasrec_sid_full_candidates]:
        reject_test_path(path)
    manifest, expected_by_split = load_valid_rows(args.split_manifest)
    cf_full_hash = file_sha256(args.cf_full_candidates)
    if cf_full_hash != FROZEN_CF_SHA256:
        raise ValueError(f"frozen CF SHA mismatch: {cf_full_hash}")
    cf_report = read_json(args.cf_candidate_report)
    sasrec_sid_report = read_json(args.sasrec_sid_candidate_report)
    if cf_report.get("test_read") is True or sasrec_sid_report.get("test_read") is True:
        raise ValueError("S4 candidate reports must not record test_read=true")

    direct_inventory = audit_direct_artifacts(direct_root)
    if not direct_inventory["ok"]:
        raise ValueError(f"direct artifact audit failed: {direct_inventory['failures']}")

    cf_full_rows = read_jsonl(args.cf_full_candidates)
    sasrec_sid_full_rows = read_jsonl(args.sasrec_sid_full_candidates)
    split_reports: dict[str, Any] = {}
    cf_split_rows: dict[str, list[dict[str, Any]]] = {}
    sasrec_sid_split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        out, report = path_for_cf_split(args.output_root, args.run_id, split)
        cf_split_rows[split] = filter_rows_by_expected(cf_full_rows, expected_by_split[split])
        sasrec_sid_split_rows[split] = filter_rows_by_expected(sasrec_sid_full_rows, expected_by_split[split])
        split_reports[split] = filter_cf_split(
            cf_full_rows,
            expected_by_split[split],
            out,
            report,
            source_cf_hash=cf_full_hash,
            run_id=args.run_id,
        )

    union_reports: dict[str, dict[str, Any]] = {split: {} for split in SPLITS}
    retention_inputs: dict[str, dict[int, dict[str, Any]]] = {split: {} for split in SPLITS}
    for split in SPLITS:
        dual = dual_qwen_metrics(cf_split_rows[split], sasrec_sid_split_rows[split])
        for k in KS:
            direct_path = direct_root / f"results/s6_cost_aware_aux/{CATEGORY}/direct_sasrec/{split}/formal_v1/k{k}/candidates.jsonl"
            direct_rows = read_jsonl(direct_path)
            union_rows = merge_union_rows(cf_split_rows[split], direct_rows)
            union_path, report_path = path_for_union(args.output_root, args.run_id, split, k)
            write_jsonl(union_path, union_rows)
            metrics = union_metrics(cf_split_rows[split], direct_rows, union_rows)
            dual_uplift = dual["dual_qwen_auxiliary_uplift"]
            direct_uplift = metrics["added_targets_beyond_cf"]
            retention = None if dual_uplift == 0 else direct_uplift / dual_uplift
            metrics.update(
                {
                    "direct_auxiliary_uplift": direct_uplift,
                    "dual_qwen_auxiliary_uplift": dual_uplift,
                    "candidate_uplift_retention": retention,
                    "candidate_gate_pass": retention is not None and retention >= 0.9,
                    "dual_qwen_comparison": dual,
                }
            )
            report = {
                "schema": "s6_cf_direct_sasrec_union_candidate_report.v1",
                "run_id": args.run_id,
                "split": split,
                "k": k,
                "inputs": {
                    "cf_split_candidates": path_for_cf_split(args.output_root, args.run_id, split)[0].as_posix(),
                    "direct_candidates": direct_path.as_posix(),
                },
                "outputs": {"candidates": union_path.as_posix()},
                "output_sha256": file_sha256(union_path),
                "metrics": metrics,
                "ordering_policy": "CF candidates in original order followed by direct-only SASRec candidates in direct rank order",
                "test_read": False,
            }
            write_json(report_path, report)
            union_reports[split][str(k)] = report
            retention_inputs[split][k] = {
                "uplift_retention": retention,
                "direct_auxiliary_uplift": direct_uplift,
                "dual_qwen_auxiliary_uplift": dual_uplift,
                "candidate_gate_pass": report["metrics"]["candidate_gate_pass"],
            }

    k_selection = select_k(retention_inputs)
    selected_k = k_selection["selected_k"]
    gate_confirmation = None
    if selected_k is not None:
        gate_confirmation = {
            "split": "valid_gate",
            "selected_k": selected_k,
            "candidate_gate_pass": retention_inputs["valid_gate"][selected_k]["candidate_gate_pass"],
            "uplift_retention": retention_inputs["valid_gate"][selected_k]["uplift_retention"],
            "note": "valid_gate inspected only after K selection; it does not revise K",
        }

    all_union = [union_reports[split][str(k)]["metrics"] for split in SPLITS for k in KS]
    safety_ok = all(
        metrics["cf_target_preservation_rate"] == 1.0
        and metrics["duplicate_candidate_count_after_union"] == 0
        and metrics["candidate_provenance_failures"] == 0
        and metrics["row_target_history_alignment_failures"] == 0
        for metrics in all_union
    )
    direct_meaningful = any(metrics["direct_only_target_count"] > 0 for metrics in all_union)
    verdict = (
        "GO_FROZEN_RANKER_VALIDATION"
        if safety_ok and direct_meaningful and k_selection["ok"] and gate_confirmation and gate_confirmation["candidate_gate_pass"]
        else "REVISE_UNION_STAGE"
    )
    return {
        "schema": "s6_cf_direct_sasrec_union_validation_report.v1",
        "verdict": verdict,
        "run_id": args.run_id,
        "evidence_integrity": {
            "split_manifest": args.split_manifest.as_posix(),
            "cf_full_candidates": args.cf_full_candidates.as_posix(),
            "cf_full_sha256": cf_full_hash,
            "cf_full_sha256_matches_frozen": cf_full_hash == FROZEN_CF_SHA256,
            "cf_candidate_report_test_read": cf_report.get("test_read"),
            "sasrec_sid_candidate_report_test_read": sasrec_sid_report.get("test_read"),
            "direct_artifacts": direct_inventory,
            "test_read": False,
        },
        "cf_split_views": split_reports,
        "union_reports": union_reports,
        "k_selection": k_selection,
        "valid_gate_confirmation": gate_confirmation,
        "cost_evidence": {
            "direct_model_inference_seconds": 24.967018,
            "direct_formal_wall_seconds": 34.233973,
            "direct_peak_cuda_allocated_bytes": 13118464,
            "sasrec_sid_qwen_comparable_validation_runtime": None,
            "cost_gate_status": "pending comparable validation-side SASRec-SID Qwen runtime evidence",
        },
        "dual_qwen_evidence": {
            "source": "frozen S4 formal full-valid CF and SASRec-SID candidate artifacts",
            "split_compatible": True,
            "test_read": False,
        },
        "next_gate": "frozen-ranker validation on selected K" if verdict == "GO_FROZEN_RANKER_VALIDATION" else "revise union stage",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S6 CF + direct-SASRec union validation.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--direct-bundle", type=Path, default=DEFAULT_DIRECT_BUNDLE)
    parser.add_argument("--direct-sha256", type=Path, default=DEFAULT_DIRECT_SHA)
    parser.add_argument("--cf-full-candidates", type=Path, default=DEFAULT_CF_FULL)
    parser.add_argument("--cf-candidate-report", type=Path, default=DEFAULT_CF_REPORT)
    parser.add_argument("--sasrec-sid-full-candidates", type=Path, default=DEFAULT_SASREC_SID_FULL)
    parser.add_argument("--sasrec-sid-candidate-report", type=Path, default=DEFAULT_SASREC_SID_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "s6_4_union_validation_report.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "run_id": args.run_id,
                    "split_manifest": args.split_manifest.as_posix(),
                    "direct_bundle": args.direct_bundle.as_posix(),
                    "cf_full_candidates": args.cf_full_candidates.as_posix(),
                    "sasrec_sid_full_candidates": args.sasrec_sid_full_candidates.as_posix(),
                    "output_root": args.output_root.as_posix(),
                    "report": args.report.as_posix(),
                    "test_read": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    bundle_sha = verify_bundle_sha(args.direct_bundle, args.direct_sha256)
    if not bundle_sha["ok"]:
        raise ValueError(f"direct bundle SHA mismatch: {bundle_sha}")
    with tempfile.TemporaryDirectory(prefix="s6_union_direct_") as tmp:
        direct_root = Path(tmp)
        safe_extract(args.direct_bundle, direct_root)
        report = run_validation(args, direct_root)
        report["evidence_integrity"]["direct_bundle_sha256"] = bundle_sha
        write_json(args.report, report)
        print(json.dumps({"verdict": report["verdict"], "report": args.report.as_posix()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
