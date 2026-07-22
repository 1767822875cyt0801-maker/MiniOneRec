#!/usr/bin/env python3
"""Diagnose P2 rerank headroom from fixed Stage 7 validation artifacts.

This script is CPU-only analysis. It reads the P1-valid Text/CF candidates,
dual-fusion candidates, and heuristic rerank output, then writes compact
diagnostics for deciding whether learned reranking is worth doing next.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/stage7_validation_protocol/valid/Industrial_and_Scientific")
DEFAULT_KS = [1, 5, 10, 20, 50, 100]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Stage 7 P2 rerank headroom.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--split", choices=["valid"], default="valid")
    parser.add_argument("--candidate-mode", choices=["exact"], default="exact")
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--gap-threshold", type=float, default=0.002)
    parser.add_argument("--fit-ratio", type=float, default=0.7)
    parser.add_argument("--split-seed", type=int, default=20260706)
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
                    raise ValueError(f"Expected object rows in {path}, got {type(value).__name__}")
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
        sample_id = normalize_id(row.get("row_index", row.get("sample_id")), idx)
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample id in {label}: {sample_id}")
        indexed[sample_id] = row
    return indexed


def list_field(row: dict[str, Any], field: str) -> list[str]:
    values = row.get(field, [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


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


def ndcg_at_ranks(ranks: list[int | None], k: int) -> float:
    import math

    total = 0.0
    for rank in ranks:
        if rank is not None and rank < k:
            total += 1.0 / math.log2(rank + 2)
    return rate(total, len(ranks))


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "not_in_candidates"
    one_based = rank + 1
    if one_based <= 5:
        return "1-5"
    if one_based <= 10:
        return "6-10"
    if one_based <= 20:
        return "11-20"
    if one_based <= 50:
        return "21-50"
    if one_based <= 100:
        return "51-100"
    return ">100"


def flatten(prefix: str, value: Any, rows: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flatten(name, child, rows)
    elif isinstance(value, list):
        rows.append({"metric": prefix, "value": json.dumps(value, ensure_ascii=False, sort_keys=True)})
    else:
        rows.append({"metric": prefix, "value": value})


def assert_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def no_test_path(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    return "test" not in lowered_parts and path.name.lower() != "test.csv"


def metric_close(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) <= EPS
    except (TypeError, ValueError):
        return False


def make_source_coverage_rows(records: list[dict[str, Any]], ks: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(records)
    scopes: list[tuple[str, int | None]] = [("full", None)] + [(f"@{k}", k) for k in ks]
    for scope, k in scopes:
        counts = Counter()
        for record in records:
            text_hit = record["text_rank"] is not None if k is None else hit_at(record["text_rank"], k)
            cf_hit = record["cf_rank"] is not None if k is None else hit_at(record["cf_rank"], k)
            union_hit = text_hit or cf_hit
            both_hit = text_hit and cf_hit
            text_only = text_hit and not cf_hit
            cf_only = cf_hit and not text_hit
            neither = not union_hit
            counts["text_hit"] += int(text_hit)
            counts["cf_hit"] += int(cf_hit)
            counts["union_hit"] += int(union_hit)
            counts["intersection_hit"] += int(both_hit)
            counts["text_only"] += int(text_only)
            counts["cf_only"] += int(cf_only)
            counts["both"] += int(both_hit)
            counts["neither"] += int(neither)
        row = {"scope": scope, "num_samples": n}
        for key in [
            "text_hit",
            "cf_hit",
            "union_hit",
            "intersection_hit",
            "text_only",
            "cf_only",
            "both",
            "neither",
        ]:
            row[f"{key}_count"] = counts[key]
            row[f"{key}_rate"] = rate(counts[key], n)
        rows.append(row)
    return rows


def make_rank_distribution_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = ["1-5", "6-10", "11-20", "21-50", "51-100", ">100", "not_in_candidates"]
    rank_sources = [
        ("fusion_rrf", "fusion_rank"),
        ("heuristic_rerank", "heuristic_rank"),
    ]
    rows: list[dict[str, Any]] = []
    n = len(records)
    for source_name, field in rank_sources:
        counts = Counter(rank_bucket(record[field]) for record in records)
        for bucket in buckets:
            rows.append(
                {
                    "rank_source": source_name,
                    "bucket": bucket,
                    "count": counts[bucket],
                    "rate": rate(counts[bucket], n),
                }
            )
    return rows


def make_topk_metrics(records: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    n = len(records)
    fusion_ranks = [record["fusion_rank"] for record in records]
    heuristic_ranks = [record["heuristic_rank"] for record in records]
    text_ranks = [record["text_rank"] for record in records]
    cf_ranks = [record["cf_rank"] for record in records]
    for k in ks:
        text_hits = sum(hit_at(rank, k) for rank in text_ranks)
        cf_hits = sum(hit_at(rank, k) for rank in cf_ranks)
        oracle_union_hits = sum(hit_at(record["text_rank"], k) or hit_at(record["cf_rank"], k) for record in records)
        fusion_hits = sum(hit_at(rank, k) for rank in fusion_ranks)
        heuristic_hits = sum(hit_at(rank, k) for rank in heuristic_ranks)
        recoverable_missed = sum(
            (hit_at(record["text_rank"], k) or hit_at(record["cf_rank"], k))
            and not hit_at(record["heuristic_rank"], k)
            for record in records
        )
        lost_after_rerank = sum(hit_at(record["fusion_rank"], k) and not hit_at(record["heuristic_rank"], k) for record in records)
        gained_after_rerank = sum(not hit_at(record["fusion_rank"], k) and hit_at(record["heuristic_rank"], k) for record in records)
        metrics[f"text_hr@{k}"] = rate(text_hits, n)
        metrics[f"cf_hr@{k}"] = rate(cf_hits, n)
        metrics[f"oracle_union_hr@{k}"] = rate(oracle_union_hits, n)
        metrics[f"fusion_rrf_hr@{k}"] = rate(fusion_hits, n)
        metrics[f"heuristic_rerank_hr@{k}"] = rate(heuristic_hits, n)
        metrics[f"fusion_rrf_ndcg@{k}"] = ndcg_at_ranks(fusion_ranks, k)
        metrics[f"heuristic_rerank_ndcg@{k}"] = ndcg_at_ranks(heuristic_ranks, k)
        metrics[f"recoverable_gap_vs_heuristic@{k}"] = rate(oracle_union_hits, n) - rate(heuristic_hits, n)
        metrics[f"fusion_gap_vs_heuristic@{k}"] = rate(fusion_hits, n) - rate(heuristic_hits, n)
        metrics[f"recoverable_missed_count@{k}"] = recoverable_missed
        metrics[f"lost_after_rerank_count@{k}"] = lost_after_rerank
        metrics[f"gained_after_rerank_count@{k}"] = gained_after_rerank
    full_union_hits = sum(record["text_rank"] is not None or record["cf_rank"] is not None for record in records)
    metrics["full_union_candidate_coverage"] = rate(full_union_hits, n)
    for k in ks:
        metrics[f"full_candidate_ceiling_gap_vs_heuristic@{k}"] = (
            metrics["full_union_candidate_coverage"] - metrics[f"heuristic_rerank_hr@{k}"]
        )
    return metrics


def make_failure_rows(records: list[dict[str, Any]], k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = Counter()
    failure_rows: list[dict[str, Any]] = []
    for record in records:
        if record["fusion_rank"] is None:
            failure_stage = "gt_not_in_union"
        elif record["fusion_rank"] >= k:
            failure_stage = f"gt_in_union_rank_gt_{k}"
        else:
            failure_stage = f"gt_in_top_{k}"
        counts[failure_stage] += 1

        heuristic_hit = hit_at(record["heuristic_rank"], k)
        if not heuristic_hit:
            row = {
                "row_index": record["sample_id"],
                "target_item_id": record["target_item_id"],
                "failure_stage": failure_stage,
                "text_rank_1_based": None if record["text_rank"] is None else record["text_rank"] + 1,
                "cf_rank_1_based": None if record["cf_rank"] is None else record["cf_rank"] + 1,
                "fusion_rank_1_based": None if record["fusion_rank"] is None else record["fusion_rank"] + 1,
                "heuristic_rank_1_based": None if record["heuristic_rank"] is None else record["heuristic_rank"] + 1,
                "in_text": record["text_rank"] is not None,
                "in_cf": record["cf_rank"] is not None,
                "in_union": record["fusion_rank"] is not None,
                "num_text_candidates": record["num_text_candidates"],
                "num_cf_candidates": record["num_cf_candidates"],
                "num_union_candidates": record["num_union_candidates"],
            }
            failure_rows.append(row)
    summary = {
        "k": k,
        "gt_not_in_union_count": counts["gt_not_in_union"],
        f"gt_in_union_rank_gt_{k}_count": counts[f"gt_in_union_rank_gt_{k}"],
        f"gt_in_top_{k}_count": counts[f"gt_in_top_{k}"],
    }
    total = len(records)
    for key, value in list(summary.items()):
        if key.endswith("_count"):
            summary[key.replace("_count", "_rate")] = rate(int(value), total)
    summary[f"heuristic_miss@{k}_rows_written"] = len(failure_rows)
    return summary, failure_rows


def make_markdown(summary: dict[str, Any], coverage_rows: list[dict[str, Any]], recommendation: dict[str, Any]) -> str:
    topk = summary["topk_metrics"]
    coverage_by_scope = {row["scope"]: row for row in coverage_rows}
    full = coverage_by_scope["full"]
    at20 = coverage_by_scope.get("@20", {})
    lines = [
        "# P2-0 Rerank Headroom Diagnosis",
        "",
        f"- split: `{summary['split']}`",
        f"- category: `{summary['category']}`",
        f"- candidate_mode: `{summary['candidate_mode']}`",
        f"- num_samples: `{summary['num_samples']}`",
        f"- overall_ok: `{summary['overall_ok']}`",
        "",
        "## Candidate Coverage",
        "",
        "| Scope | Text | CF | Union | Both | Text only | CF only | Neither |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Full | {full['text_hit_rate']:.6f} | {full['cf_hit_rate']:.6f} | "
            f"{full['union_hit_rate']:.6f} | {full['both_rate']:.6f} | "
            f"{full['text_only_rate']:.6f} | {full['cf_only_rate']:.6f} | {full['neither_rate']:.6f} |"
        ),
    ]
    if at20:
        lines.append(
            f"| @20 | {at20['text_hit_rate']:.6f} | {at20['cf_hit_rate']:.6f} | "
            f"{at20['union_hit_rate']:.6f} | {at20['both_rate']:.6f} | "
            f"{at20['text_only_rate']:.6f} | {at20['cf_only_rate']:.6f} | {at20['neither_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Ranking Headroom",
            "",
            "| K | Oracle union HR | Fusion RRF HR | Heuristic HR | Recoverable gap |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for k in [10, 20, 50]:
        if f"oracle_union_hr@{k}" not in topk:
            continue
        lines.append(
            f"| {k} | {topk[f'oracle_union_hr@{k}']:.6f} | "
            f"{topk[f'fusion_rrf_hr@{k}']:.6f} | "
            f"{topk[f'heuristic_rerank_hr@{k}']:.6f} | "
            f"{topk[f'recoverable_gap_vs_heuristic@{k}']:.6f} |"
        )
    failure = summary["ranking_failure@20"]
    lines.extend(
        [
            "",
            "## Ranking Failures At 20",
            "",
            f"- GT not in union: `{failure['gt_not_in_union_count']}`",
            f"- GT in union but fusion rank > 20: `{failure['gt_in_union_rank_gt_20_count']}`",
            f"- GT in fusion Top-20: `{failure['gt_in_top_20_count']}`",
            f"- Heuristic miss rows written: `{failure['heuristic_miss@20_rows_written']}`",
            "",
            "## Recommendation",
            "",
            f"- decision: `{recommendation['decision']}`",
            f"- reason: `{recommendation['reason']}`",
            f"- next_step: `{recommendation['next_step']}`",
            "",
        ]
    )
    return "\n".join(lines)


def make_recommendation(summary: dict[str, Any], gap_threshold: float, fit_ratio: float, split_seed: int) -> dict[str, Any]:
    n = int(summary["num_samples"])
    fit_count = int(round(n * fit_ratio))
    select_count = n - fit_count
    topk = summary["topk_metrics"]
    gap20 = float(topk.get("recoverable_gap_vs_heuristic@20", 0.0))
    coverage20 = summary["source_coverage_at_20"]
    has_complementarity = coverage20["text_only_count"] > 0 and coverage20["cf_only_count"] > 0
    if gap20 >= gap_threshold and has_complementarity:
        decision = "continue_to_p2_1"
        reason = (
            f"recoverable_gap@20={gap20:.6f} >= {gap_threshold:.6f}, "
            "and both Text-only and CF-only @20 hits exist."
        )
        next_step = "Create a fixed valid_fit/valid_select manifest, then train a minimal source-aware learned ranker."
    else:
        decision = "consider_p3_before_learned_rerank"
        reason = (
            f"recoverable_gap@20={gap20:.6f}; either the gap is below {gap_threshold:.6f} "
            "or source complementarity is weak."
        )
        next_step = "Prioritize behavior representation upgrade before spending more effort on learned rerank."
    return {
        "decision": decision,
        "reason": reason,
        "next_step": next_step,
        "valid_fit_count": fit_count,
        "valid_select_count": select_count,
        "fit_ratio": fit_ratio,
        "split_seed": split_seed,
        "split_policy": "stable hash over row_index; do not access test",
    }


def make_protocol_markdown(recommendation: dict[str, Any], summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# P2 Data Protocol Recommendation",
            "",
            f"- decision: `{recommendation['decision']}`",
            f"- reason: `{recommendation['reason']}`",
            f"- next_step: `{recommendation['next_step']}`",
            "",
            "## Proposed P2-1 Split",
            "",
            f"- source split: `{summary['split']}` only",
            f"- num_samples: `{summary['num_samples']}`",
            f"- valid_fit_count: `{recommendation['valid_fit_count']}`",
            f"- valid_select_count: `{recommendation['valid_select_count']}`",
            f"- split_seed: `{recommendation['split_seed']}`",
            "- split key: `row_index`",
            "- policy: train/select reranker variants on valid_fit, choose on valid_select, never inspect test",
            "",
            "## Minimal P2-2 Feature Set",
            "",
            "- `text_present`, `cf_present`, `both_sources`",
            "- `text_rank`, `cf_rank`, reciprocal ranks",
            "- `source_count`, `fusion_score`, `fusion_rank`",
            "- no target-label leakage in inference features",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = args.out_dir or (root / "p2_headroom")
    ks = sorted(set(k for k in args.ks if k > 0))
    if 20 not in ks:
        ks.append(20)
        ks = sorted(ks)

    text_candidates_path = root / "text/candidates/candidates.jsonl"
    cf_candidates_path = root / "cf/candidates/candidates.jsonl"
    fusion_jsonl_path = root / "fusion/dual_fused_candidates.jsonl"
    rerank_jsonl_path = root / "rerank/reranked_candidates.jsonl"
    fusion_report_path = root / "fusion/dual_fusion_report.json"
    rerank_report_path = root / "rerank/rerank_report.json"
    metrics_summary_path = root / "summary/metrics_summary.json"
    consistency_report_path = root / "summary/consistency_report.json"

    input_paths = [
        text_candidates_path,
        cf_candidates_path,
        fusion_jsonl_path,
        rerank_jsonl_path,
        fusion_report_path,
        rerank_report_path,
        metrics_summary_path,
        consistency_report_path,
    ]
    for path in input_paths:
        assert_file(path)

    text_rows = index_rows(read_jsonl(text_candidates_path), "text candidates")
    cf_rows = index_rows(read_jsonl(cf_candidates_path), "cf candidates")
    fusion_rows = index_rows(read_jsonl(fusion_jsonl_path), "fusion candidates")
    rerank_rows = index_rows(read_jsonl(rerank_jsonl_path), "rerank candidates")
    fusion_report = read_json(fusion_report_path)
    rerank_report = read_json(rerank_report_path)
    metrics_summary = read_json(metrics_summary_path)
    consistency_report = read_json(consistency_report_path)

    id_sets = {
        "text": set(text_rows),
        "cf": set(cf_rows),
        "fusion": set(fusion_rows),
        "rerank": set(rerank_rows),
    }
    all_ids = sorted(set().union(*id_sets.values()), key=item_sort_key)
    if len({frozenset(values) for values in id_sets.values()}) != 1:
        counts = {name: len(values) for name, values in id_sets.items()}
        raise ValueError(f"Input row ids are not aligned: {counts}")

    records: list[dict[str, Any]] = []
    target_mismatches: list[dict[str, Any]] = []
    for sample_id in all_ids:
        rows = {
            "text": text_rows[sample_id],
            "cf": cf_rows[sample_id],
            "fusion": fusion_rows[sample_id],
            "rerank": rerank_rows[sample_id],
        }
        targets = {
            name: str(row.get("target_item_id", "")).strip()
            for name, row in rows.items()
            if str(row.get("target_item_id", "")).strip()
        }
        unique_targets = set(targets.values())
        if len(unique_targets) != 1:
            target_mismatches.append({"sample_id": sample_id, "targets": targets})
        target_item_id = next(iter(unique_targets), "")

        text_items = list_field(rows["text"], "candidate_item_ids")
        cf_items = list_field(rows["cf"], "candidate_item_ids")
        fusion_items = list_field(rows["fusion"], "candidate_item_ids")
        reranked_items = list_field(rows["rerank"], "reranked_item_ids")
        records.append(
            {
                "sample_id": sample_id,
                "target_item_id": target_item_id,
                "text_rank": rank_of(target_item_id, text_items),
                "cf_rank": rank_of(target_item_id, cf_items),
                "fusion_rank": rank_of(target_item_id, fusion_items),
                "heuristic_rank": rank_of(target_item_id, reranked_items),
                "num_text_candidates": len(text_items),
                "num_cf_candidates": len(cf_items),
                "num_union_candidates": len(fusion_items),
                "num_reranked_candidates": len(reranked_items),
            }
        )

    if target_mismatches:
        preview = target_mismatches[:5]
        raise ValueError(f"Target item mismatch across artifacts. First mismatches: {preview}")

    coverage_rows = make_source_coverage_rows(records, ks)
    rank_distribution_rows = make_rank_distribution_rows(records)
    topk_metrics = make_topk_metrics(records, ks)
    failure_summary_20, failure_rows = make_failure_rows(records, 20)
    coverage_by_scope = {row["scope"]: row for row in coverage_rows}

    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add_check("root is valid split", "valid" in [part.lower() for part in root.parts], root.as_posix())
    for path in input_paths:
        add_check(f"input under root: {path.as_posix()}", is_under(path, root), path.as_posix())
        add_check(f"input has no test path: {path.as_posix()}", no_test_path(path), path.as_posix())
    for path in [
        out_dir / "headroom_summary.json",
        out_dir / "headroom_summary.csv",
        out_dir / "headroom_summary.md",
        out_dir / "source_coverage.csv",
        out_dir / "target_rank_distribution.csv",
        out_dir / "ranking_failures.jsonl",
        out_dir / "data_protocol_recommendation.md",
    ]:
        add_check(f"output under root: {path.as_posix()}", is_under(path, root), path.as_posix())
        add_check(f"output has no test path: {path.as_posix()}", no_test_path(path), path.as_posix())
    add_check("consistency overall_ok", consistency_report.get("overall_ok") is True, consistency_report.get("overall_ok"))
    add_check("summary split valid", metrics_summary.get("split") == args.split, metrics_summary.get("split"))
    add_check("summary candidate mode exact", metrics_summary.get("candidate_mode") == args.candidate_mode, metrics_summary.get("candidate_mode"))
    add_check("fusion eval_split valid", fusion_report.get("inputs", {}).get("eval_split") == args.split, fusion_report.get("inputs", {}).get("eval_split"))
    add_check("rerank eval_split valid", rerank_report.get("inputs", {}).get("eval_split") == args.split, rerank_report.get("inputs", {}).get("eval_split"))
    add_check("row count matches summary", len(records) == int(metrics_summary.get("num_valid_rows", -1)), len(records))
    add_check("text_hr20 matches summary", metric_close(topk_metrics["text_hr@20"], metrics_summary.get("text_hr20")), topk_metrics["text_hr@20"])
    add_check("cf_hr20 matches summary", metric_close(topk_metrics["cf_hr@20"], metrics_summary.get("cf_hr20")), topk_metrics["cf_hr@20"])
    add_check("oracle_union_hr20 matches summary", metric_close(topk_metrics["oracle_union_hr@20"], metrics_summary.get("union_hr20")), topk_metrics["oracle_union_hr@20"])
    add_check(
        "heuristic_rerank_hr20 matches summary",
        metric_close(topk_metrics["heuristic_rerank_hr@20"], metrics_summary.get("heuristic_rerank_hr20")),
        topk_metrics["heuristic_rerank_hr@20"],
    )

    summary: dict[str, Any] = {
        "split": args.split,
        "category": args.category,
        "candidate_mode": args.candidate_mode,
        "num_samples": len(records),
        "input_root": root.as_posix(),
        "output_dir": out_dir.as_posix(),
        "topk_metrics": topk_metrics,
        "source_coverage_full": coverage_by_scope["full"],
        "source_coverage_at_20": coverage_by_scope["@20"],
        "ranking_failure@20": failure_summary_20,
        "checks": checks,
        "overall_ok": all(check["ok"] for check in checks),
        "report_inputs": {
            "text_candidates": text_candidates_path.as_posix(),
            "cf_candidates": cf_candidates_path.as_posix(),
            "fusion_jsonl": fusion_jsonl_path.as_posix(),
            "rerank_jsonl": rerank_jsonl_path.as_posix(),
            "fusion_report": fusion_report_path.as_posix(),
            "rerank_report": rerank_report_path.as_posix(),
            "metrics_summary": metrics_summary_path.as_posix(),
            "consistency_report": consistency_report_path.as_posix(),
        },
    }
    recommendation = make_recommendation(summary, args.gap_threshold, args.fit_ratio, args.split_seed)
    summary["recommendation"] = recommendation

    flat_rows: list[dict[str, Any]] = []
    flatten("", summary, flat_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "headroom_summary.json", summary)
    write_csv(out_dir / "headroom_summary.csv", ["metric", "value"], flat_rows)
    write_csv(out_dir / "source_coverage.csv", sorted({key for row in coverage_rows for key in row}), coverage_rows)
    write_csv(
        out_dir / "target_rank_distribution.csv",
        ["rank_source", "bucket", "count", "rate"],
        rank_distribution_rows,
    )
    write_jsonl(out_dir / "ranking_failures.jsonl", failure_rows)
    with open(out_dir / "headroom_summary.md", "w", encoding="utf-8") as f:
        f.write(make_markdown(summary, coverage_rows, recommendation))
    with open(out_dir / "data_protocol_recommendation.md", "w", encoding="utf-8") as f:
        f.write(make_protocol_markdown(recommendation, summary))

    if not summary["overall_ok"]:
        failed = [check for check in checks if not check["ok"]]
        raise SystemExit(f"Headroom diagnosis finished but checks failed: {failed}")

    print(f"Wrote headroom summary: {out_dir / 'headroom_summary.md'}")
    print(
        "P2-0 summary: "
        f"union@20={topk_metrics['oracle_union_hr@20']} "
        f"heuristic@20={topk_metrics['heuristic_rerank_hr@20']} "
        f"gap@20={topk_metrics['recoverable_gap_vs_heuristic@20']} "
        f"decision={recommendation['decision']}"
    )


if __name__ == "__main__":
    main()
