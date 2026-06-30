#!/usr/bin/env python3
"""Summarize SID-only smoke runs across SID versions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


NOT_AVAILABLE = "not_available"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact SID-only smoke grid summary CSV.")
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--versions", nargs="+", required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--sid-versions-root", type=Path, default=Path("data/Amazon/sid_versions"))
    parser.add_argument("--sample-size", default="10000")
    parser.add_argument("--num-epochs", default="1")
    parser.add_argument("--run-label", default="noearly")
    parser.add_argument("--num-beams", default="20")
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def version_result_dir(args: argparse.Namespace, version: str) -> Path:
    suffix = f"_{args.run_label}" if args.run_label else ""
    legacy_industrial = {
        "text_mbk_k256_dedup",
        "cs_alpha0.7_k512_dedup",
    }
    if args.category == "Industrial_and_Scientific" and version in legacy_industrial:
        name = f"calc_plus_sidonly_Industrial_{version}_sample{args.sample_size}_ep{args.num_epochs}{suffix}_beam{args.num_beams}"
    else:
        name = f"calc_plus_sidonly_{args.category}_{version}_sample{args.sample_size}_ep{args.num_epochs}{suffix}_beam{args.num_beams}"
    path = args.results_root / name
    if path.exists():
        return path
    matches = sorted(args.results_root.glob(f"calc_plus_sidonly_*{version}*sample{args.sample_size}_ep{args.num_epochs}{suffix}_beam{args.num_beams}"))
    return matches[0] if matches else path


def generation_report_path(args: argparse.Namespace, version: str) -> Path:
    return args.sid_versions_root / version / args.category / "reports" / "generation_report.json"


def infer_embedding_type(version: str) -> str:
    if version.startswith("text"):
        return "text"
    if version.startswith("cs_alpha"):
        return "cs"
    if version.startswith("cf"):
        return "cf"
    return NOT_AVAILABLE


def infer_codebook_size(version: str) -> str:
    match = re.search(r"_k(\d+)", version)
    return match.group(1) if match else NOT_AVAILABLE


def get_nested(data: dict[str, Any], keys: list[str], default: Any = NOT_AVAILABLE) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_row(args: argparse.Namespace, version: str) -> dict[str, Any]:
    result_dir = version_result_dir(args, version)
    report_path = result_dir / f"eval_report_{version}.json"
    diag_path = result_dir / f"diagnose_{version}.json"
    gen_path = generation_report_path(args, version)

    row: dict[str, Any] = {
        "sid_version": version,
        "embedding_type": infer_embedding_type(version),
        "codebook_size": infer_codebook_size(version),
        "eval_report": report_path.as_posix(),
        "diagnostic_json": diag_path.as_posix(),
        "generation_report": gen_path.as_posix(),
    }

    if gen_path.exists():
        gen = load_json(gen_path)
        row.update({
            "pre_dedup_collision": get_nested(gen, ["pre_dedup", "collision_rate"]),
            "post_dedup_collision": get_nested(gen, ["post_dedup", "collision_rate"]),
            "num_items": gen.get("num_items", get_nested(gen, ["post_dedup", "num_items"])),
            "unique_sid": gen.get("num_unique_sid", get_nested(gen, ["post_dedup", "num_unique_sid"])),
        })
    else:
        row.update({
            "pre_dedup_collision": NOT_AVAILABLE,
            "post_dedup_collision": NOT_AVAILABLE,
            "num_items": NOT_AVAILABLE,
            "unique_sid": NOT_AVAILABLE,
        })

    if report_path.exists():
        report = load_json(report_path)
        sid = report.get("sid_level_hr_ndcg", {})
        row.update({
            "overall_hr20": sid.get("hr@20", NOT_AVAILABLE),
            "overall_ndcg20": sid.get("ndcg@20", NOT_AVAILABLE),
            "invalid_rate": get_nested(report, ["validity", "invalid_sid_rate"]),
            "duplicate_rate": get_nested(report, ["duplicate_generation", "duplicate_generation_rate"]),
        })
        sid_len = report.get("sid_length_distribution", {}).get("targets", {})
        row["sid_len3_count"] = sid_len.get("3", 0)
        row["sid_len4_count"] = sid_len.get("4", 0)
    else:
        row.update({
            "overall_hr20": NOT_AVAILABLE,
            "overall_ndcg20": NOT_AVAILABLE,
            "invalid_rate": NOT_AVAILABLE,
            "duplicate_rate": NOT_AVAILABLE,
            "sid_len3_count": NOT_AVAILABLE,
            "sid_len4_count": NOT_AVAILABLE,
        })

    if diag_path.exists():
        diag = load_json(diag_path)
        compact = diag.get("compact_summary", {})
        row.update({
            "len3_hr20": compact.get("len3_hr@20", NOT_AVAILABLE),
            "len4_hr20": compact.get("len4_hr@20", NOT_AVAILABLE),
            "head_hr20": compact.get("head_hr@20", NOT_AVAILABLE),
            "mid_hr20": compact.get("mid_hr@20", NOT_AVAILABLE),
            "tail_hr20": compact.get("tail_hr@20", NOT_AVAILABLE),
            "prefix1_hr20_len4": compact.get("prefix1_hr@20_len4", NOT_AVAILABLE),
            "prefix2_hr20_len4": compact.get("prefix2_hr@20_len4", NOT_AVAILABLE),
            "prefix3_hr20_len4": compact.get("prefix3_hr@20_len4", NOT_AVAILABLE),
            "prefix4_hr20_len4": compact.get("prefix4_hr@20_len4", NOT_AVAILABLE),
        })
    else:
        row.update({
            "len3_hr20": NOT_AVAILABLE,
            "len4_hr20": NOT_AVAILABLE,
            "head_hr20": NOT_AVAILABLE,
            "mid_hr20": NOT_AVAILABLE,
            "tail_hr20": NOT_AVAILABLE,
            "prefix1_hr20_len4": NOT_AVAILABLE,
            "prefix2_hr20_len4": NOT_AVAILABLE,
            "prefix3_hr20_len4": NOT_AVAILABLE,
            "prefix4_hr20_len4": NOT_AVAILABLE,
        })

    return row


def main() -> None:
    args = parse_args()
    fieldnames = [
        "sid_version",
        "embedding_type",
        "codebook_size",
        "pre_dedup_collision",
        "post_dedup_collision",
        "num_items",
        "unique_sid",
        "sid_len3_count",
        "sid_len4_count",
        "overall_hr20",
        "overall_ndcg20",
        "len3_hr20",
        "len4_hr20",
        "head_hr20",
        "mid_hr20",
        "tail_hr20",
        "prefix1_hr20_len4",
        "prefix2_hr20_len4",
        "prefix3_hr20_len4",
        "prefix4_hr20_len4",
        "invalid_rate",
        "duplicate_rate",
        "eval_report",
        "diagnostic_json",
        "generation_report",
    ]
    rows = [build_row(args, version) for version in args.versions]
    write_csv(args.output_csv, rows, fieldnames)
    print(f"Wrote grid summary CSV: {args.output_csv}")
    for row in rows:
        print(
            row["sid_version"],
            f"HR@20={row['overall_hr20']}",
            f"NDCG@20={row['overall_ndcg20']}",
            f"len4_hr20={row['len4_hr20']}",
        )


if __name__ == "__main__":
    main()
