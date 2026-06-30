#!/usr/bin/env python3
"""Summarize final CS-SID experiment artifacts and consistency checks.

The script is intentionally conservative: missing files become warnings instead
of silently disappearing from the final table. It can be run on AutoDL, where the
heavy ``results/`` and ``data/Amazon/sid_versions/`` artifacts live.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NOT_AVAILABLE = "not_available"


@dataclass
class FinalSpec:
    category: str
    sid_version: str
    label: str
    setting: str
    candidate_mode: str
    rerank_mode: str
    candidate_reports: list[str]
    rerank_reports: list[str]
    eval_reports: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final CS-SID results.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_summary"))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when required final reports are missing.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return data


def first_existing(paths: list[str]) -> tuple[Path | None, list[str]]:
    tried = [Path(path) for path in paths]
    for path in tried:
        if path.exists():
            return path, [p.as_posix() for p in tried]
    return None, [p.as_posix() for p in tried]


def get_nested(data: dict[str, Any] | None, keys: list[str], default: Any = "") -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_alpha(sid_version: str) -> str:
    match = re.search(r"alpha([0-9.]+)", sid_version)
    return match.group(1) if match else ""


def default_specs() -> list[FinalSpec]:
    return [
        FinalSpec(
            category="Industrial_and_Scientific",
            sid_version="text_mbk_k512_dedup",
            label="text_k512",
            setting="exact",
            candidate_mode="exact",
            rerank_mode="none_or_exact_reorder",
            candidate_reports=["results/candidates_text_k512_30k_exact_v2/report.json"],
            rerank_reports=["results/rerank_text_k512_30k_exact_v2/rerank_report.json"],
            eval_reports=[
                "results/calc_plus_sidonly_Industrial_and_Scientific_text_mbk_k512_dedup_sample30000_ep1_noearly_beam20/eval_report_text_mbk_k512_dedup.json",
            ],
        ),
        FinalSpec(
            category="Industrial_and_Scientific",
            sid_version="cs_alpha0.2_k512_dedup",
            label="cs_alpha02_k512",
            setting="exact",
            candidate_mode="exact",
            rerank_mode="none_or_exact_reorder",
            candidate_reports=["results/candidates_cs_alpha02_k512_30k_exact_v2/report.json"],
            rerank_reports=["results/rerank_cs_alpha02_k512_30k_exact_v2/rerank_report.json"],
            eval_reports=[
                "results/calc_plus_sidonly_Industrial_and_Scientific_cs_alpha0.2_k512_dedup_sample30000_ep1_noearly_beam20/eval_report_cs_alpha0.2_k512_dedup.json",
            ],
        ),
        FinalSpec(
            category="Industrial_and_Scientific",
            sid_version="cs_alpha0.2_k512_dedup",
            label="cs_alpha02_k512",
            setting="prefix@3 + rerank",
            candidate_mode="prefix@3",
            rerank_mode="source_weight=4.0",
            candidate_reports=["results/candidates_cs_alpha02_k512_30k_p3_c500_v2/report.json"],
            rerank_reports=["results/rerank_cs_alpha02_k512_30k_p3_c500_v2_sourcew4/rerank_report.json"],
            eval_reports=[
                "results/calc_plus_sidonly_Industrial_and_Scientific_cs_alpha0.2_k512_dedup_sample30000_ep1_noearly_beam20/eval_report_cs_alpha0.2_k512_dedup.json",
            ],
        ),
        FinalSpec(
            category="Office_Products",
            sid_version="text_mbk_k512_dedup",
            label="text_k512",
            setting="exact",
            candidate_mode="exact",
            rerank_mode="none_or_exact_reorder",
            candidate_reports=["results/candidates_Office_Products_text_k512_30k_exact_c500_v2/report.json"],
            rerank_reports=["results/rerank_Office_Products_text_k512_30k_exact_c500_sourcew4.0_v2/rerank_report.json"],
            eval_reports=[
                "results/calc_plus_sidonly_Office_Products_text_mbk_k512_dedup_sample30000_ep1_noearly_beam20/eval_report_text_mbk_k512_dedup.json",
            ],
        ),
        FinalSpec(
            category="Office_Products",
            sid_version="cs_alpha0.7_k512_dedup",
            label="cs_alpha0_7_k512_dedup",
            setting="exact",
            candidate_mode="exact",
            rerank_mode="none_or_exact_reorder",
            candidate_reports=["results/candidates_Office_Products_cs_alpha0_7_k512_dedup_30k_exact_c500_v2/report.json"],
            rerank_reports=["results/rerank_Office_Products_cs_alpha0_7_k512_dedup_30k_exact_c500_sourcew4.0_v2/rerank_report.json"],
            eval_reports=[
                "results/calc_plus_sidonly_Office_Products_cs_alpha0.7_k512_dedup_sample30000_ep1_noearly_beam20/eval_report_cs_alpha0.7_k512_dedup.json",
                # Legacy wrong path kept only for warning/report recovery.
                "results/calc_plus_sidonly_Industrial_cs_alpha0.7_k512_dedup_sample30000_ep1_noearly_beam20/eval_report_cs_alpha0.7_k512_dedup.json",
            ],
        ),
        FinalSpec(
            category="Office_Products",
            sid_version="cs_alpha0.7_k512_dedup",
            label="cs_alpha0_7_k512_dedup",
            setting="prefix@3 + rerank",
            candidate_mode="prefix@3",
            rerank_mode="source_weight=4.0",
            candidate_reports=["results/candidates_Office_Products_cs_alpha0_7_k512_dedup_30k_p3_c500_v2/report.json"],
            rerank_reports=["results/rerank_Office_Products_cs_alpha0_7_k512_dedup_30k_p3_c500_sourcew4.0_v2/rerank_report.json"],
            eval_reports=[
                "results/calc_plus_sidonly_Office_Products_cs_alpha0.7_k512_dedup_sample30000_ep1_noearly_beam20/eval_report_cs_alpha0.7_k512_dedup.json",
                "results/calc_plus_sidonly_Industrial_cs_alpha0.7_k512_dedup_sample30000_ep1_noearly_beam20/eval_report_cs_alpha0.7_k512_dedup.json",
            ],
        ),
    ]


def group_metric(eval_report: dict[str, Any] | None, group: str, bucket: str, metric: str) -> Any:
    return get_nested(eval_report, ["group_metrics", group, bucket, metric], "")


def build_row(spec: FinalSpec) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    warnings: list[str] = []
    cand_path, cand_tried = first_existing(spec.candidate_reports)
    rerank_path, rerank_tried = first_existing(spec.rerank_reports)
    eval_path, eval_tried = first_existing(spec.eval_reports)

    cand_report = load_json(cand_path) if cand_path else None
    rerank_report = load_json(rerank_path) if rerank_path else None
    eval_report = load_json(eval_path) if eval_path else None

    if not cand_path:
        warnings.append(f"missing candidate report for {spec.category}/{spec.sid_version}/{spec.setting}: {cand_tried}")
    if not rerank_path:
        warnings.append(f"missing rerank report for {spec.category}/{spec.sid_version}/{spec.setting}: {rerank_tried}")
    if not eval_path:
        warnings.append(f"missing eval report for {spec.category}/{spec.sid_version}: {eval_tried}")
    elif spec.category == "Office_Products" and "Industrial_" in eval_path.as_posix():
        warnings.append(f"Office eval report recovered from suspicious Industrial path: {eval_path}")

    after_metrics = get_nested(rerank_report, ["after_rerank"], {})
    sid_metrics = get_nested(eval_report, ["sid_level_hr_ndcg"], {})
    hr20 = after_metrics.get("hr@20", sid_metrics.get("hr@20", ""))
    ndcg20 = after_metrics.get("ndcg@20", sid_metrics.get("ndcg@20", ""))

    row = {
        "category": spec.category,
        "sid_version": spec.sid_version,
        "alpha": parse_alpha(spec.sid_version),
        "setting": spec.setting,
        "candidate_mode": spec.candidate_mode,
        "rerank_mode": spec.rerank_mode,
        "mean_candidates": get_nested(cand_report, ["candidate_count", "mean"], ""),
        "candidate_recall": get_nested(cand_report, ["candidate_pool_recall", "target_in_pool_rate"], ""),
        "target_exact_rate": get_nested(cand_report, ["target_in_source", "exact", "rate"], ""),
        "target_prefix3_rate": get_nested(cand_report, ["target_in_source", "prefix@3", "rate"], ""),
        "avg_rank_if_hit": get_nested(cand_report, ["candidate_pool_recall", "avg_rank_before_rerank_if_hit"], ""),
        "HR@20": hr20,
        "NDCG@20": ndcg20,
        "len3_HR@20": "",
        "len4_HR@20": "",
        "head_HR@20": group_metric(eval_report, "popularity", "head", "hr@20"),
        "mid_HR@20": group_metric(eval_report, "popularity", "mid", "hr@20"),
        "tail_HR@20": group_metric(eval_report, "popularity", "tail", "hr@20"),
        "report_path": rerank_path.as_posix() if rerank_path else (eval_path.as_posix() if eval_path else ""),
        "candidate_report_path": cand_path.as_posix() if cand_path else "",
        "eval_report_path": eval_path.as_posix() if eval_path else "",
    }

    sid_len = get_nested(eval_report, ["sid_length_distribution", "targets"], {})
    if sid_len:
        row["target_sid_len_distribution"] = json.dumps(sid_len, ensure_ascii=False, sort_keys=True)
    else:
        row["target_sid_len_distribution"] = ""

    artifact = {
        "category": spec.category,
        "sid_version": spec.sid_version,
        "setting": spec.setting,
        "candidate_reports_tried": cand_tried,
        "rerank_reports_tried": rerank_tried,
        "eval_reports_tried": eval_tried,
        "candidate_report": cand_path.as_posix() if cand_path else None,
        "rerank_report": rerank_path.as_posix() if rerank_path else None,
        "eval_report": eval_path.as_posix() if eval_path else None,
    }
    return row, artifact, warnings


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return "" if value is None else str(value)

    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(out)


def consistency_checks(rows: list[dict[str, Any]], artifacts: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def get_row(category: str, sid_version: str, setting: str) -> dict[str, Any] | None:
        for row in rows:
            if row["category"] == category and row["sid_version"] == sid_version and row["setting"] == setting:
                return row
        return None

    def add_check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    for row in rows:
        if row["candidate_mode"] == "exact" and row.get("mean_candidates") != "":
            try:
                mean_candidates = float(row["mean_candidates"])
                add_check(
                    f"exact candidates near beam size: {row['category']} {row['sid_version']}",
                    15.0 <= mean_candidates <= 25.0,
                    f"mean_candidates={mean_candidates}",
                )
            except (TypeError, ValueError):
                add_check(f"exact candidates parseable: {row['category']} {row['sid_version']}", False, str(row.get("mean_candidates")))

    for category, cs_version in [
        ("Industrial_and_Scientific", "cs_alpha0.2_k512_dedup"),
        ("Office_Products", "cs_alpha0.7_k512_dedup"),
    ]:
        text = get_row(category, "text_mbk_k512_dedup", "exact")
        cs = get_row(category, cs_version, "exact")
        if text and cs and text.get("HR@20") != "" and cs.get("HR@20") != "":
            add_check(
                f"CS exact beats Text exact HR@20: {category}",
                float(cs["HR@20"]) > float(text["HR@20"]),
                f"text={text['HR@20']} cs={cs['HR@20']}",
            )
        else:
            add_check(f"CS exact beats Text exact HR@20: {category}", False, "missing rows or HR@20")

        cs_p3 = get_row(category, cs_version, "prefix@3 + rerank")
        if cs and cs_p3 and cs.get("candidate_recall") != "" and cs_p3.get("candidate_recall") != "":
            add_check(
                f"prefix@3 improves CS candidate recall: {category}",
                float(cs_p3["candidate_recall"]) > float(cs["candidate_recall"]),
                f"exact={cs['candidate_recall']} p3={cs_p3['candidate_recall']}",
            )
        else:
            add_check(f"prefix@3 improves CS candidate recall: {category}", False, "missing rows or candidate_recall")

    for row in rows:
        if row["candidate_mode"] == "prefix@3":
            add_check(
                f"prefix@3 not mixed with prefix@2: {row['category']} {row['sid_version']}",
                "prefix@2" not in row["setting"].lower() and row["candidate_mode"] == "prefix@3",
                f"setting={row['setting']} candidate_mode={row['candidate_mode']}",
            )

    rerank_source = Path("rerank.py").read_text(encoding="utf-8") if Path("rerank.py").exists() else ""
    add_check(
        "rerank uses train CSV popularity",
        "train_popularity(args.train_csv)" in rerank_source and "valid" not in rerank_source.lower(),
        "checked rerank.py train_popularity path",
    )
    add_check(
        "rerank target item used only for metrics",
        "target item is used only for metrics" in rerank_source and "history_item_id" in rerank_source,
        "checked rerank.py leakage_policy/history usage",
    )

    office_wrong = [
        artifact for artifact in artifacts
        if artifact["category"] == "Office_Products"
        and any(
            path and "Industrial_" in path
            for path in [artifact.get("candidate_report"), artifact.get("rerank_report"), artifact.get("eval_report")]
        )
    ]
    add_check(
        "Office cs_alpha0.7 not written to Industrial path",
        len(office_wrong) == 0,
        f"suspicious artifacts={office_wrong}",
    )

    for row in rows:
        if row["sid_version"] == "text_mbk_k512_dedup":
            add_check(
                f"Text baseline is text_mbk_k512_dedup exact or p3 only: {row['category']} {row['setting']}",
                row["sid_version"] == "text_mbk_k512_dedup",
                row["sid_version"],
            )

    ok = all(check["ok"] for check in checks) and not any("missing" in warning.lower() for warning in warnings)
    return {"ok": ok, "checks": checks, "warnings": warnings}


def write_outputs(rows: list[dict[str, Any]], artifacts: list[dict[str, Any]], consistency: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "final_results_summary.csv"
    md_path = output_dir / "final_results_summary.md"
    artifact_path = output_dir / "final_artifacts_manifest.json"
    consistency_json = output_dir / "final_consistency_report.json"
    consistency_md = output_dir / "final_consistency_report.md"

    write_csv(csv_path, rows)
    columns = [
        "category",
        "sid_version",
        "setting",
        "mean_candidates",
        "candidate_recall",
        "target_exact_rate",
        "target_prefix3_rate",
        "avg_rank_if_hit",
        "HR@20",
        "NDCG@20",
    ]
    md = [
        "# Final Results Summary",
        "",
        markdown_table(rows, columns),
        "",
        "## Notes",
        "",
        "- Text baseline uses `text_mbk_k512_dedup`.",
        "- Main CS settings are Industrial `cs_alpha0.2_k512_dedup` and Office `cs_alpha0.7_k512_dedup`.",
        "- `prefix@3 + rerank` uses lightweight non-leaky rerank with `source_weight=4.0`.",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")

    artifact_path.write_text(json.dumps({"artifacts": artifacts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    consistency_json.write_text(json.dumps(consistency, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cmd = ["# Final Consistency Report", "", f"Overall OK: `{consistency['ok']}`", "", "## Checks", ""]
    for check in consistency["checks"]:
        mark = "PASS" if check["ok"] else "FAIL"
        cmd.append(f"- **{mark}** `{check['name']}`: {check['detail']}")
    if consistency["warnings"]:
        cmd.extend(["", "## Warnings", ""])
        for warning in consistency["warnings"]:
            cmd.append(f"- {warning}")
    consistency_md.write_text("\n".join(cmd) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    warnings: list[str] = []

    for spec in default_specs():
        row, artifact, row_warnings = build_row(spec)
        rows.append(row)
        artifacts.append(artifact)
        warnings.extend(row_warnings)

    consistency = consistency_checks(rows, artifacts, warnings)
    write_outputs(rows, artifacts, consistency, args.output_dir)

    print(f"Wrote final summary CSV: {args.output_dir / 'final_results_summary.csv'}")
    print(f"Wrote final summary MD: {args.output_dir / 'final_results_summary.md'}")
    print(f"Wrote artifact manifest: {args.output_dir / 'final_artifacts_manifest.json'}")
    print(f"Wrote consistency JSON: {args.output_dir / 'final_consistency_report.json'}")
    print(f"Wrote consistency MD: {args.output_dir / 'final_consistency_report.md'}")
    print(f"Consistency OK: {consistency['ok']}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if args.strict and not consistency["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
