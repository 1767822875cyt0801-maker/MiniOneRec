#!/usr/bin/env python3
"""Run the Stage 7 P3 valid-only fusion parameter sweep.

The runner sweeps RRF lambda_text and frozen-ranker source_weight on existing
valid exact candidate files. It never trains a ranker and refuses test paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CATEGORY = "Industrial_and_Scientific"
DEFAULT_LAMBDAS = [0.4, 0.5, 0.6]
DEFAULT_SOURCE_WEIGHTS = [2.0, 4.0, 6.0]
DEFAULT_KS = [1, 5, 10, 20, 50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 7 P3 valid fusion sweep.")
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--eval-split", choices=["valid"], default="valid")
    parser.add_argument("--candidate-mode", choices=["exact"], default="exact")
    parser.add_argument("--text-candidates", type=Path, required=True)
    parser.add_argument("--behavior-candidates", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--item-emb", type=Path, required=True)
    parser.add_argument("--row-index", type=Path, required=True)
    parser.add_argument("--frozen-ranker", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--source-weights", type=float, nargs="+", default=DEFAULT_SOURCE_WEIGHTS)
    parser.add_argument("--k-rrf", type=float, default=60.0)
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def no_test_path(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    return "test" not in lowered_parts and path.name.lower() != "test.csv"


def require_no_test_paths(paths: list[Path]) -> None:
    bad = [path.as_posix() for path in paths if not no_test_path(path)]
    if bad:
        raise SystemExit(f"P3 valid sweep refuses test paths: {bad}")


def require_inputs(paths: list[Path]) -> None:
    missing = [path.as_posix() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required P3 sweep input files are missing: {missing}")


def safe_float(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def config_id(lambda_text: float, source_weight: float) -> str:
    return f"lambda{safe_float(lambda_text)}_source{safe_float(source_weight)}"


def command_string(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def run_logged(command: list[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write("$ " + command_string(command) + "\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)


def sweep_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for lambda_text in args.lambdas:
        for source_weight in args.source_weights:
            cid = config_id(lambda_text, source_weight)
            cfg_dir = args.out_dir / cid
            fusion_dir = cfg_dir / "fusion"
            ranker_dir = cfg_dir / "frozen_history_ranker"
            fusion_cmd = [
                args.python,
                "fuse_dual_sid_candidates.py",
                "--text-candidates",
                args.text_candidates.as_posix(),
                "--behavior-candidates",
                args.behavior_candidates.as_posix(),
                "--eval-csv",
                args.eval_csv.as_posix(),
                "--eval-split",
                args.eval_split,
                "--out-dir",
                fusion_dir.as_posix(),
                "--ks",
                *[str(k) for k in args.ks],
                "--lambda-text",
                str(lambda_text),
                "--k-rrf",
                str(args.k_rrf),
            ]
            apply_cmd = [
                args.python,
                "scripts/apply_stage7_p2_history_ranker.py",
                "--candidate-jsonl",
                (fusion_dir / "dual_fused_candidates.jsonl").as_posix(),
                "--frozen-ranker",
                args.frozen_ranker.as_posix(),
                "--train-csv",
                args.train_csv.as_posix(),
                "--item-emb",
                args.item_emb.as_posix(),
                "--row-index",
                args.row_index.as_posix(),
                "--output-dir",
                ranker_dir.as_posix(),
                "--eval-split",
                args.eval_split,
                "--source-weight",
                str(source_weight),
                "--ks",
                *[str(k) for k in args.ks],
            ]
            plan.append(
                {
                    "config_id": cid,
                    "lambda_text": lambda_text,
                    "source_weight": source_weight,
                    "config_dir": cfg_dir,
                    "fusion_dir": fusion_dir,
                    "ranker_dir": ranker_dir,
                    "fusion_cmd": fusion_cmd,
                    "apply_cmd": apply_cmd,
                }
            )
    return plan


def summarize_config(entry: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    fusion_report = read_json(entry["fusion_dir"] / "dual_fusion_report.json")
    ranker_report = read_json(entry["ranker_dir"] / "ranker_apply_report.json")
    return {
        "category": args.category,
        "eval_split": args.eval_split,
        "candidate_mode": args.candidate_mode,
        "config_id": entry["config_id"],
        "lambda_text": entry["lambda_text"],
        "source_weight": entry["source_weight"],
        "k_rrf": args.k_rrf,
        "text_hr20": fusion_report["text_stream"].get("hr@20"),
        "behavior_hr20": fusion_report["behavior_stream"].get("hr@20"),
        "union_hr20": fusion_report["union_recall"].get("union_hit@20"),
        "rrf_hr20": fusion_report["rrf_fusion"].get("hr@20"),
        "rrf_ndcg20": fusion_report["rrf_fusion"].get("ndcg@20"),
        "history_ranker_hr20": ranker_report["history_aware_rerank"].get("hr@20"),
        "history_ranker_ndcg20": ranker_report["history_aware_rerank"].get("ndcg@20"),
        "mean_candidates": ranker_report["candidate_count"].get("mean"),
        "max_candidates": ranker_report["candidate_count"].get("max"),
        "frozen_ranker_sha256_before": ranker_report["frozen_ranker_integrity"].get("sha256_before"),
        "frozen_ranker_sha256_after": ranker_report["frozen_ranker_integrity"].get("sha256_after"),
        "frozen_ranker_unchanged": ranker_report["frozen_ranker_integrity"].get("unchanged"),
        "config_dir": entry["config_dir"].as_posix(),
    }


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select best P3 config from an empty summary.")

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        ndcg = float(row.get("history_ranker_ndcg20") or 0.0)
        hr = float(row.get("history_ranker_hr20") or 0.0)
        mean_candidates = float(row.get("mean_candidates") or 0.0)
        simplicity = -abs(float(row.get("lambda_text") or 0.0) - 0.5) - abs(float(row.get("source_weight") or 0.0) - 4.0) / 10.0
        return (ndcg, hr, -mean_candidates, simplicity)

    selected = max(rows, key=key)
    return {
        "selection_rule": "max valid NDCG@20, tie valid HR@20, tie fewer candidates, tie closer to lambda=0.5/source_weight=4.0",
        "selected_config_id": selected["config_id"],
        "selected_row": selected,
    }


def write_markdown(path: Path, rows: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    lines = [
        "# Stage 7 P3 Valid Fusion Sweep",
        "",
        f"- selection_rule: `{selection['selection_rule']}`",
        f"- selected_config_id: `{selection['selected_config_id']}`",
        "",
        "| Config | Lambda | Source Weight | HR@20 | NDCG@20 | Mean Candidates | Frozen Unchanged |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {config_id} | {lambda_text:g} | {source_weight:g} | {hr:.6f} | {ndcg:.6f} | {cand:.2f} | {unchanged} |".format(
                config_id=row["config_id"],
                lambda_text=float(row["lambda_text"]),
                source_weight=float(row["source_weight"]),
                hr=float(row.get("history_ranker_hr20") or 0.0),
                ndcg=float(row.get("history_ranker_ndcg20") or 0.0),
                cand=float(row.get("mean_candidates") or 0.0),
                unchanged=row.get("frozen_ranker_unchanged"),
            )
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    paths = [
        args.text_candidates,
        args.behavior_candidates,
        args.eval_csv,
        args.train_csv,
        args.item_emb,
        args.row_index,
        args.frozen_ranker,
        args.out_dir,
    ]
    require_no_test_paths(paths)
    plan = sweep_plan(args)

    print("Stage 7 P3 valid fusion sweep")
    print(f"  category={args.category}")
    print(f"  split={args.eval_split}")
    print(f"  candidate_mode={args.candidate_mode}")
    print(f"  out_dir={args.out_dir}")
    print(f"  combinations={len(plan)}")

    if args.dry_run:
        missing = [path.as_posix() for path in paths[:-1] if not path.is_file()]
        if missing:
            print("  missing_inputs=" + json.dumps(missing, ensure_ascii=False))
        for entry in plan:
            print(f"\n[{entry['config_id']}]")
            print(command_string(entry["fusion_cmd"]))
            print(command_string(entry["apply_cmd"]))
        print("\nDRY_RUN complete. No files were written.")
        return

    require_inputs(paths[:-1])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for entry in plan:
        cfg_dir = entry["config_dir"]
        if cfg_dir.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing config directory without --force: {cfg_dir}")
        cfg_dir.mkdir(parents=True, exist_ok=True)
        log_file = cfg_dir / "run.log"
        config = {
            "category": args.category,
            "eval_split": args.eval_split,
            "candidate_mode": args.candidate_mode,
            "config_id": entry["config_id"],
            "lambda_text": entry["lambda_text"],
            "source_weight": entry["source_weight"],
            "k_rrf": args.k_rrf,
            "text_candidates": args.text_candidates.as_posix(),
            "behavior_candidates": args.behavior_candidates.as_posix(),
            "eval_csv": args.eval_csv.as_posix(),
            "train_csv": args.train_csv.as_posix(),
            "item_emb": args.item_emb.as_posix(),
            "row_index": args.row_index.as_posix(),
            "frozen_ranker": args.frozen_ranker.as_posix(),
        }
        write_json(cfg_dir / "config.json", config)
        run_logged(entry["fusion_cmd"], log_file)
        run_logged(entry["apply_cmd"], log_file)
        rows.append(summarize_config(entry, args))

    fieldnames = [
        "category",
        "eval_split",
        "candidate_mode",
        "config_id",
        "lambda_text",
        "source_weight",
        "k_rrf",
        "text_hr20",
        "behavior_hr20",
        "union_hr20",
        "rrf_hr20",
        "rrf_ndcg20",
        "history_ranker_hr20",
        "history_ranker_ndcg20",
        "mean_candidates",
        "max_candidates",
        "frozen_ranker_sha256_before",
        "frozen_ranker_sha256_after",
        "frozen_ranker_unchanged",
        "config_dir",
    ]
    summary_csv = args.out_dir / "fusion_sweep_valid.csv"
    write_csv(summary_csv, fieldnames, rows)
    selection = select_best(rows)
    write_json(args.out_dir / "selected_config.json", selection)
    write_markdown(args.out_dir / "fusion_sweep_valid.md", rows, selection)
    print(f"Wrote P3 sweep summary: {summary_csv}")
    print(f"Selected: {selection['selected_config_id']}")


if __name__ == "__main__":
    main()
