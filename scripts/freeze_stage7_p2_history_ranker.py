#!/usr/bin/env python3
"""Freeze the accepted P2 history-aware reranker.

This is a closeout/protocol step. It does not train or rerank; it validates the
P2 artifact chain and writes a compact frozen manifest for downstream P3 work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/stage7_validation_protocol/valid/Industrial_and_Scientific")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the Stage 7 P2 history-aware ranker.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--split", choices=["valid"], default="valid")
    parser.add_argument("--candidate-mode", choices=["exact"], default="exact")
    parser.add_argument("--min-hr20-gain", type=float, default=0.0)
    parser.add_argument("--min-ndcg20-gain", type=float, default=0.0)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def assert_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def no_test_path(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    return "test" not in lowered_parts and path.name.lower() != "test.csv"


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def metric(report: dict[str, Any], split: str, model: str, name: str) -> float:
    return float(report["split_metrics"][split][model][name])


def make_metric_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ["valid_fit", "valid_select", "all_valid"]:
        for model_name in ["fusion_rrf", "heuristic_rerank", "learned_rerank"]:
            values = report["split_metrics"][split_name][model_name]
            label = "history_aware_rerank" if model_name == "learned_rerank" else model_name
            rows.append(
                {
                    "split": split_name,
                    "model": label,
                    "hr20": values.get("hr@20"),
                    "ndcg20": values.get("ndcg@20"),
                    "hr10": values.get("hr@10"),
                    "ndcg10": values.get("ndcg@10"),
                    "hr5": values.get("hr@5"),
                    "ndcg5": values.get("ndcg@5"),
                }
            )
    for split_name in ["valid_fit", "valid_select", "all_valid"]:
        delta = report["split_metrics"][split_name]["delta"]
        rows.append(
            {
                "split": split_name,
                "model": "history_minus_heuristic",
                "hr20": delta.get("learned_minus_heuristic_hr@20"),
                "ndcg20": delta.get("learned_minus_heuristic_ndcg@20"),
                "hr10": delta.get("learned_minus_heuristic_hr@10"),
                "ndcg10": delta.get("learned_minus_heuristic_ndcg@10"),
                "hr5": delta.get("learned_minus_heuristic_hr@5"),
                "ndcg5": delta.get("learned_minus_heuristic_ndcg@5"),
            }
        )
    return rows


def top_feature_rows(feature_weights_path: Path, limit: int = 12) -> list[dict[str, Any]]:
    rows = read_csv_rows(feature_weights_path)
    rows.sort(key=lambda row: -float(row.get("abs_weight", 0.0)))
    return rows[:limit]


def make_markdown(manifest: dict[str, Any], metric_rows: list[dict[str, Any]]) -> str:
    frozen = manifest["frozen_ranker"]
    selection = manifest["selection"]
    lines = [
        "# P2 Frozen History-aware Ranker",
        "",
        f"- split: `{manifest['split']}`",
        f"- category: `{manifest['category']}`",
        f"- candidate_mode: `{manifest['candidate_mode']}`",
        f"- frozen_ranker_id: `{frozen['ranker_id']}`",
        f"- selected_config: `{frozen['selected_config_id']}`",
        f"- overall_ok: `{manifest['overall_ok']}`",
        "",
        "## Selection",
        "",
        f"- decision: `{selection['decision']}`",
        f"- valid_select HR@20 gain: `{selection['valid_select_hr20_gain']}`",
        f"- valid_select NDCG@20 gain: `{selection['valid_select_ndcg20_gain']}`",
        "",
        "## Metrics",
        "",
        "| Split | Model | HR@20 | NDCG@20 |",
        "|---|---|---:|---:|",
    ]
    for row in metric_rows:
        if row["model"] == "history_minus_heuristic":
            continue
        lines.append(f"| {row['split']} | {row['model']} | {float(row['hr20']):.6f} | {float(row['ndcg20']):.6f} |")
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- This manifest freezes P2 ranking-side selection only.",
            "- The frozen ranker was selected on `valid_select`.",
            "- No test path is read or written by this closeout step.",
            "- Downstream P3 experiments should treat this ranker as fixed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = args.out_dir or (root / "p2_frozen_ranker")

    paths = {
        "p1_consistency": root / "summary/consistency_report.json",
        "p2_headroom": root / "p2_headroom/headroom_summary.json",
        "p2_valid_split": root / "p2_valid_split/valid_split_summary.json",
        "p2_minimal_report": root / "p2_minimal_ranker/ranker_report.json",
        "p2_history_report": root / "p2_history_ranker/ranker_report.json",
        "p2_history_model": root / "p2_history_ranker/model.json",
        "p2_history_grid": root / "p2_history_ranker/model_grid.csv",
        "p2_history_feature_weights": root / "p2_history_ranker/feature_weights.csv",
    }
    for path in paths.values():
        assert_file(path)

    p1 = read_json(paths["p1_consistency"])
    headroom = read_json(paths["p2_headroom"])
    valid_split = read_json(paths["p2_valid_split"])
    minimal = read_json(paths["p2_minimal_report"])
    history = read_json(paths["p2_history_report"])
    model = read_json(paths["p2_history_model"])

    selection = history["selection"]["freeze_recommendation"]
    valid_select_hr20_gain = float(selection["valid_select_hr20_gain"])
    valid_select_ndcg20_gain = float(selection["valid_select_ndcg20_gain"])
    selected_config_id = str(history["selected_config_id"])
    ranker_id = f"p2_history_ranker__{args.category}__{args.split}__{args.candidate_mode}__{selected_config_id}"

    input_paths = list(paths.values())
    output_paths = [
        out_dir / "frozen_ranker_manifest.json",
        out_dir / "frozen_ranker_summary.md",
        out_dir / "frozen_ranker_metrics.csv",
        out_dir / "frozen_ranker_files.json",
    ]
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    for path in input_paths:
        if path.as_posix().startswith("results/"):
            add_check(f"input under root: {path.as_posix()}", is_under(path, root), path.as_posix())
        add_check(f"input has no test path: {path.as_posix()}", no_test_path(path), path.as_posix())
    for path in output_paths:
        add_check(f"output under root: {path.as_posix()}", is_under(path, root), path.as_posix())
        add_check(f"output has no test path: {path.as_posix()}", no_test_path(path), path.as_posix())

    add_check("P1 consistency overall_ok", p1.get("overall_ok") is True, p1.get("overall_ok"))
    add_check("P2-0 headroom overall_ok", headroom.get("overall_ok") is True, headroom.get("overall_ok"))
    add_check("P2-1 split overall_ok", valid_split.get("overall_ok") is True, valid_split.get("overall_ok"))
    add_check(
        "P2-2 minimal ranker not frozen",
        minimal.get("selection", {}).get("freeze_recommendation", {}).get("decision") == "do_not_freeze_p2_minimal_ranker",
        minimal.get("selection", {}).get("freeze_recommendation", {}).get("decision"),
    )
    add_check("P2-3 history ranker overall_ok", history.get("overall_ok") is True, history.get("overall_ok"))
    add_check("P2-3 freeze decision", selection.get("decision") == "freeze_p2_history_ranker", selection.get("decision"))
    add_check("valid_select HR@20 gain positive", valid_select_hr20_gain > args.min_hr20_gain, valid_select_hr20_gain)
    add_check("valid_select NDCG@20 gain positive", valid_select_ndcg20_gain > args.min_ndcg20_gain, valid_select_ndcg20_gain)
    add_check("model selected_config matches report", model.get("selected_config_id") == selected_config_id, model.get("selected_config_id"))
    add_check("category matches", history.get("category") == args.category, history.get("category"))
    add_check("split valid", history.get("split") == args.split, history.get("split"))
    add_check("candidate mode exact", history.get("candidate_mode") == args.candidate_mode, history.get("candidate_mode"))

    frozen_files = {
        name: {
            "path": path.as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    metric_rows = make_metric_rows(history)
    manifest = {
        "split": args.split,
        "category": args.category,
        "candidate_mode": args.candidate_mode,
        "input_root": root.as_posix(),
        "output_dir": out_dir.as_posix(),
        "frozen_ranker": {
            "ranker_id": ranker_id,
            "ranker_type": "history_aware_linear_pairwise_logistic",
            "selected_config_id": selected_config_id,
            "model_path": paths["p2_history_model"].as_posix(),
            "report_path": paths["p2_history_report"].as_posix(),
            "feature_weights_path": paths["p2_history_feature_weights"].as_posix(),
            "feature_names": model.get("feature_names", []),
            "history_feature_names": model.get("history_feature_names", []),
            "selected_config": history.get("selected_config", {}),
            "heuristic_component_weights": model.get("heuristic_component_weights", {}),
        },
        "selection": {
            "decision": selection.get("decision"),
            "criterion": history.get("selection", {}).get("criterion"),
            "valid_select_hr20_gain": valid_select_hr20_gain,
            "valid_select_ndcg20_gain": valid_select_ndcg20_gain,
            "valid_select_history_hr20": metric(history, "valid_select", "learned_rerank", "hr@20"),
            "valid_select_heuristic_hr20": metric(history, "valid_select", "heuristic_rerank", "hr@20"),
            "valid_select_history_ndcg20": metric(history, "valid_select", "learned_rerank", "ndcg@20"),
            "valid_select_heuristic_ndcg20": metric(history, "valid_select", "heuristic_rerank", "ndcg@20"),
        },
        "metrics": {
            "valid_fit": history["split_metrics"]["valid_fit"],
            "valid_select": history["split_metrics"]["valid_select"],
            "all_valid": history["split_metrics"]["all_valid"],
        },
        "top_feature_weights": top_feature_rows(paths["p2_history_feature_weights"]),
        "frozen_files": frozen_files,
        "checks": checks,
        "overall_ok": all(check["ok"] for check in checks),
        "downstream_policy": {
            "p3_should_use_frozen_ranker": True,
            "do_not_reselect_on_test": True,
            "test_allowed_only_after_final_protocol_freeze": True,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "frozen_ranker_manifest.json", manifest)
    write_json(out_dir / "frozen_ranker_files.json", frozen_files)
    write_csv(
        out_dir / "frozen_ranker_metrics.csv",
        ["split", "model", "hr20", "ndcg20", "hr10", "ndcg10", "hr5", "ndcg5"],
        metric_rows,
    )
    with open(out_dir / "frozen_ranker_summary.md", "w", encoding="utf-8") as f:
        f.write(make_markdown(manifest, metric_rows))

    if not manifest["overall_ok"]:
        failed = [check for check in checks if not check["ok"]]
        raise SystemExit(f"P2 frozen ranker closeout failed checks: {failed}")

    print(f"Wrote frozen ranker manifest: {out_dir / 'frozen_ranker_manifest.json'}")
    print(
        "P2 frozen ranker summary: "
        f"ranker_id={ranker_id} "
        f"valid_select_hr20_gain={valid_select_hr20_gain} "
        f"valid_select_ndcg20_gain={valid_select_ndcg20_gain}"
    )


if __name__ == "__main__":
    main()
