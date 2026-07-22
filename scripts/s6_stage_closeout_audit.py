#!/usr/bin/env python3
"""S6 validation-only stage closeout and cost-evidence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CATEGORY = "Industrial_and_Scientific"
DEFAULT_OUTPUT_ROOT = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific"
DEFAULT_REPORT = DEFAULT_OUTPUT_ROOT / "s6_stage_closeout_report.json"
DEFAULT_DOC = ROOT / "docs/s6_stage_closeout.md"
SPLITS = ("valid_fit", "valid_select", "valid_gate")
DISPLAY_SPLITS = ("valid_fit", "valid_select")
KS = (1, 5, 10, 20)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def reject_test_path(path: Path) -> None:
    lowered = [part.lower() for part in path.parts]
    if "test" in lowered or "final_test" in lowered or path.name.lower() == "test.csv":
        raise ValueError(f"S6 closeout refuses test path: {path}")


def require_file(path: Path) -> dict[str, Any]:
    reject_test_path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return {"path": path.as_posix(), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    enriched = dict(metrics)
    if "cf_preservation_rate_at20" not in enriched and enriched.get("cf_hits_at20") is not None:
        cf_hits = enriched.get("cf_hits_at20") or 0
        enriched["cf_preservation_rate_at20"] = (
            enriched.get("cf_hits_preserved_at20", 0) / cf_hits if cf_hits else 1.0
        )
    keys = [
        "num_samples",
        "hr@20",
        "ndcg@20",
        "hits@20",
        "mrr",
        "target_in_pool_count",
        "target_in_pool_rate",
        "union_target_in_pool_count",
        "union_target_in_pool_rate",
        "cf_hits_at20",
        "cf_hits_preserved_at20",
        "cf_hits_lost_at20",
        "cf_preservation_rate_at20",
        "direct_only_targets_available_at20",
        "direct_only_targets_recovered_at20",
        "direct_only_recovery_rate_at20",
        "candidate_to_top20_conversion_rate",
        "average_candidate_pool_size",
    ]
    return {key: enriched.get(key) for key in keys if key in enriched}


def oracle_from(metrics: dict[str, Any]) -> dict[str, Any]:
    count = int(metrics["union_target_in_pool_count"])
    total = int(metrics["num_samples"])
    return {
        "num_samples": total,
        "hr@20": count / total if total else 0.0,
        "ndcg@20": None,
        "hits@20": count,
        "mrr": None,
        "target_in_pool_count": count,
        "target_in_pool_rate": count / total if total else 0.0,
        "cf_hits_at20": metrics.get("cf_hits_at20"),
        "cf_hits_preserved_at20": metrics.get("cf_hits_at20"),
        "cf_hits_lost_at20": 0,
        "cf_preservation_rate_at20": 1.0,
        "direct_only_targets_available_at20": metrics.get("direct_only_targets_available_at20"),
        "direct_only_targets_recovered_at20": metrics.get("direct_only_targets_available_at20"),
        "direct_only_recovery_rate_at20": 1.0,
        "protocol_gate_status": "candidate_oracle_not_ranker",
    }


def by_config(report: dict[str, Any], config_id: str) -> dict[str, Any]:
    for row in report["fit_select_results"]:
        if row["config"]["config_id"] == config_id:
            return row
    raise KeyError(config_id)


def method_rows(s6_5: dict[str, Any], s6_6: dict[str, Any], s6_6r: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, source in [
        ("CF-only", "cf_only"),
        ("Direct SASRec direct order", "direct_order"),
        ("Fixed RRF", "fixed_rrf"),
    ]:
        item = {"method": method, "source": "s6_5", "gate_status": "baseline"}
        for split in DISPLAY_SPLITS:
            item[split] = compact_metrics(s6_5["per_split"][split]["baselines"][source])
        item["aggregate"] = compact_metrics(s6_5["aggregate"][source])
        rows.append(item)

    oracle = {"method": "CF + direct K20 candidate oracle", "source": "s6_5_union_pool", "gate_status": "candidate_layer_only"}
    for split in DISPLAY_SPLITS:
        oracle[split] = oracle_from(s6_5["per_split"][split]["baselines"]["cf_only"])
    oracle["aggregate"] = oracle_from(s6_5["aggregate"]["cf_only"])
    rows.append(oracle)

    projected = {"method": "S6-5 frozen projected ranker", "source": "s6_5", "gate_status": "best_validated_ranking_but_valid_select_cf_preservation_failed"}
    for split in DISPLAY_SPLITS:
        projected[split] = compact_metrics(s6_5["per_split"][split]["projected_ranker"])
    projected["valid_gate"] = compact_metrics(s6_5["per_split"]["valid_gate"]["projected_ranker"])
    projected["aggregate"] = compact_metrics(s6_5["aggregate"]["projected_ranker"])
    rows.append(projected)

    for name, cfg in [
        ("S6-6 aggressive residual", "alpha1_l20.001_clip0.5_cfq0"),
        ("S6-6 preservation residual", "alpha0.1_l20.001_clip0.25_cfq18"),
    ]:
        found = by_config(s6_6, cfg)
        item = {"method": name, "source": "s6_6", "config_id": cfg, "gate_status": "not_selected_valid_gate_unopened"}
        for split in DISPLAY_SPLITS:
            item[split] = compact_metrics(found["metrics"][split])
        item["aggregate"] = None
        rows.append(item)

    for name, cfg in [
        ("S6-6R best-HR promotion policy", "mp3_thr0.800224"),
        ("S6-6R best-preservation policy", "mp2_thr0.958302"),
    ]:
        found = by_config(s6_6r, cfg)
        item = {"method": name, "source": "s6_6r", "config_id": cfg, "gate_status": "not_selected_valid_gate_unopened"}
        for split in DISPLAY_SPLITS:
            item[split] = compact_metrics(found["metrics"][split])
        item["aggregate"] = None
        rows.append(item)
    return rows


def comparable_qwen_cost_status() -> dict[str, Any]:
    formal_root = ROOT / "incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036"
    manifest = formal_root / "results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/formal/valid/full_valid/artifact_manifest.json"
    metrics = formal_root / "results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/formal/valid/full_valid/summary/metrics_summary.json"
    found_manifest = manifest.is_file()
    found_metrics = metrics.is_file()
    timing_fields: dict[str, Any] = {}
    if found_metrics:
        metrics_data = read_json(metrics)
        timing_fields = {key: value for key, value in metrics_data.items() if any(token in key.lower() for token in ("time", "second", "runtime", "wall", "latency"))}
    return {
        "status": "COST_EVIDENCE_PENDING",
        "comparable_evidence_found": False,
        "formal_sasrec_sid_manifest_present": found_manifest,
        "formal_sasrec_sid_metrics_present": found_metrics,
        "formal_sasrec_sid_timing_fields": timing_fields,
        "reason": "formal SASRec-SID valid artifacts record rows, beam, max_new_tokens and checkpoint, but no audited model-time or wall-time fields",
        "frozen_model_config_input": {
            "stream": "treatment",
            "sid_version": "sasrec_v3_k512_dedup",
            "checkpoint": "outputs/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/formal/valid/sft/final_checkpoint",
            "eval_csv": "data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/valid.csv",
            "rows": 4532,
            "num_beams": 50,
            "max_new_tokens": 6,
        },
        "required_fields": [
            "host",
            "gpu_name",
            "python_env",
            "model_checkpoint",
            "eval_csv_sha256",
            "prediction_rows",
            "candidate_rows",
            "num_beams",
            "max_new_tokens",
            "model_generation_wall_seconds",
            "candidate_eval_wall_seconds",
            "total_wall_seconds",
            "peak_cuda_allocated_bytes",
        ],
        "autodl_profiling_command": (
            "cd /root/autodl-tmp/projects/MiniOneRec && "
            "mkdir -p results/s6_cost_aware_aux/Industrial_and_Scientific/cost_profile_qwen_sasrec_sid_v1/logs && "
            "/usr/bin/time -v -o results/s6_cost_aware_aux/Industrial_and_Scientific/cost_profile_qwen_sasrec_sid_v1/logs/time_valid_sasrec_sid_qwen.txt "
            "env BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B "
            "STAGE=candidates TRAIN_STREAM=treatment CONFIG_MODE=formal DRY_RUN=0 "
            "RESULTS_ROOT=results/s6_cost_aware_aux/Industrial_and_Scientific/cost_profile_qwen_sasrec_sid_v1 "
            "OUTPUTS_ROOT=outputs/s4_sasrec_sid_valid "
            "CANDIDATE_ROW_LIMIT=0 NUM_BEAMS=50 MAX_NEW_TOKENS=6 OVERWRITE=0 REUSE_COMPLETE=0 "
            "bash scripts/run_s4_sasrec_sid_valid.sh "
            "2>&1 | tee results/s6_cost_aware_aux/Industrial_and_Scientific/cost_profile_qwen_sasrec_sid_v1/logs/run_valid_sasrec_sid_qwen.log"
        ),
    }


def artifact_inventory(output_root: Path) -> dict[str, Any]:
    paths = {
        "checkpoint_compatibility": output_root / "s6_checkpoint_compatibility_report.json",
        "validation_split": output_root / "s6_validation_split_manifest.json",
        "smoke_determinism": output_root / "s6_2_smoke_audit_report.json",
        "formal_direct": output_root / "s6_3_formal_direct_sasrec_audit_report.json",
        "union_validation": output_root / "s6_4_union_validation_report.json",
        "frozen_projected_ranker": output_root / "s6_5_frozen_ranker_validation_report.json",
        "residual_ranker_negative": output_root / "s6_6_lightweight_ranker_validation_report.json",
        "promotion_gate_negative": output_root / "s6_6r_direct_promotion_gate_report.json",
        "frozen_projection_manifest": output_root / "frozen_ranker/frozen_ranker_v1/projection_manifest.json",
        "s6_6_policy_grid": output_root / "lightweight_ranker/lightweight_v1/predeclared_config_grid.json",
        "s6_6r_policy_grid": output_root / "direct_promotion/promotion_v1/predeclared_policy_grid.json",
    }
    return {name: require_file(path) for name, path in paths.items()}


def write_doc(path: Path, report: dict[str, Any]) -> None:
    metric_lines = []
    for row in report["final_metric_table"]:
        vf = row.get("valid_fit") or {}
        vs = row.get("valid_select") or {}
        metric_lines.append(
            "| {method} | {gate} | {vf_hr} | {vf_ndcg} | {vf_cf} | {vf_dr} | {vs_hr} | {vs_ndcg} | {vs_cf} | {vs_dr} |".format(
                method=row["method"],
                gate=row["gate_status"],
                vf_hr=fmt(vf.get("hr@20")),
                vf_ndcg=fmt(vf.get("ndcg@20")),
                vf_cf=fmt(vf.get("cf_preservation_rate_at20")),
                vf_dr=fmt(vf.get("direct_only_targets_recovered_at20")),
                vs_hr=fmt(vs.get("hr@20")),
                vs_ndcg=fmt(vs.get("ndcg@20")),
                vs_cf=fmt(vs.get("cf_preservation_rate_at20")),
                vs_dr=fmt(vs.get("direct_only_targets_recovered_at20")),
            )
        )
    text = f"""# S6 Stage Closeout

## Verdict

`{report['overall_verdict']}`

S6 closes with direct retrieval and candidate-level complementarity validated, while ranking conversion remains partial. S6-5 frozen projected ranker is the strongest validated ranking result, but it does not pass the strict 97% CF-preservation gate on `valid_select`. S6-6 and S6-6R are retained as negative evidence, and no further S6 ranker tuning is allowed.

## Motivation

S6 tested whether a cheap direct item-level SASRec retrieval stream can replace or augment the expensive SASRec-SID Qwen generation stream. The stage separated retrieval/candidate evidence from ranking evidence so that candidate success would not be mistaken for production-ready ranking success.

## Protocol

- Validation-only; no test data.
- K selected from validation candidate evidence and frozen at K20.
- `valid_fit` trains/calibrates lightweight revisions.
- `valid_select` selects one final configuration when allowed.
- `valid_gate` is opened only after a configuration passes fit/select.
- S6-6 and S6-6R did not pass fit/select, so their `valid_gate` remained unopened.

## Stage Timeline

- S6-0 checkpoint and protocol audit: `{report['stage_timeline']['s6_0']}`.
- S6-1 direct exporter implementation.
- S6-2 smoke and deterministic reproduction: `{report['stage_timeline']['s6_2']}`.
- S6-3 formal direct-SASRec validation: `{report['stage_timeline']['s6_3']}`.
- S6-4 CF + direct union validation: `{report['stage_timeline']['s6_4']}`.
- S6-5 frozen projected-ranker validation: `{report['stage_timeline']['s6_5']}`.
- S6-6 residual ranker revision: `{report['stage_timeline']['s6_6']}`.
- S6-6R direct-promotion revision: `{report['stage_timeline']['s6_6r']}`.

## Final Metric Table

| Method | Gate status | fit HR@20 | fit NDCG@20 | fit CF preserve | fit direct recovery | select HR@20 | select NDCG@20 | select CF preserve | select direct recovery |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

## Runtime and Cost Evidence

Direct SASRec validation runtime:

- model inference: `{report['cost_evidence']['direct_runtime']['model_inference_seconds']}` seconds / 4532 samples
- wall: `{report['cost_evidence']['direct_runtime']['formal_wall_seconds']}` seconds
- peak CUDA allocated: `{report['cost_evidence']['direct_runtime']['peak_cuda_allocated_bytes']}` bytes

Comparable SASRec-SID Qwen validation timing status: `{report['cost_evidence']['qwen_cost_status']['status']}`.

Exact profiling command is recorded in the JSON report and must be run only as cost evidence. It must not alter model selection.

## Candidate Conclusions

Direct SASRec retrieval is validated as a candidate source. K20 was selected as the smallest candidate budget satisfying validation candidate uplift gates. Direct retrieval adds target recall beyond CF while preserving CF candidate pool membership in the union.

## Ranking Conclusions

Fixed RRF is insufficient. S6-5 frozen projected ranker converts auxiliary recall into ranking gains, but its `valid_select` CF preservation is `91/95 = 95.79%`, below the required 97%. Therefore S6 ranking is partial, not a production GO.

## Failed Alternatives

S6-6 residual ranking showed a sharp preservation-versus-recovery trade-off. S6-6R CF-anchored promotion preserved CF under conservative policies but lost too much direct gain; aggressive promotion recovered more direct targets but violated CF preservation. Both are retained as negative evidence.

## Limitations

- Comparable SASRec-SID Qwen validation-side runtime evidence is pending.
- No S6 method is authorized to use test data for tuning.
- No additional thresholds, model families, or ranker configurations may be added under S6.

## Deployment Recommendation

Retain direct SASRec as a validated auxiliary candidate generator. Do not deploy a replacement ranking policy from S6 as production-ready under the strict CF-preservation requirement. If deployed experimentally, S6-5 should be labeled as a partial ranking result with known CF-preservation limitation.

## Future Research

- Cost-only profiling for frozen SASRec-SID Qwen valid generation.
- New validation-only ranker research with explicit CF-preservation regularization, not S6 continuation.
- Independent future test policy after a new validation protocol is frozen.

## Boundary

S6 is closed without test access and without further ranker tuning. S6 negative experiments remain first-class evidence.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_report(output_root: Path) -> dict[str, Any]:
    inventory = artifact_inventory(output_root)
    s6_0 = read_json(output_root / "s6_checkpoint_compatibility_report.json")
    s6_2 = read_json(output_root / "s6_2_smoke_audit_report.json")
    s6_3 = read_json(output_root / "s6_3_formal_direct_sasrec_audit_report.json")
    s6_4 = read_json(output_root / "s6_4_union_validation_report.json")
    s6_5 = read_json(output_root / "s6_5_frozen_ranker_validation_report.json")
    s6_6 = read_json(output_root / "s6_6_lightweight_ranker_validation_report.json")
    s6_6r = read_json(output_root / "s6_6r_direct_promotion_gate_report.json")
    final_metric_table = method_rows(s6_5, s6_6, s6_6r)
    qwen_cost = comparable_qwen_cost_status()
    report = {
        "schema": "s6_stage_closeout_report.v1",
        "overall_verdict": "S6_DIRECT_RETRIEVAL_GO_RANKING_PARTIAL_CLOSEOUT",
        "test_read": False,
        "artifact_inventory": inventory,
        "stage_timeline": {
            "s6_0": s6_0["verdict"],
            "s6_2": s6_2["verdict"],
            "s6_3": s6_3["verdict"],
            "s6_4": s6_4["verdict"],
            "s6_5": s6_5["verdict"],
            "s6_6": s6_6["verdict"],
            "s6_6r": s6_6r["verdict"],
        },
        "valid_gate_isolation": {
            "s6_5": "opened by protocol after frozen projected validation",
            "s6_6": "not opened; no fit/select configuration passed",
            "s6_6r": "not opened; no fit/select policy passed",
        },
        "final_metric_table": final_metric_table,
        "candidate_layer_conclusion": {
            "direct_retrieval_validated": True,
            "k20_selected": True,
            "aggregate_union_target_in_pool": s6_5["aggregate"]["cf_only"]["union_target_in_pool_count"],
            "aggregate_cf_target_in_pool": s6_5["aggregate"]["cf_only"]["target_in_pool_count"],
            "added_targets_beyond_cf": s6_5["aggregate"]["cf_only"]["union_target_in_pool_count"] - s6_5["aggregate"]["cf_only"]["target_in_pool_count"],
            "direct_order_hits_at20": s6_5["aggregate"]["direct_order"]["hits@20"],
        },
        "ranking_layer_conclusion": {
            "strongest_validated_method": "S6-5 frozen projected ranker",
            "valid_select_cf_preservation": "91/95 = 95.79%",
            "required_cf_preservation": ">=97%",
            "ranking_status": "partial",
            "further_s6_ranker_tuning": "stopped",
        },
        "negative_evidence": {
            "s6_6": {
                "verdict": s6_6["verdict"],
                "selected": s6_6["selected"],
                "valid_gate_opened": False,
            },
            "s6_6r": {
                "verdict": s6_6r["verdict"],
                "selected": s6_6r["selected"],
                "valid_gate_opened": s6_6r["valid_gate"]["opened"],
            },
        },
        "cost_evidence": {
            "direct_runtime": {
                "model_inference_seconds": 24.967018,
                "formal_wall_seconds": 34.233973,
                "peak_cuda_allocated_bytes": 13118464,
                "samples": 4532,
            },
            "qwen_cost_status": qwen_cost,
        },
        "retained_canonical_artifacts": [
            "formal-v3 direct SASRec checkpoint compatibility report",
            "direct candidate exporter",
            "formal direct K20 candidates",
            "K20 union_v1 candidates",
            "S6-5 frozen projected outputs",
            "S6-6/S6-6R negative evidence",
            "S6 stage closeout report",
        ],
        "ablation_only_artifacts": [
            "K50/K100 union views",
            "non-selected S6-6 residual configurations",
            "non-selected S6-6R promotion policies",
        ],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build S6 stage closeout report.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.output_root, args.report, args.doc]:
        reject_test_path(path)
    report = build_report(args.output_root)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "verdict": report["overall_verdict"], "test_read": False}, indent=2, sort_keys=True))
        return
    write_json(args.report, report)
    write_doc(args.doc, report)
    print(json.dumps({"verdict": report["overall_verdict"], "report": args.report.as_posix(), "doc": args.doc.as_posix()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
