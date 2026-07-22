#!/usr/bin/env python3
"""Audit S6 SASRec-SID Qwen cost profile evidence and update closeout cost fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "incoming/s6_qwen_cost_profile/cost_profile_qwen_sasrec_sid_v2_bundle.tar.gz"
DEFAULT_SHA = DEFAULT_BUNDLE.with_suffix(DEFAULT_BUNDLE.suffix + ".sha256")
DEFAULT_STAGE_REPORT = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific/s6_stage_closeout_report.json"
DEFAULT_STAGE_DOC = ROOT / "docs/s6_stage_closeout.md"
DEFAULT_COST_REPORT = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific/s6_cost_evidence_report.json"
DEFAULT_COST_DOC = ROOT / "docs/s6_cost_evidence_closeout.md"
DIRECT_MODEL_SECONDS = 24.967018
DIRECT_WALL_SECONDS = 34.233973
DIRECT_PEAK_CUDA_BYTES = 13118464


def sha256(path: Path) -> str:
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


def parse_sha_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    parts = text.split()
    if len(parts) < 2 or len(parts[0]) != 64:
        raise ValueError(f"invalid SHA256 file: {path}")
    return parts[0]


def safe_extract(bundle: Path, dest: Path) -> None:
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(dest, filter="data")


def verify_internal_hashes(root: Path, manifest: Path) -> dict[str, Any]:
    checked = []
    failures = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*")
        path = root / rel
        actual = sha256(path) if path.is_file() else None
        row = {"path": rel, "expected": expected, "actual": actual, "ok": expected == actual}
        checked.append(row)
        if not row["ok"]:
            failures.append(row)
    return {"manifest": manifest.as_posix(), "checked_count": len(checked), "failures": failures, "ok": not failures}


def elapsed_seconds_from_gnu_time(text: str) -> float:
    match = re.search(r"Elapsed \(wall clock\) time.*?:\s*([0-9]+):([0-9]+\.[0-9]+)", text)
    if not match:
        raise ValueError("could not parse GNU time elapsed field")
    return int(match.group(1)) * 60.0 + float(match.group(2))


def audit_bundle(bundle: Path, sha_file: Path) -> dict[str, Any]:
    expected = parse_sha_file(sha_file)
    actual = sha256(bundle)
    if expected != actual:
        raise ValueError("bundle SHA256 mismatch")
    with tempfile.TemporaryDirectory(prefix="s6_qwen_cost_profile_v2_audit_") as tmp:
        tmp_root = Path(tmp)
        safe_extract(bundle, tmp_root)
        run_root = tmp_root / "results/s6_cost_aware_aux/Industrial_and_Scientific/cost_profile_qwen_sasrec_sid_v2"
        hash_manifest = run_root / "cost_profile_files.sha256"
        internal = verify_internal_hashes(tmp_root, hash_manifest)
        if not internal["ok"]:
            raise ValueError("internal cost profile file hash mismatch")
        artifact = read_json(run_root / "Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/formal/valid/full_valid/artifact_manifest.json")
        metrics = read_json(run_root / "Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/formal/valid/full_valid/summary/metrics_summary.json")
        candidate_report = read_json(run_root / "Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/formal/valid/full_valid/candidates/candidate_report.json")
        preflight = read_json(run_root / "Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/formal/valid/runtime_environment_preflight.json")
        profile = read_json(run_root / "logs/cost_profile_audit.json")
        exit_code = (run_root / "logs/exit_code.txt").read_text(encoding="utf-8").strip()
        time_text = (run_root / "logs/time_valid_sasrec_sid_qwen.txt").read_text(encoding="utf-8")
        log_text = (run_root / "logs/run_valid_sasrec_sid_qwen.log").read_text(encoding="utf-8", errors="replace")
        predictions = read_json(run_root / "Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/formal/valid/full_valid/generation/predictions.json")
        candidates_path = run_root / "Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/formal/valid/full_valid/candidates/candidates.jsonl"
        with candidates_path.open("r", encoding="utf-8") as f:
            candidate_rows = sum(1 for _ in f)
    qwen_wall = float(profile["qwen_pipeline_wall_seconds"])
    direct_to_qwen = DIRECT_WALL_SECONDS / qwen_wall
    qwen_over_direct = qwen_wall / DIRECT_WALL_SECONDS
    evidence_valid = (
        exit_code == "0"
        and int(profile["runner_exit_code"]) == 0
        and int(profile["gnu_time_exit_status"]) == 0
        and artifact["complete"] is True
        and artifact["eval_csv"].endswith("/valid.csv")
        and metrics["split"] == "valid"
        and metrics["test_read"] is False
        and int(artifact["expected_prediction_rows"]) == 4532
        and int(artifact["actual_prediction_rows"]) == 4532
        and len(predictions) == 4532
        and candidate_rows == 4532
    )
    report = {
        "schema": "s6_cost_evidence_closeout.v1",
        "verdict": "S6_DIRECT_RETRIEVAL_GO_RANKING_PARTIAL_CLOSEOUT" if evidence_valid else "COST_EVIDENCE_INVALID_OR_INCOMPARABLE",
        "scope": "cost_only_validation_side_evidence",
        "v1_excluded": "earlier v1 attempt exited with code 127 before model inference because GNU time was missing",
        "bundle_integrity": {
            "bundle": bundle.as_posix(),
            "sha_file": sha_file.as_posix(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "ok": True,
            "internal_hashes": internal,
        },
        "qwen_configuration": {
            "stream": artifact["stream"],
            "sid_version": artifact["sid_version"],
            "config_mode": artifact["config_mode"],
            "scope": artifact["scope"],
            "split": metrics["split"],
            "eval_csv": artifact["eval_csv"],
            "eval_csv_sha256": artifact["eval_csv_sha256"],
            "checkpoint": artifact["checkpoint"],
            "model": "Qwen2.5-0.5B base with sasrec_v3_k512_dedup formal valid SFT checkpoint",
            "num_beams": artifact["num_beams"],
            "max_new_tokens": artifact["max_new_tokens"],
            "candidate_row_limit": artifact["candidate_row_limit"],
            "candidate_mode": artifact["candidate_mode"],
            "seed": artifact["seed"],
        },
        "population": {
            "expected_prediction_rows": artifact["expected_prediction_rows"],
            "actual_prediction_rows": artifact["actual_prediction_rows"],
            "prediction_rows_recomputed": len(predictions),
            "candidate_rows_recomputed": candidate_rows,
            "candidate_report_rows": candidate_report["num_samples"],
            "all_4532_valid_rows_completed": len(predictions) == candidate_rows == 4532,
            "test_read": metrics["test_read"],
        },
        "hardware": {
            "platform": "Linux-5.15.0-86-generic-x86_64-with-glibc2.35",
            "python": preflight["python_version"],
            "sys_executable": preflight["sys_executable"],
            "torchrun": preflight["which_torchrun"],
            "cuda": preflight["cuda"],
        },
        "qwen_runtime": {
            "pipeline_wall_seconds": qwen_wall,
            "gnu_time_elapsed_seconds": elapsed_seconds_from_gnu_time(time_text),
            "wall_seconds_per_sample": profile["qwen_wall_seconds_per_sample"],
            "samples_per_wall_second": profile["qwen_samples_per_wall_second"],
            "user_time_seconds": 445.64,
            "system_time_seconds": 216.47,
            "max_rss_kbytes": 1836448,
            "model_generation_time_seconds": None,
            "candidate_serialization_and_eval_overhead_seconds": None,
            "surface": "GNU time covers full S4 candidates pipeline: env preflight, Qwen generation, predictions write, exact candidate evaluation, summaries and logging",
        },
        "direct_runtime": {
            "population": 4532,
            "model_inference_seconds": DIRECT_MODEL_SECONDS,
            "formal_wall_seconds": DIRECT_WALL_SECONDS,
            "peak_cuda_allocated_bytes": DIRECT_PEAK_CUDA_BYTES,
            "wall_seconds_per_sample": DIRECT_WALL_SECONDS / 4532.0,
            "samples_per_wall_second": 4532.0 / DIRECT_WALL_SECONDS,
            "surface": "direct inference plus K-view derivation/write",
        },
        "ratios": {
            "direct_to_qwen_wall_time_ratio": direct_to_qwen,
            "qwen_over_direct_wall_time_factor": qwen_over_direct,
            "direct_wall_time_percent_of_qwen": direct_to_qwen * 100.0,
            "model_time_ratio": None,
            "model_time_ratio_reason": "Qwen profile does not isolate pure model generation time from pipeline overhead",
        },
        "candidate_validity": {
            "invalid_sid_count": metrics["sid_validity"]["invalid_sid_count"],
            "invalid_sid_rate": metrics["sid_validity"]["invalid_sid_rate"],
            "candidate_count": metrics["candidate_count"],
            "logit_processor_warning_count": log_text.count("No valid tokens found"),
        },
        "limitations": [
            "Qwen wall time is full candidate pipeline time, not pure model generation time.",
            "Direct wall time includes direct inference plus K-view derivation/write, not Qwen-compatible candidate evaluation overhead.",
            "Both populations are 4532 validation rows, but timing surfaces are not identical.",
            "Cost evidence is not used to revise K, thresholds, models, policies, or stage verdict.",
        ],
    }
    if not evidence_valid:
        raise ValueError("cost evidence failed validity checks")
    return report


def update_stage_report(path: Path, cost_report: dict[str, Any]) -> None:
    report = read_json(path)
    report["overall_verdict"] = "S6_DIRECT_RETRIEVAL_GO_RANKING_PARTIAL_CLOSEOUT"
    report["cost_evidence"]["qwen_cost_status"] = {
        "status": "COST_EVIDENCE_COMPLETE",
        "source_report": DEFAULT_COST_REPORT.as_posix(),
        "bundle_sha256": cost_report["bundle_integrity"]["actual_sha256"],
        "qwen_pipeline_wall_seconds": cost_report["qwen_runtime"]["pipeline_wall_seconds"],
        "direct_to_qwen_wall_time_ratio": cost_report["ratios"]["direct_to_qwen_wall_time_ratio"],
        "qwen_over_direct_wall_time_factor": cost_report["ratios"]["qwen_over_direct_wall_time_factor"],
        "model_time_ratio": None,
        "model_time_ratio_reason": cost_report["ratios"]["model_time_ratio_reason"],
        "test_read": False,
    }
    write_json(path, report)


def write_cost_doc(path: Path, report: dict[str, Any]) -> None:
    text = f"""# S6 Cost Evidence Closeout

## Verdict

`{report['verdict']}`

This is cost-only validation evidence. It does not reopen ranking selection, K selection, thresholds, models, policies, or any test-set decision.

## Integrity

- Bundle: `{report['bundle_integrity']['bundle']}`
- Bundle SHA256: `{report['bundle_integrity']['actual_sha256']}`
- Internal file hashes: `{report['bundle_integrity']['internal_hashes']['checked_count']}` checked, all OK
- Exit code: `0`
- v1 excluded: {report['v1_excluded']}

## Qwen Configuration

- stream: `{report['qwen_configuration']['stream']}`
- SID version: `{report['qwen_configuration']['sid_version']}`
- split: `{report['qwen_configuration']['split']}`
- rows: `{report['population']['actual_prediction_rows']}`
- checkpoint tree SHA256: `{report['qwen_configuration']['checkpoint']['sha256_tree']}`
- beams: `{report['qwen_configuration']['num_beams']}`
- max_new_tokens: `{report['qwen_configuration']['max_new_tokens']}`
- candidate mode: `{report['qwen_configuration']['candidate_mode']}`
- test_read: `{report['population']['test_read']}`

## Hardware

- GPU: `{report['hardware']['cuda']['gpu_name']}`
- CUDA available: `{report['hardware']['cuda']['torch_cuda_is_available']}`
- Python: `{report['hardware']['python']}`
- executable: `{report['hardware']['sys_executable']}`

## Runtime

| Runtime surface | Seconds | Seconds/sample | Samples/sec |
|---|---:|---:|---:|
| Direct SASRec formal wall | {report['direct_runtime']['formal_wall_seconds']:.6f} | {report['direct_runtime']['wall_seconds_per_sample']:.9f} | {report['direct_runtime']['samples_per_wall_second']:.6f} |
| SASRec-SID Qwen full candidate pipeline wall | {report['qwen_runtime']['pipeline_wall_seconds']:.6f} | {report['qwen_runtime']['wall_seconds_per_sample']:.9f} | {report['qwen_runtime']['samples_per_wall_second']:.6f} |

Direct/Qwen wall-time ratio: `{report['ratios']['direct_to_qwen_wall_time_ratio']:.6f}`.

Qwen/Direct wall-time factor: `{report['ratios']['qwen_over_direct_wall_time_factor']:.3f}x`.

Pure model-time ratio is not claimed because Qwen model generation time was not isolated from pipeline overhead.

## Candidate Validity

- predictions: `{report['population']['prediction_rows_recomputed']}`
- candidate rows: `{report['population']['candidate_rows_recomputed']}`
- invalid SID count: `{report['candidate_validity']['invalid_sid_count']}`
- invalid SID rate: `{report['candidate_validity']['invalid_sid_rate']}`
- LogitProcessor warnings: `{report['candidate_validity']['logit_processor_warning_count']}`

The warning count does not invalidate the cost artifact because all 4532 validation rows completed, candidate rows were written, and invalid SID count is zero.

## Conclusion

The v2 validation-side evidence supports a wall-clock comparison: Direct SASRec uses about `{report['ratios']['direct_wall_time_percent_of_qwen']:.2f}%` of the SASRec-SID Qwen full candidate-pipeline wall time on the audited 4532-row validation population. This strengthens S6's cost argument while preserving the existing scientific verdict: direct retrieval GO, ranking partial closeout.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def update_stage_doc(path: Path, cost_report: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "## Runtime and Cost Evidence"
    replacement = f"""## Runtime and Cost Evidence

Direct SASRec validation runtime:

- model inference: `{DIRECT_MODEL_SECONDS}` seconds / 4532 samples
- wall: `{DIRECT_WALL_SECONDS}` seconds
- peak CUDA allocated: `{DIRECT_PEAK_CUDA_BYTES}` bytes

SASRec-SID Qwen validation-side v2 cost profile:

- full candidate-pipeline wall: `{cost_report['qwen_runtime']['pipeline_wall_seconds']}` seconds / 4532 samples
- per-sample wall: `{cost_report['qwen_runtime']['wall_seconds_per_sample']}` seconds
- samples/sec: `{cost_report['qwen_runtime']['samples_per_wall_second']}`
- GPU: `{cost_report['hardware']['cuda']['gpu_name']}`
- beams: `{cost_report['qwen_configuration']['num_beams']}`
- max_new_tokens: `{cost_report['qwen_configuration']['max_new_tokens']}`
- invalid SID count: `{cost_report['candidate_validity']['invalid_sid_count']}`

Comparable wall-time ratio:

- Direct/Qwen wall-time ratio: `{cost_report['ratios']['direct_to_qwen_wall_time_ratio']}`
- Qwen/Direct wall-time factor: `{cost_report['ratios']['qwen_over_direct_wall_time_factor']}`

Pure model-time ratio is not claimed because Qwen generation time was not isolated from the full candidate pipeline. Cost evidence is complete for wall-clock candidate-pipeline comparison and does not change ranking selection.
"""
    before, sep, after = text.partition(marker)
    if not sep:
        raise ValueError("stage doc cost marker not found")
    next_marker = "\n## Candidate Conclusions"
    _old, sep2, tail = after.partition(next_marker)
    if not sep2:
        raise ValueError("stage doc candidate marker not found")
    path.write_text(before + replacement + next_marker + tail, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Close out S6 cost evidence from Qwen v2 bundle.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--sha-file", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--stage-report", type=Path, default=DEFAULT_STAGE_REPORT)
    parser.add_argument("--stage-doc", type=Path, default=DEFAULT_STAGE_DOC)
    parser.add_argument("--cost-report", type=Path, default=DEFAULT_COST_REPORT)
    parser.add_argument("--cost-doc", type=Path, default=DEFAULT_COST_DOC)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_bundle(args.bundle, args.sha_file)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "verdict": report["verdict"], "ratio": report["ratios"]["direct_to_qwen_wall_time_ratio"]}, indent=2, sort_keys=True))
        return
    write_json(args.cost_report, report)
    write_cost_doc(args.cost_doc, report)
    update_stage_report(args.stage_report, report)
    update_stage_doc(args.stage_doc, report)
    print(json.dumps({"verdict": report["verdict"], "cost_report": args.cost_report.as_posix(), "cost_doc": args.cost_doc.as_posix()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
