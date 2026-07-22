#!/usr/bin/env python3
"""S4 Strong Behavior-SID valid-only experiment guardrail and command planner.

This module intentionally does not train a model. It validates the frozen SID
inputs, writes parity/preflight manifests, and emits isolated AutoDL commands
for SFT, valid-only generation, and exact candidate evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils_sid import normalize_sid, parse_sid_tokens


EXPECTED_CATEGORY = "Industrial_and_Scientific"
EXPECTED_TRAIN_ROWS = 36259
EXPECTED_VALID_ROWS = 4532
EXPECTED_NUM_ITEMS = 3686
EXPECTED_SPLIT = "valid"
DEFAULT_BASELINE_VERSION = "cf_k512_dedup"
DEFAULT_TREATMENT_VERSION = "sasrec_v3_k512_dedup"
DEFAULT_TOPK = [1, 5, 10, 20]
DEFAULT_RESULTS_ROOT = Path("results/s4_sasrec_sid_valid")
DEFAULT_OUTPUTS_ROOT = Path("outputs/s4_sasrec_sid_valid")
DEFAULT_SMOKE_SAMPLE = 2000
DEFAULT_FORMAL_SAMPLE = 30000
DEFAULT_MAX_NEW_TOKENS = 6


@dataclass(frozen=True)
class SidVersionPaths:
    version: str
    root: Path
    train_csv: Path
    valid_csv: Path
    item2sid: Path
    sid2items: Path
    valid_sid_set: Path
    index_json: Path
    info_txt: Path
    reports_dir: Path


@dataclass(frozen=True)
class S4Config:
    category: str
    seed: int
    config_mode: str
    base_model: Path
    baseline: SidVersionPaths
    treatment: SidVersionPaths
    results_dir: Path
    output_dir: Path
    prediction_file: Path
    candidates_jsonl: Path
    candidate_report: Path
    num_gpus: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: str
    cutoff_len: int
    num_train_epochs: str
    bf16: str
    sample: int
    gradient_checkpointing: str
    num_beams: int
    max_new_tokens: int
    length_penalty: str
    max_pred_sids: int
    max_candidates: int
    dry_run: bool
    allow_missing_base_model: bool


@dataclass(frozen=True)
class StreamArtifacts:
    stream: str
    scope: str
    sid: SidVersionPaths
    results_dir: Path
    output_dir: Path
    checkpoint_dir: Path
    generation_dir: Path
    candidates_dir: Path
    summary_dir: Path
    prediction_file: Path
    candidates_jsonl: Path
    candidate_report: Path
    metrics_summary: Path
    artifact_manifest: Path
    eval_csv: Path


def mode_defaults(config_mode: str) -> dict[str, Any]:
    if config_mode == "smoke":
        return {
            "sample": DEFAULT_SMOKE_SAMPLE,
            "num_train_epochs": "1",
            "smoke_is_bounded": True,
            "budget_name": "sid_only_smoke",
        }
    if config_mode == "formal":
        return {
            "sample": DEFAULT_FORMAL_SAMPLE,
            "num_train_epochs": "1",
            "smoke_is_bounded": False,
            "budget_name": "sid_only_formal_30k",
        }
    raise ValueError(f"Unknown config_mode: {config_mode}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_fingerprint(path: Path) -> dict[str, Any]:
    try:
        exists = path.exists()
    except OSError as exc:
        return {
            "exists": False,
            "path": path.as_posix(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not exists:
        return {"exists": False, "sha256": None, "path": path.as_posix()}
    if path.is_file():
        return {
            "exists": True,
            "type": "file",
            "path": path.as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    files = sorted(p for p in path.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for file_path in files:
        rel = file_path.relative_to(path).as_posix()
        h.update(rel.encode("utf-8") + b"\0")
        h.update(sha256(file_path).encode("ascii") + b"\0")
    return {
        "exists": True,
        "type": "directory",
        "path": path.as_posix(),
        "file_count": len(files),
        "sha256_tree": h.hexdigest(),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def module_probe(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {
            "available": True,
            "version": str(getattr(module, "__version__", "unknown")),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def import_symbol_probe(module_name: str, symbol_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        getattr(module, symbol_name)
        return {"available": True}
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_runtime_environment() -> dict[str, Any]:
    report: dict[str, Any] = {
        "sys_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "which_torchrun": shutil.which("torchrun"),
        "modules": {},
        "imports": {},
        "cuda": {},
        "ok": True,
        "failures": [],
        "recommendation": "",
    }
    for module_name in ["numpy", "torch", "transformers", "accelerate", "peft"]:
        report["modules"][module_name] = module_probe(module_name)

    report["imports"]["transformers.AutoModelForCausalLM"] = import_symbol_probe(
        "transformers", "AutoModelForCausalLM"
    )
    report["imports"]["transformers.AutoTokenizer"] = import_symbol_probe(
        "transformers", "AutoTokenizer"
    )
    report["imports"]["transformers.generation.utils.GenerationMixin"] = import_symbol_probe(
        "transformers.generation.utils", "GenerationMixin"
    )

    torch_module = None
    try:
        torch_module = importlib.import_module("torch")
        cuda_available = bool(torch_module.cuda.is_available())
        report["cuda"]["torch_cuda_is_available"] = cuda_available
        report["cuda"]["device_count"] = int(torch_module.cuda.device_count()) if cuda_available else 0
        report["cuda"]["gpu_name"] = (
            torch_module.cuda.get_device_name(0)
            if cuda_available and torch_module.cuda.device_count() > 0
            else None
        )
    except Exception as exc:
        report["cuda"]["error"] = f"{type(exc).__name__}: {exc}"

    if not report["which_torchrun"]:
        report["failures"].append("torchrun executable not found on PATH")
    for module_name, info in report["modules"].items():
        if not info.get("available"):
            report["failures"].append(f"module import failed: {module_name}: {info.get('error')}")
    for symbol_name, info in report["imports"].items():
        if not info.get("available"):
            report["failures"].append(f"symbol import failed: {symbol_name}: {info.get('error')}")

    report["ok"] = not report["failures"]
    if not report["ok"]:
        report["recommendation"] = (
            "Failing before torchrun. Activate the intended AutoDL training environment; "
            "do not run S4 smoke from the base Python environment."
        )
    return report


def assert_runtime_environment_ok(report: dict[str, Any]) -> None:
    if report.get("ok"):
        return
    details = "\n".join(f"- {failure}" for failure in report.get("failures", []))
    raise RuntimeError(
        "S4 runtime environment preflight failed before torchrun.\n"
        f"{details}\n"
        "Do not use the base Python environment; activate the intended training environment."
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def reject_test_path(path: Path, label: str) -> None:
    text = path.as_posix().lower()
    name = path.name.lower()
    if "/test/" in text or name == "test.csv" or "test_" in name or "_test" in name:
        raise ValueError(f"{label} must not reference test data: {path}")


def require_valid_path(path: Path, label: str) -> None:
    reject_test_path(path, label)
    text = path.as_posix().lower()
    if "valid" not in text:
        raise ValueError(f"{label} must explicitly reference valid split: {path}")


def resolve_sid_paths(sid_root: Path, version: str, category: str) -> SidVersionPaths:
    root = sid_root / version / category
    return SidVersionPaths(
        version=version,
        root=root,
        train_csv=root / "train.csv",
        valid_csv=root / "valid.csv",
        item2sid=root / "item2sid.json",
        sid2items=root / "sid2items.json",
        valid_sid_set=root / "valid_sid_set.json",
        index_json=root / "index.json",
        info_txt=root / "info.txt",
        reports_dir=root / "reports",
    )


def sid_for_stream(config: S4Config, stream: str) -> SidVersionPaths:
    if stream == "baseline":
        return config.baseline
    if stream == "treatment":
        return config.treatment
    raise ValueError(f"stream must be baseline or treatment, got: {stream}")


def candidate_scope(candidate_row_limit: int) -> str:
    if candidate_row_limit < 0:
        raise ValueError(f"candidate_row_limit must be >= 0, got: {candidate_row_limit}")
    return f"preflight_rows{candidate_row_limit}" if candidate_row_limit > 0 else "full_valid"


def stream_artifacts(
    config: S4Config,
    stream: str,
    eval_csv: Path | None = None,
    candidate_row_limit: int = 0,
) -> StreamArtifacts:
    sid = sid_for_stream(config, stream)
    scope = candidate_scope(candidate_row_limit)
    stream_results_dir = replace_path_part(config.results_dir, config.treatment.version, sid.version)
    results_dir = stream_results_dir / scope
    output_dir = replace_path_part(config.output_dir, config.treatment.version, sid.version)
    generation_dir = results_dir / "generation"
    candidates_dir = results_dir / "candidates"
    summary_dir = results_dir / "summary"
    return StreamArtifacts(
        stream=stream,
        scope=scope,
        sid=sid,
        results_dir=results_dir,
        output_dir=output_dir,
        checkpoint_dir=output_dir / "final_checkpoint",
        generation_dir=generation_dir,
        candidates_dir=candidates_dir,
        summary_dir=summary_dir,
        prediction_file=generation_dir / "predictions.json",
        candidates_jsonl=candidates_dir / "candidates.jsonl",
        candidate_report=candidates_dir / "candidate_report.json",
        metrics_summary=summary_dir / "metrics_summary.json",
        artifact_manifest=results_dir / "artifact_manifest.json",
        eval_csv=eval_csv or sid.valid_csv,
    )


def load_sid_mapping(path: Path) -> dict[str, str]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise TypeError(f"item2sid must be a JSON object: {path}")
    return {str(item_id): normalize_sid(str(sid)) for item_id, sid in raw.items()}


def load_index_token_set(path: Path) -> set[str]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise TypeError(f"index must be a JSON object: {path}")
    tokens: set[str] = set()
    for item_id, sid_value in raw.items():
        if not isinstance(sid_value, list):
            raise ValueError(f"index SID for item {item_id} must be a token list")
        for token in sid_value:
            token = str(token)
            if not parse_sid_tokens(token):
                raise ValueError(f"Malformed SID token in {path}: {token}")
            tokens.add(token)
    return tokens


def validate_sid_version(paths: SidVersionPaths, expected_train_rows: int, expected_valid_rows: int) -> dict[str, Any]:
    required_files = [
        paths.train_csv,
        paths.valid_csv,
        paths.item2sid,
        paths.sid2items,
        paths.valid_sid_set,
        paths.index_json,
        paths.info_txt,
    ]
    missing = [p.as_posix() for p in required_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"SID version {paths.version} missing required files: {missing}")
    for label, path in [("train_csv", paths.train_csv), ("valid_csv", paths.valid_csv)]:
        reject_test_path(path, f"{paths.version}.{label}")
    require_valid_path(paths.valid_csv, f"{paths.version}.valid_csv")

    train_rows = csv_row_count(paths.train_csv)
    valid_rows = csv_row_count(paths.valid_csv)
    if train_rows != expected_train_rows:
        raise ValueError(f"{paths.version} train rows mismatch: {train_rows} != {expected_train_rows}")
    if valid_rows != expected_valid_rows:
        raise ValueError(f"{paths.version} valid rows mismatch: {valid_rows} != {expected_valid_rows}")

    item2sid = load_sid_mapping(paths.item2sid)
    if len(item2sid) != EXPECTED_NUM_ITEMS:
        raise ValueError(f"{paths.version} item2sid size mismatch: {len(item2sid)} != {EXPECTED_NUM_ITEMS}")
    bad_sids = [sid for sid in item2sid.values() if not parse_sid_tokens(sid)]
    if bad_sids:
        raise ValueError(f"{paths.version} contains malformed SIDs, sample={bad_sids[:3]}")
    sid_values = list(item2sid.values())
    if len(set(sid_values)) != len(sid_values):
        raise ValueError(f"{paths.version} full SID mapping is not unique")

    index_tokens = load_index_token_set(paths.index_json)
    d_tokens = [token for token in index_tokens if token.startswith("<d_")]
    if not d_tokens:
        raise ValueError(f"{paths.version} index has no <d_i> tokens; dedup token integrity cannot be checked")

    return {
        "version": paths.version,
        "root": paths.root.as_posix(),
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "num_items": len(item2sid),
        "unique_full_sids": len(set(sid_values)),
        "sid_parse_rate": 1.0,
        "token_count": len(index_tokens),
        "d_token_count": len(d_tokens),
        "hashes": {
            "train_csv": sha256(paths.train_csv),
            "valid_csv": sha256(paths.valid_csv),
            "item2sid": sha256(paths.item2sid),
            "sid2items": sha256(paths.sid2items),
            "valid_sid_set": sha256(paths.valid_sid_set),
            "index_json": sha256(paths.index_json),
            "info_txt": sha256(paths.info_txt),
        },
    }


def validate_output_isolation(config: S4Config) -> None:
    forbidden_fragments = [
        "cf_k512_dedup",
        "stage7_validation_protocol",
        "candidates_Industrial_and_Scientific_text",
    ]
    for label, path in [
        ("results_dir", config.results_dir),
        ("output_dir", config.output_dir),
        ("prediction_file", config.prediction_file),
        ("candidates_jsonl", config.candidates_jsonl),
        ("candidate_report", config.candidate_report),
    ]:
        text = path.as_posix()
        if config.treatment.version not in text:
            raise ValueError(f"{label} must include treatment SID version for isolation: {path}")
        if EXPECTED_SPLIT not in text:
            raise ValueError(f"{label} must include valid split for leakage visibility: {path}")
        if any(fragment in text for fragment in forbidden_fragments):
            raise ValueError(f"{label} points at a forbidden old artifact namespace: {path}")


def validate_stream_output_isolation(artifacts: StreamArtifacts) -> None:
    expected = artifacts.sid.version
    for label, path in [
        ("results_dir", artifacts.results_dir),
        ("output_dir", artifacts.output_dir),
        ("prediction_file", artifacts.prediction_file),
        ("candidates_jsonl", artifacts.candidates_jsonl),
        ("candidate_report", artifacts.candidate_report),
        ("metrics_summary", artifacts.metrics_summary),
        ("artifact_manifest", artifacts.artifact_manifest),
    ]:
        text = path.as_posix()
        if expected not in text:
            raise ValueError(f"{artifacts.stream}.{label} must include {expected}: {path}")
        if EXPECTED_SPLIT not in text:
            raise ValueError(f"{artifacts.stream}.{label} must include valid split: {path}")
    if artifacts.stream == "baseline" and DEFAULT_TREATMENT_VERSION in artifacts.results_dir.as_posix():
        raise ValueError(f"baseline output must not use treatment result dir: {artifacts.results_dir}")
    if artifacts.stream == "treatment" and DEFAULT_BASELINE_VERSION in artifacts.output_dir.as_posix():
        raise ValueError(f"treatment output must not use baseline output dir: {artifacts.output_dir}")


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], text=True).splitlines()
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        return {"branch": branch, "commit": commit, "dirty": bool(status), "status_short": status}
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {"error": str(exc)}


def build_train_command(config: S4Config, sid: SidVersionPaths, output_dir: Path) -> list[str]:
    return [
        "bash",
        "scripts/run_sft_ddp.sh",
        "--category",
        config.category,
        "--base-model",
        config.base_model.as_posix(),
        "--train-csv",
        sid.train_csv.as_posix(),
        "--valid-csv",
        sid.valid_csv.as_posix(),
        "--sid-index-path",
        sid.index_json.as_posix(),
        "--item-meta-path",
        f"data/Amazon/index/{config.category}.item.json",
        "--output-dir",
        output_dir.as_posix(),
        "--num-gpus",
        str(config.num_gpus),
        "--per-device-train-batch-size",
        str(config.per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--learning-rate",
        config.learning_rate,
        "--cutoff-len",
        str(config.cutoff_len),
        "--num-train-epochs",
        config.num_train_epochs,
        "--bf16",
        config.bf16,
        "--sample",
        str(config.sample),
        "--gradient-checkpointing",
        config.gradient_checkpointing,
    ]


def build_generation_command(config: S4Config, artifacts: StreamArtifacts) -> list[str]:
    if config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0 for S4 generation command")
    return [
        "python3",
        "evaluate.py",
        "--base_model",
        artifacts.checkpoint_dir.as_posix(),
        "--train_file",
        artifacts.sid.train_csv.as_posix(),
        "--info_file",
        artifacts.sid.info_txt.as_posix(),
        "--category",
        config.category,
        "--test_data_path",
        artifacts.eval_csv.as_posix(),
        "--result_json_data",
        artifacts.prediction_file.as_posix(),
        "--batch_size",
        "4",
        "--seed",
        str(config.seed),
        "--length_penalty",
        config.length_penalty,
        "--max_new_tokens",
        str(config.max_new_tokens),
        "--num_beams",
        str(config.num_beams),
    ]


def build_candidate_eval_command(config: S4Config, artifacts: StreamArtifacts) -> list[str]:
    cmd = [
        "python3",
        "evaluate_candidates.py",
        "--prediction-file",
        artifacts.prediction_file.as_posix(),
        "--eval-csv",
        artifacts.eval_csv.as_posix(),
        "--eval-split",
        EXPECTED_SPLIT,
        "--item2sid",
        artifacts.sid.item2sid.as_posix(),
        "--sid2items",
        artifacts.sid.sid2items.as_posix(),
        "--valid-sid-set",
        artifacts.sid.valid_sid_set.as_posix(),
        "--output-jsonl",
        artifacts.candidates_jsonl.as_posix(),
        "--output-report",
        artifacts.candidate_report.as_posix(),
        "--topk",
    ]
    cmd.extend(str(k) for k in DEFAULT_TOPK)
    cmd.extend([
        "--max-pred-sids",
        str(config.max_pred_sids),
        "--max-candidates",
        str(config.max_candidates),
    ])
    return cmd


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def build_command_plan(config: S4Config) -> dict[str, Any]:
    baseline_artifacts = stream_artifacts(config, "baseline")
    treatment_artifacts = stream_artifacts(config, "treatment")
    return {
        "baseline_train": build_train_command(config, config.baseline, baseline_artifacts.output_dir),
        "treatment_train": build_train_command(config, config.treatment, config.output_dir),
        "baseline_valid_generation": build_generation_command(config, baseline_artifacts),
        "baseline_exact_candidate_eval": build_candidate_eval_command(config, baseline_artifacts),
        "baseline_summary": build_summary_command(config, "baseline"),
        "treatment_valid_generation": build_generation_command(config, treatment_artifacts),
        "treatment_exact_candidate_eval": build_candidate_eval_command(config, treatment_artifacts),
        "treatment_summary": build_summary_command(config, "treatment"),
    }


def replace_path_part(path: Path, old: str, new: str) -> Path:
    parts = list(path.parts)
    try:
        idx = parts.index(old)
    except ValueError as exc:
        raise ValueError(f"Path does not contain expected component {old!r}: {path}") from exc
    parts[idx] = new
    return Path(*parts)


def summarize_candidate_report(report_path: Path, comparison_out: Path | None = None) -> dict[str, Any]:
    report = read_json(report_path)
    ranks: list[int | None] = []
    # If the report was generated by evaluate_candidates.py it does not store MRR directly.
    # Recompute MRR from the paired JSONL when the path is available.
    jsonl_path = Path(report.get("output_jsonl", ""))
    if jsonl_path.is_file():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ranks.append(json.loads(line)["candidate_hit_rank_0_based"])
    mrr = 0.0
    if ranks:
        mrr = sum(0.0 if rank is None else 1.0 / (rank + 1) for rank in ranks) / len(ranks)
    summary = {
        "source_report": report_path.as_posix(),
        "split": report.get("inputs", {}).get("eval_split"),
        "num_samples": report.get("num_samples"),
        "item_level_before_rerank": report.get("item_level_before_rerank", {}),
        "mrr": mrr,
        "candidate_count": report.get("candidate_count", {}),
        "candidate_pool_recall": report.get("candidate_pool_recall", {}),
        "sid_validity": report.get("sid_validity", {}),
        "test_read": False,
    }
    if comparison_out is not None:
        write_json(comparison_out, summary)
    return summary


def build_summary_command(config: S4Config, stream: str, candidate_row_limit: int = 0) -> list[str]:
    sid_root = config.baseline.root.parents[1]
    results_root = config.results_dir.parents[4]
    outputs_root = config.output_dir.parents[5]
    return [
        "python3",
        "scripts/s4_sasrec_sid_valid_pipeline.py",
        "--action",
        "summarize-candidates",
        "--stream",
        stream,
        "--category",
        config.category,
        "--sid-root",
        sid_root.as_posix(),
        "--baseline-version",
        config.baseline.version,
        "--treatment-version",
        config.treatment.version,
        "--base-model",
        config.base_model.as_posix(),
        "--results-root",
        results_root.as_posix(),
        "--outputs-root",
        outputs_root.as_posix(),
        "--seed",
        str(config.seed),
        "--config-mode",
        config.config_mode,
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--candidate-row-limit",
        str(candidate_row_limit),
    ]


def recommended_max_new_tokens_from_sid_lengths(info_paths: list[Path]) -> dict[str, Any]:
    sid_lengths: dict[str, int] = {}
    max_sid_tokens = 0
    for info_path in info_paths:
        with info_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sid = line.split("\t", 1)[0].strip()
                tokens = parse_sid_tokens(sid)
                if not tokens:
                    raise ValueError(f"Malformed SID in info file {info_path}: {sid}")
                length = len(tokens)
                sid_lengths[str(length)] = sid_lengths.get(str(length), 0) + 1
                max_sid_tokens = max(max_sid_tokens, length)
    return {
        "sid_token_length_distribution": sid_lengths,
        "max_sid_tokens": max_sid_tokens,
        "newline_tokens": 1,
        "eos_tokens": 1,
        "recommended_max_new_tokens": max_sid_tokens + 2,
        "policy": "SID tokens plus newline plus EOS; ConstrainedLogitsProcessor allows EOS-only continuation after terminal EOS.",
    }


def create_limited_eval_csv(source_csv: Path, output_csv: Path, row_limit: int) -> Path:
    require_valid_path(source_csv, "source_eval_csv")
    if row_limit <= 0:
        return source_csv
    rows = read_csv_rows(source_csv)
    if row_limit > len(rows):
        raise ValueError(f"candidate row limit {row_limit} exceeds eval rows {len(rows)}")
    with source_csv.open("r", encoding="utf-8", newline="") as f:
        fieldnames = list(csv.DictReader(f).fieldnames or [])
    write_csv_rows(output_csv, rows[:row_limit], fieldnames)
    return output_csv


def ensure_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_probe"
    try:
        probe.write_text("ok\n", encoding="utf-8")
    finally:
        if probe.exists():
            probe.unlink()


def candidate_target_paths(artifacts: StreamArtifacts) -> list[Path]:
    return [artifacts.prediction_file, artifacts.candidates_jsonl, artifacts.candidate_report]


def legacy_unscoped_artifacts(config: S4Config, stream: str) -> dict[str, str]:
    sid = sid_for_stream(config, stream)
    legacy_root = replace_path_part(config.results_dir, config.treatment.version, sid.version)
    return {
        "generation_prediction": (legacy_root / "generation" / "predictions.json").as_posix(),
        "candidates_jsonl": (legacy_root / "candidates" / "candidates.jsonl").as_posix(),
        "candidate_report": (legacy_root / "candidates" / "candidate_report.json").as_posix(),
    }


def quarantine_commands(artifacts: StreamArtifacts) -> list[str]:
    stamp = "YYYYMMDD_HHMMSS"
    return [
        f"mkdir -p {shlex.quote((artifacts.results_dir.parent / 'quarantine').as_posix())}",
        (
            "mv "
            f"{shlex.quote(artifacts.results_dir.as_posix())} "
            f"{shlex.quote((artifacts.results_dir.parent / 'quarantine' / (artifacts.scope + '_' + stamp)).as_posix())}"
        ),
    ]


def candidate_manifest_payload(
    config: S4Config,
    artifacts: StreamArtifacts,
    candidate_row_limit: int,
    complete: bool,
    actual_prediction_rows: int | None = None,
) -> dict[str, Any]:
    expected_rows = csv_row_count(artifacts.eval_csv) if artifacts.eval_csv.is_file() else None
    return {
        "schema": "s4_candidate_artifact_manifest.v1",
        "stream": artifacts.stream,
        "sid_version": artifacts.sid.version,
        "config_mode": config.config_mode,
        "scope": artifacts.scope,
        "candidate_row_limit": candidate_row_limit,
        "expected_prediction_rows": expected_rows,
        "actual_prediction_rows": actual_prediction_rows,
        "eval_csv": artifacts.eval_csv.as_posix(),
        "eval_csv_sha256": sha256(artifacts.eval_csv) if artifacts.eval_csv.is_file() else None,
        "checkpoint": path_fingerprint(artifacts.checkpoint_dir),
        "item2sid_sha256": sha256(artifacts.sid.item2sid) if artifacts.sid.item2sid.is_file() else None,
        "sid2items_sha256": sha256(artifacts.sid.sid2items) if artifacts.sid.sid2items.is_file() else None,
        "num_beams": config.num_beams,
        "max_new_tokens": config.max_new_tokens,
        "candidate_mode": "exact_full_sid_only",
        "seed": config.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "artifacts": {
            "predictions": path_fingerprint(artifacts.prediction_file),
            "candidates_jsonl": path_fingerprint(artifacts.candidates_jsonl),
            "candidate_report": path_fingerprint(artifacts.candidate_report),
            "metrics_summary": path_fingerprint(artifacts.metrics_summary),
        },
    }


def write_candidate_artifact_manifest(
    config: S4Config,
    artifacts: StreamArtifacts,
    candidate_row_limit: int,
    complete: bool,
    actual_prediction_rows: int | None = None,
) -> dict[str, Any]:
    manifest = candidate_manifest_payload(
        config,
        artifacts,
        candidate_row_limit=candidate_row_limit,
        complete=complete,
        actual_prediction_rows=actual_prediction_rows,
    )
    write_json(artifacts.artifact_manifest, manifest)
    return manifest


def candidate_scope_from_manifest(manifest: dict[str, Any]) -> str:
    scope = manifest.get("scope")
    if isinstance(scope, str):
        return scope
    row_limit = int(manifest.get("candidate_row_limit", 0))
    return candidate_scope(row_limit)


def prediction_row_count(path: Path) -> int:
    data = read_json(path)
    if not isinstance(data, list):
        raise TypeError(f"Prediction file must be a JSON list: {path}")
    return len(data)


def audit_candidate_artifacts(
    config: S4Config,
    artifacts: StreamArtifacts,
    candidate_row_limit: int,
) -> dict[str, Any]:
    targets = candidate_target_paths(artifacts)
    existing_targets = [path for path in targets if path.exists()]
    target_info = {path.as_posix(): path_fingerprint(path) for path in targets}
    legacy_paths = legacy_unscoped_artifacts(config, artifacts.stream)
    legacy_existing = [path for path in legacy_paths.values() if Path(path).exists()]
    base = {
        "stream": artifacts.stream,
        "sid_version": artifacts.sid.version,
        "scope": artifacts.scope,
        "candidate_row_limit": candidate_row_limit,
        "results_dir": artifacts.results_dir.as_posix(),
        "artifact_manifest": artifacts.artifact_manifest.as_posix(),
        "target_artifacts": target_info,
        "legacy_unscoped_artifacts": {
            "paths": legacy_paths,
            "existing": legacy_existing,
            "classification": "legacy_unknown" if legacy_existing else "missing",
        },
        "quarantine_commands": quarantine_commands(artifacts),
    }
    if not existing_targets and not artifacts.artifact_manifest.exists():
        return {**base, "classification": "missing", "reusable": False}
    if any(path.is_file() and path.stat().st_size == 0 for path in existing_targets):
        return {**base, "classification": "empty", "reusable": False}
    if not artifacts.artifact_manifest.exists():
        return {**base, "classification": "legacy_unknown", "reusable": False}
    try:
        manifest = read_json(artifacts.artifact_manifest)
    except Exception as exc:
        return {
            **base,
            "classification": "malformed",
            "reusable": False,
            "error": f"artifact_manifest: {type(exc).__name__}: {exc}",
        }
    if not isinstance(manifest, dict):
        return {**base, "classification": "malformed", "reusable": False, "error": "artifact_manifest is not an object"}
    if manifest.get("stream") != artifacts.stream:
        return {**base, "classification": "wrong_stream", "reusable": False, "manifest": manifest}
    if manifest.get("sid_version") != artifacts.sid.version:
        return {**base, "classification": "wrong_sid_version", "reusable": False, "manifest": manifest}
    if candidate_scope_from_manifest(manifest) != artifacts.scope:
        return {**base, "classification": "wrong_scope", "reusable": False, "manifest": manifest}
    if int(manifest.get("candidate_row_limit", -1)) != candidate_row_limit:
        return {**base, "classification": "wrong_scope", "reusable": False, "manifest": manifest}
    if any(not path.is_file() for path in targets):
        return {**base, "classification": "partial", "reusable": False, "manifest": manifest}
    try:
        actual_rows = prediction_row_count(artifacts.prediction_file)
        read_json(artifacts.candidate_report)
        with artifacts.candidates_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)
    except Exception as exc:
        return {
            **base,
            "classification": "malformed",
            "reusable": False,
            "manifest": manifest,
            "error": f"{type(exc).__name__}: {exc}",
        }
    expected_rows = manifest.get("expected_prediction_rows")
    if expected_rows is not None and actual_rows != int(expected_rows):
        return {
            **base,
            "classification": "partial",
            "reusable": False,
            "manifest": manifest,
            "actual_prediction_rows": actual_rows,
        }
    if not manifest.get("complete"):
        return {
            **base,
            "classification": "partial",
            "reusable": False,
            "manifest": manifest,
            "actual_prediction_rows": actual_rows,
        }
    classification = "bounded" if candidate_row_limit > 0 else "full_valid"
    return {
        **base,
        "classification": classification,
        "reusable": True,
        "manifest": manifest,
        "actual_prediction_rows": actual_rows,
    }


def guard_candidate_artifacts_for_run(
    audit: dict[str, Any],
    overwrite: bool,
    reuse_complete: bool,
) -> str:
    classification = audit["classification"]
    if classification == "missing":
        return "run"
    if classification in {"bounded", "full_valid"}:
        if reuse_complete:
            return "reuse"
        if overwrite:
            return "run"
        raise FileExistsError(
            "Refusing to overwrite complete S4 candidate artifacts. "
            "Set REUSE_COMPLETE=1 to reuse them, or OVERWRITE=1 for an intentional same-scope rerun."
        )
    if classification == "partial" and overwrite:
        return "run"
    raise FileExistsError(
        "Refusing unsafe S4 candidate artifact state "
        f"({classification}). Quarantine manually before rerun: {audit['quarantine_commands']}"
    )


def prepare_candidate_outputs(
    config: S4Config,
    stream: str,
    overwrite: bool,
    candidate_row_limit: int,
    reuse_complete: bool = False,
    create_dirs: bool = True,
) -> dict[str, Any]:
    base_artifacts = stream_artifacts(config, stream, candidate_row_limit=candidate_row_limit)
    eval_csv = base_artifacts.sid.valid_csv
    if candidate_row_limit > 0:
        eval_csv = base_artifacts.generation_dir / f"valid_first_{candidate_row_limit}.csv"
    artifacts = stream_artifacts(config, stream, eval_csv=eval_csv, candidate_row_limit=candidate_row_limit)
    validate_stream_output_isolation(artifacts)
    if create_dirs:
        for directory in [artifacts.generation_dir, artifacts.candidates_dir, artifacts.summary_dir]:
            ensure_writable_dir(directory)
    if create_dirs and candidate_row_limit > 0:
        create_limited_eval_csv(base_artifacts.sid.valid_csv, eval_csv, candidate_row_limit)
    audit = audit_candidate_artifacts(config, artifacts, candidate_row_limit)
    decision = guard_candidate_artifacts_for_run(audit, overwrite=overwrite, reuse_complete=reuse_complete)
    if create_dirs and decision == "run":
        write_candidate_artifact_manifest(
            config,
            artifacts,
            candidate_row_limit=candidate_row_limit,
            complete=False,
            actual_prediction_rows=None,
        )
    prep = {
        "stream": stream,
        "sid_version": artifacts.sid.version,
        "scope": artifacts.scope,
        "candidate_row_limit": candidate_row_limit,
        "eval_csv": artifacts.eval_csv.as_posix(),
        "expected_prediction_rows": csv_row_count(artifacts.eval_csv) if artifacts.eval_csv.is_file() else (
            candidate_row_limit if candidate_row_limit > 0 else EXPECTED_VALID_ROWS
        ),
        "generation_dir": artifacts.generation_dir.as_posix(),
        "candidates_dir": artifacts.candidates_dir.as_posix(),
        "summary_dir": artifacts.summary_dir.as_posix(),
        "prediction_file": artifacts.prediction_file.as_posix(),
        "candidates_jsonl": artifacts.candidates_jsonl.as_posix(),
        "candidate_report": artifacts.candidate_report.as_posix(),
        "metrics_summary": artifacts.metrics_summary.as_posix(),
        "artifact_manifest": artifacts.artifact_manifest.as_posix(),
        "audit": audit,
        "decision": decision,
        "reused": decision == "reuse",
        "commands": {
            "generation": shell_join(build_generation_command(config, artifacts)),
            "candidate_eval": shell_join(build_candidate_eval_command(config, artifacts)),
            "summary": shell_join(build_summary_command(config, stream, candidate_row_limit=candidate_row_limit)),
        },
        "partial_artifacts": {
            path.as_posix(): path.exists() for path in [
                artifacts.prediction_file,
                artifacts.candidates_jsonl,
                artifacts.candidate_report,
            ]
        },
    }
    if create_dirs:
        write_json(artifacts.summary_dir / "candidate_prepare_manifest.json", prep)
    return prep


def validate_predictions(prediction_file: Path, eval_csv: Path) -> dict[str, Any]:
    require_valid_path(eval_csv, "eval_csv")
    if not prediction_file.is_file():
        raise FileNotFoundError(f"Prediction file was not produced: {prediction_file}")
    data = read_json(prediction_file)
    if not isinstance(data, list):
        raise TypeError(f"Prediction file must be a JSON list: {prediction_file}")
    expected_rows = csv_row_count(eval_csv)
    actual_rows = len(data)
    if actual_rows != expected_rows:
        raise ValueError(
            f"prediction row count mismatch: predictions={actual_rows}, eval_rows={expected_rows}"
        )
    return {
        "prediction_file": prediction_file.as_posix(),
        "eval_csv": eval_csv.as_posix(),
        "prediction_rows": actual_rows,
        "expected_rows": expected_rows,
        "row_alignment_ok": True,
    }


def finalize_candidate_artifacts(
    config: S4Config,
    stream: str,
    candidate_row_limit: int,
) -> dict[str, Any]:
    base_artifacts = stream_artifacts(config, stream, candidate_row_limit=candidate_row_limit)
    eval_csv = base_artifacts.sid.valid_csv
    if candidate_row_limit > 0:
        eval_csv = base_artifacts.generation_dir / f"valid_first_{candidate_row_limit}.csv"
    artifacts = stream_artifacts(config, stream, eval_csv=eval_csv, candidate_row_limit=candidate_row_limit)
    prediction = validate_predictions(artifacts.prediction_file, artifacts.eval_csv)
    if not artifacts.candidates_jsonl.is_file():
        raise FileNotFoundError(f"Candidate JSONL was not produced: {artifacts.candidates_jsonl}")
    if not artifacts.candidate_report.is_file():
        raise FileNotFoundError(f"Candidate report was not produced: {artifacts.candidate_report}")
    manifest = write_candidate_artifact_manifest(
        config,
        artifacts,
        candidate_row_limit=candidate_row_limit,
        complete=True,
        actual_prediction_rows=prediction["prediction_rows"],
    )
    audit = audit_candidate_artifacts(config, artifacts, candidate_row_limit)
    return {"manifest": manifest, "audit": audit}


def build_manifest(config: S4Config, baseline_report: dict[str, Any], treatment_report: dict[str, Any]) -> dict[str, Any]:
    baseline_tokens = load_index_token_set(config.baseline.index_json)
    treatment_tokens = load_index_token_set(config.treatment.index_json)
    identical_parameters = {
        "base_model": config.base_model.as_posix(),
        "seed": config.seed,
        "category": config.category,
        "split": EXPECTED_SPLIT,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "valid_rows": EXPECTED_VALID_ROWS,
        "num_gpus": config.num_gpus,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "cutoff_len": config.cutoff_len,
        "num_train_epochs": config.num_train_epochs,
        "bf16": config.bf16,
        "sample": config.sample,
        "gradient_checkpointing": config.gradient_checkpointing,
        "num_beams": config.num_beams,
        "max_new_tokens": config.max_new_tokens,
        "length_penalty": config.length_penalty,
        "max_pred_sids": config.max_pred_sids,
        "max_candidates": config.max_candidates,
        "effective_train_sample": config.sample,
        "effective_train_rows_available": EXPECTED_TRAIN_ROWS,
        "effective_steps": "computed by Trainer from sample, epochs, world size, batch size, and gradient accumulation",
        "evaluator": "evaluate_candidates.py",
        "candidate_mode": "exact_full_sid_only",
        "resume": "disabled_by_default",
        "overwrite": "disabled_by_default",
    }
    return {
        "stage": "S4",
        "purpose": "SASRec Strong Behavior-SID generative recommendation valid-only fair comparison",
        "category": config.category,
        "config_mode": config.config_mode,
        "baseline_sid_version": config.baseline.version,
        "treatment_sid_version": config.treatment.version,
        "base_model": path_fingerprint(config.base_model),
        "runtime_environment": {
            "preflight": collect_runtime_environment(),
            "strict_preflight_required_before_torchrun": True,
        },
        "tokenizer_initial_state": "loaded from explicit base_model before SID token extension",
        "special_tokens": {
            "baseline_count": len(baseline_tokens),
            "treatment_count": len(treatment_tokens),
            "token_string_sets_equal": baseline_tokens == treatment_tokens,
            "only_in_baseline_sample": sorted(baseline_tokens - treatment_tokens)[:10],
            "only_in_treatment_sample": sorted(treatment_tokens - baseline_tokens)[:10],
            "d_tokens_preserved": any(token.startswith("<d_") for token in treatment_tokens),
        },
        "baseline_input": baseline_report,
        "treatment_input": treatment_report,
        "identical_parameters": identical_parameters,
        "training_contract": {
            "config_mode": config.config_mode,
            "budget_name": mode_defaults(config.config_mode)["budget_name"],
            "smoke_is_bounded": config.config_mode == "smoke",
            "effective_train_sample": config.sample,
            "formal_uses_predefined_budget": config.config_mode == "formal",
            "env_overrides_are_recorded_in_identical_parameters": True,
        },
        "must_differ": {
            "sid_mapping": "item_id -> SID differs by construction",
            "sid_csv_content": "train/valid rows are rewritten with version-specific SID tokens",
            "output_dirs": "isolated by SID version, seed, mode, and valid split",
        },
        "unconfirmed_local_evidence": [
            "Historical CF-SID formal checkpoint/base-model parity was not proven from local artifacts.",
        ],
        "commands": {
            name: shell_join(cmd) for name, cmd in build_command_plan(config).items()
        },
        "contracts": {
            "split": EXPECTED_SPLIT,
            "valid_only": True,
            "test_read": False,
            "test_paths_rejected": True,
            "formal_results_not_generated_locally": True,
            "no_text_strong_fusion": True,
            "no_p3_ranker_modification": True,
        },
        "git": git_state(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S4 Strong Behavior-SID valid-only parity and runner planner.")
    parser.add_argument(
        "--action",
        choices=[
            "parity-audit",
            "dry-run",
            "summarize-candidates",
            "env-preflight",
            "plan-candidates",
            "prepare-candidates",
            "audit-existing",
            "validate-predictions",
            "finalize-candidates",
            "token-budget",
        ],
        default="dry-run",
    )
    parser.add_argument("--category", default=EXPECTED_CATEGORY)
    parser.add_argument("--sid-root", type=Path, default=Path("data/Amazon/sid_versions"))
    parser.add_argument("--baseline-version", default=DEFAULT_BASELINE_VERSION)
    parser.add_argument("--treatment-version", default=DEFAULT_TREATMENT_VERSION)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-mode", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", default="2e-5")
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--num-train-epochs", default="1")
    parser.add_argument("--bf16", default="True")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", default="False")
    parser.add_argument("--num-beams", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--length-penalty", default="0.0")
    parser.add_argument("--max-pred-sids", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--allow-missing-base-model", action="store_true")
    parser.add_argument("--candidate-report", type=Path, default=None)
    parser.add_argument("--preflight-output", type=Path, default=None)
    parser.add_argument("--stream", choices=["baseline", "treatment"], default="treatment")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--reuse-complete", action="store_true")
    parser.add_argument("--candidate-row-limit", type=int, default=0)
    parser.add_argument("--prediction-file", type=Path, default=None)
    parser.add_argument("--eval-csv", type=Path, default=None)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> S4Config:
    baseline = resolve_sid_paths(args.sid_root, args.baseline_version, args.category)
    treatment = resolve_sid_paths(args.sid_root, args.treatment_version, args.category)
    defaults = mode_defaults(args.config_mode)
    sample = defaults["sample"] if args.sample is None else args.sample
    max_new_tokens = DEFAULT_MAX_NEW_TOKENS if args.max_new_tokens is None else args.max_new_tokens
    if args.config_mode == "smoke" and sample <= 0:
        raise ValueError("S4 smoke must use a bounded positive sample budget; refusing sample=-1")
    if max_new_tokens <= 0:
        raise ValueError("S4 max_new_tokens must be > 0; default is 6 for 4-level SID plus newline plus EOS")
    suffix = f"{args.category}/{args.treatment_version}/seed{args.seed}/{args.config_mode}/valid"
    results_dir = args.results_root / suffix
    output_dir = args.outputs_root / suffix / "sft"
    return S4Config(
        category=args.category,
        seed=args.seed,
        config_mode=args.config_mode,
        base_model=args.base_model,
        baseline=baseline,
        treatment=treatment,
        results_dir=results_dir,
        output_dir=output_dir,
        prediction_file=results_dir / "generation" / "predictions.json",
        candidates_jsonl=results_dir / "candidates" / "candidates.jsonl",
        candidate_report=results_dir / "candidates" / "candidate_report.json",
        num_gpus=args.num_gpus,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        sample=sample,
        gradient_checkpointing=args.gradient_checkpointing,
        num_beams=args.num_beams,
        max_new_tokens=max_new_tokens,
        length_penalty=args.length_penalty,
        max_pred_sids=args.max_pred_sids,
        max_candidates=args.max_candidates,
        dry_run=args.action == "dry-run",
        allow_missing_base_model=args.allow_missing_base_model,
    )


def run_audit(config: S4Config, write_manifest: bool) -> dict[str, Any]:
    if config.category != EXPECTED_CATEGORY:
        raise ValueError(f"S4 currently frozen for {EXPECTED_CATEGORY}, got {config.category}")
    if not config.allow_missing_base_model and not config.base_model.exists():
        raise FileNotFoundError(f"Explicit base model path does not exist: {config.base_model}")
    validate_output_isolation(config)
    baseline_report = validate_sid_version(config.baseline, EXPECTED_TRAIN_ROWS, EXPECTED_VALID_ROWS)
    treatment_report = validate_sid_version(config.treatment, EXPECTED_TRAIN_ROWS, EXPECTED_VALID_ROWS)
    manifest = build_manifest(config, baseline_report, treatment_report)
    if write_manifest:
        write_json(config.results_dir / "s4_parity_manifest.json", manifest)
        write_json(config.results_dir / "command_plan.json", build_command_plan(config))
    return manifest


def main() -> None:
    args = parse_args()
    config = make_config(args)
    if args.action == "token-budget":
        info = recommended_max_new_tokens_from_sid_lengths([
            config.baseline.info_txt,
            config.treatment.info_txt,
        ])
        print(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.action == "env-preflight":
        report = collect_runtime_environment()
        out = args.preflight_output or (config.results_dir / "runtime_environment_preflight.json")
        write_json(out, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        try:
            assert_runtime_environment_ok(report)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from None
        return
    if args.action == "summarize-candidates":
        artifacts = stream_artifacts(config, args.stream, candidate_row_limit=args.candidate_row_limit)
        report_path = args.candidate_report or artifacts.candidate_report
        require_valid_path(report_path, "candidate_report")
        out = artifacts.metrics_summary
        summary = summarize_candidate_report(report_path, out)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.action == "plan-candidates":
        prep = prepare_candidate_outputs(
            config,
            args.stream,
            overwrite=args.overwrite_existing,
            candidate_row_limit=args.candidate_row_limit,
            reuse_complete=args.reuse_complete,
            create_dirs=False,
        )
        print(json.dumps(prep, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.action == "prepare-candidates":
        prep = prepare_candidate_outputs(
            config,
            args.stream,
            overwrite=args.overwrite_existing,
            candidate_row_limit=args.candidate_row_limit,
            reuse_complete=args.reuse_complete,
            create_dirs=True,
        )
        print(json.dumps(prep, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.action == "audit-existing":
        base_artifacts = stream_artifacts(config, args.stream, candidate_row_limit=args.candidate_row_limit)
        eval_csv = base_artifacts.sid.valid_csv
        if args.candidate_row_limit > 0:
            eval_csv = base_artifacts.generation_dir / f"valid_first_{args.candidate_row_limit}.csv"
        artifacts = stream_artifacts(
            config,
            args.stream,
            eval_csv=eval_csv,
            candidate_row_limit=args.candidate_row_limit,
        )
        audit = audit_candidate_artifacts(config, artifacts, args.candidate_row_limit)
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.action == "validate-predictions":
        artifacts = stream_artifacts(config, args.stream, candidate_row_limit=args.candidate_row_limit)
        prediction_file = args.prediction_file or artifacts.prediction_file
        eval_csv = args.eval_csv or artifacts.eval_csv
        result = validate_predictions(prediction_file, eval_csv)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.action == "finalize-candidates":
        result = finalize_candidate_artifacts(
            config,
            args.stream,
            candidate_row_limit=args.candidate_row_limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return

    manifest = run_audit(config, write_manifest=args.action == "parity-audit")
    print("S4 SASRec Strong Behavior-SID valid-only plan")
    print(f"  action={args.action}")
    print(f"  config_mode={config.config_mode}")
    print(f"  base_model={config.base_model}")
    print(f"  baseline={config.baseline.root}")
    print(f"  treatment={config.treatment.root}")
    print(f"  results_dir={config.results_dir}")
    print(f"  output_dir={config.output_dir}")
    print(f"  split={EXPECTED_SPLIT}")
    print(f"  test_read={manifest['contracts']['test_read']}")
    print("Resolved commands:")
    for name, cmd in manifest["commands"].items():
        print(f"  [{name}] {cmd}")
    if args.action == "dry-run":
        print("DRY_RUN: no checkpoint, prediction, candidate, or metric artifact was written.")
    else:
        print(f"Wrote {config.results_dir / 's4_parity_manifest.json'}")


if __name__ == "__main__":
    main()
