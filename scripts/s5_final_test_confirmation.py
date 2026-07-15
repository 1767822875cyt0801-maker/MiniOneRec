#!/usr/bin/env python3
"""Freeze S5 auxiliary fusion config and prepare final test confirmation.

Local use is manifest/dry-run only. Final test execution requires both
``DRY_RUN=0`` and ``CONFIRM_FINAL_TEST=1`` so test labels cannot be consumed by
accident.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import s5_auxiliary_fusion_conversion as s5


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)).resolve()
CATEGORY = "Industrial_and_Scientific"
RESULT_ROOT_REL = Path("results/s5_auxiliary_fusion/Industrial_and_Scientific")
CONFIG_PATH_REL = Path("configs/s5_auxiliary_fusion/frozen_release_config.json")
RELEASE_MANIFEST_REL = RESULT_ROOT_REL / "s5_1_release_manifest.json"
SYNC_MANIFEST_REL = RESULT_ROOT_REL / "s5_1_autodl_sync_manifest.json"
DOC_PATH_REL = Path("docs/s5_1_frozen_release_and_test_protocol.md")
PARTIAL_RESUME_DOC_REL = Path("docs/s5_1_partial_final_test_resume.md")
PARTIAL_RESUME_TEMPLATE_REL = RESULT_ROOT_REL / "s5_1_partial_resume_manifest_template.json"
PARTIAL_RESUME_SYNC_REL = RESULT_ROOT_REL / "s5_1_partial_resume_sync_manifest.json"
S5_0_CLOSEOUT_REL = RESULT_ROOT_REL / "s5_0_protocol_closeout.json"
S5_0_SELECTED_REL = RESULT_ROOT_REL / "s5_0_selected_config.json"
S5_0_LEADERBOARD_REL = RESULT_ROOT_REL / "s5_0_valid_select_leaderboard.csv"
S5_0_REPORT_REL = RESULT_ROOT_REL / "s5_0_valid_report_metrics.json"
FORMAL_ROOT_REL = Path("incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036")
S4_RESULT_ROOT_REL = FORMAL_ROOT_REL / "results/s4_sasrec_sid_valid/Industrial_and_Scientific"
CONFIG_PATH = PROJECT_ROOT / CONFIG_PATH_REL
RELEASE_MANIFEST = PROJECT_ROOT / RELEASE_MANIFEST_REL
SYNC_MANIFEST = PROJECT_ROOT / SYNC_MANIFEST_REL
DOC_PATH = PROJECT_ROOT / DOC_PATH_REL
PARTIAL_RESUME_DOC = PROJECT_ROOT / PARTIAL_RESUME_DOC_REL
PARTIAL_RESUME_TEMPLATE = PROJECT_ROOT / PARTIAL_RESUME_TEMPLATE_REL
PARTIAL_RESUME_SYNC = PROJECT_ROOT / PARTIAL_RESUME_SYNC_REL
S5_0_CLOSEOUT = PROJECT_ROOT / S5_0_CLOSEOUT_REL
S5_0_SELECTED = PROJECT_ROOT / S5_0_SELECTED_REL
S5_0_LEADERBOARD = PROJECT_ROOT / S5_0_LEADERBOARD_REL
S5_0_REPORT = PROJECT_ROOT / S5_0_REPORT_REL
S4_RESULT_ROOT = PROJECT_ROOT / S4_RESULT_ROOT_REL
BASE_MODEL_IDENTITY = "Qwen2.5-0.5B"
BASE_MODEL_ENV = "BASE_MODEL"
PROJECT_ROOT_ENV = "PROJECT_ROOT"
CONFIRM_FINAL_TEST_RESUME_ENV = "CONFIRM_FINAL_TEST_RESUME"
EXPECTED_TEST_ROWS = 4533
SOURCE_SPECIFIC_FEATURES_ZEROED = [
    "text_present",
    "cf_present",
    "both_sources",
    "text_only",
    "cf_only",
    "source_count",
    "text_rank_filled",
    "cf_rank_filled",
    "reciprocal_text_rank",
    "reciprocal_cf_rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S5 final test confirmation runner.")
    actions = {
        "freeze-release",
        "dry-run",
        "run-final-test",
        "finalize-test",
        "audit",
        "audit-partial-final-test",
        "resume-final-test-after-cf",
    }
    parser.add_argument("--action", default="dry-run")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--resume-manifest", type=Path, default=PARTIAL_RESUME_TEMPLATE_REL)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    if args.action not in actions:
        parser.error(f"invalid choice: {args.action}")
    return args


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def reject_machine_absolute(value: str | Path, label: str = "path") -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be repository-relative, not absolute: {path}")
    if any(part == ".." for part in path.parts):
        raise ValueError(f"{label} must not contain path traversal: {path}")
    return path


def resolve_repo_path(value: str | Path, project_root: Path | None = None, label: str = "path") -> Path:
    root = (project_root or PROJECT_ROOT).resolve()
    rel = reject_machine_absolute(value, label)
    resolved = (root / rel).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} resolved outside PROJECT_ROOT: {value}")
    return resolved


def file_sha256_rel(path: str | Path, project_root: Path | None = None) -> str:
    return file_sha256(resolve_repo_path(path, project_root))


def assert_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def model_path_rel() -> Path:
    return Path("results/stage7_validation_protocol/valid/Industrial_and_Scientific/p2_history_ranker/model.json")


def load_model() -> dict[str, Any]:
    return read_json(resolve_repo_path(model_path_rel(), label="ranker model"))


def feature_contract(model: dict[str, Any]) -> dict[str, Any]:
    feature_names = list(model["feature_names"])
    retained = [name for name in feature_names if name not in SOURCE_SPECIFIC_FEATURES_ZEROED]
    return {
        "projection_name": "source_independent_projection",
        "ordered_feature_names": feature_names,
        "source_specific_features_zeroed_after_normalization": SOURCE_SPECIFIC_FEATURES_ZEROED,
        "retained_normalized_features": retained,
        "zeroing_stage": "after normalization",
        "zero_semantics": "normalized zero equals the frozen training-distribution feature mean",
        "missing_value_policy": "same raw feature construction as S5-0; missing source identity ranks are ignored by post-normalization zeroing",
        "dimension_count": len(feature_names),
        "ranker_input_column_order_fixed": True,
        "cf_only_and_cf_sasrec_dimension_match": True,
        "text_cf_identity_misuse_prevention": "text_* and cf_* identity features are zeroed after normalization; SASRec is not mapped into text_* for the frozen release path",
        "normalization_leakage": "uses frozen P2 ranker feature_stats from valid_fit training only; final test does not recompute normalization",
    }


def project_normalized_features(raw: dict[str, float], model: dict[str, Any]) -> list[float]:
    names = model["feature_names"]
    means = model["feature_stats"]["mean"]
    stds = model["feature_stats"]["std"]
    out: list[float] = []
    for idx, name in enumerate(names):
        if name in SOURCE_SPECIFIC_FEATURES_ZEROED:
            out.append(0.0)
        else:
            out.append((float(raw[name]) - float(means[idx])) / float(stds[idx]))
    return out


def stream_artifact(stream: str) -> dict[str, Any]:
    if stream == "cf":
        sid = "cf_k512_dedup"
    elif stream == "sasrec":
        sid = "sasrec_v3_k512_dedup"
    else:
        raise ValueError(stream)
    full_rel = S4_RESULT_ROOT_REL / sid / "seed42/formal/valid/full_valid"
    full = resolve_repo_path(full_rel, label=f"{stream} formal artifact root")
    manifest = read_json(full / "artifact_manifest.json")
    return {
        "sid_version": sid,
        "formal_valid_candidates": (full_rel / "candidates/candidates.jsonl").as_posix(),
        "formal_valid_candidates_sha256": file_sha256(full / "candidates/candidates.jsonl"),
        "formal_artifact_manifest": (full_rel / "artifact_manifest.json").as_posix(),
        "formal_artifact_manifest_sha256": file_sha256(full / "artifact_manifest.json"),
        "formal_checkpoint_path": reject_machine_absolute(manifest["checkpoint"]["path"], f"{stream} checkpoint").as_posix(),
        "formal_checkpoint_sha256_tree": manifest["checkpoint"]["sha256_tree"],
        "item2sid_sha256": manifest["item2sid_sha256"],
        "sid2items_sha256": manifest["sid2items_sha256"],
        "valid_eval_csv_sha256": manifest["eval_csv_sha256"],
    }


def release_config() -> dict[str, Any]:
    model = load_model()
    selected = read_json(S5_0_SELECTED)
    contract = feature_contract(model)
    config = {
        "schema": "s5_frozen_release_config.v1",
        "category": CATEGORY,
        "split_for_final_confirmation": "test",
        "selected_candidate_policy": "source_aware_rrf_lam0.75_bonus0.01",
        "policy": "source_aware_rrf",
        "lambda_sasrec": 0.75,
        "source_bonus": 0.01,
        "ranker_compatibility_mode": "source_independent_projection",
        "candidate_mode": "exact_full_sid_only",
        "baseline_stream": stream_artifact("cf"),
        "auxiliary_stream": stream_artifact("sasrec"),
        "num_beams": 50,
        "max_new_tokens": 6,
        "max_pred_sids": 50,
        "max_candidates": 1000,
        "seed": 42,
        "base_model": {
            "identity": BASE_MODEL_IDENTITY,
            "runtime_env_var": BASE_MODEL_ENV,
            "storage_contract": "runtime-resolved absolute path supplied by BASE_MODEL; not a cross-machine immutable identity",
        },
        "ranker": {
            "name": "P2-4 frozen history-aware ranker",
            "model_path": model_path_rel().as_posix(),
            "model_sha256": file_sha256_rel(model_path_rel()),
            "model_type": model["model_type"],
            "selected_config": model["selected_config"],
            "training_policy": model["training_policy"],
            "feature_schema_hash": canonical_sha256(model["feature_names"]),
            "normalization_metadata_hash": canonical_sha256(model["feature_stats"]),
        },
        "projection_contract": contract,
        "projection_contract_hash": canonical_sha256(contract),
        "projection_implementation": {
            "script": "scripts/s5_final_test_confirmation.py",
            "function": "project_normalized_features",
            "source_specific_features_zeroed": SOURCE_SPECIFIC_FEATURES_ZEROED,
        },
        "s5_0_evidence": {
            "selected_config_path": S5_0_SELECTED_REL.as_posix(),
            "selected_config_sha256": file_sha256_rel(S5_0_SELECTED_REL),
            "selected_config_canonical_hash": canonical_sha256(selected),
            "valid_select_leaderboard_path": S5_0_LEADERBOARD_REL.as_posix(),
            "valid_select_leaderboard_sha256": file_sha256_rel(S5_0_LEADERBOARD_REL),
            "valid_report_metrics_path": S5_0_REPORT_REL.as_posix(),
            "valid_report_metrics_sha256": file_sha256_rel(S5_0_REPORT_REL),
            "protocol_closeout_path": S5_0_CLOSEOUT_REL.as_posix(),
            "protocol_closeout_sha256": file_sha256_rel(S5_0_CLOSEOUT_REL),
        },
        "immutability": {
            "test_result_must_not_change_parameters": True,
            "environment_overrides_allowed": False,
            "final_test_requires_confirm_final_test": True,
            "default_dry_run": True,
        },
        "path_contract": {
            "repository_relative_path": "stored for every repository-owned artifact",
            "content_sha256": "stored for immutable identity and verified before use",
            "runtime_resolved_absolute_path": "derived from PROJECT_ROOT on the executing machine; never used as cross-machine identity",
            "project_root_env_var": PROJECT_ROOT_ENV,
        },
        "output_root": "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test",
    }
    config["selected_config_hash"] = canonical_sha256(
        {
            "policy": config["policy"],
            "lambda_sasrec": config["lambda_sasrec"],
            "source_bonus": config["source_bonus"],
            "ranker_compatibility_mode": config["ranker_compatibility_mode"],
            "candidate_mode": config["candidate_mode"],
            "num_beams": config["num_beams"],
            "max_new_tokens": config["max_new_tokens"],
            "seed": config["seed"],
        }
    )
    return config


def verify_immutable_config(config: dict[str, Any]) -> None:
    expected = {
        "selected_candidate_policy": "source_aware_rrf_lam0.75_bonus0.01",
        "policy": "source_aware_rrf",
        "lambda_sasrec": 0.75,
        "source_bonus": 0.01,
        "ranker_compatibility_mode": "source_independent_projection",
        "candidate_mode": "exact_full_sid_only",
        "num_beams": 50,
        "max_new_tokens": 6,
        "max_pred_sids": 50,
        "max_candidates": 1000,
        "seed": 42,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"Frozen config mismatch: {mismatches}")


def verify_release_hashes(config: dict[str, Any]) -> None:
    checks = [
        (config["ranker"]["model_path"], config["ranker"]["model_sha256"], "ranker model"),
        (config["s5_0_evidence"]["selected_config_path"], config["s5_0_evidence"]["selected_config_sha256"], "S5-0 selected config"),
        (config["s5_0_evidence"]["valid_select_leaderboard_path"], config["s5_0_evidence"]["valid_select_leaderboard_sha256"], "S5-0 leaderboard"),
        (config["s5_0_evidence"]["valid_report_metrics_path"], config["s5_0_evidence"]["valid_report_metrics_sha256"], "S5-0 report"),
    ]
    for stream_key in ["baseline_stream", "auxiliary_stream"]:
        stream = config[stream_key]
        checks.append((stream["formal_valid_candidates"], stream["formal_valid_candidates_sha256"], stream_key))
        checks.append((stream["formal_artifact_manifest"], stream["formal_artifact_manifest_sha256"], f"{stream_key} manifest"))
    for path, expected, label in checks:
        resolved = resolve_repo_path(path, label=label)
        assert_file(resolved)
        actual = file_sha256(resolved)
        if actual != expected:
            raise ValueError(f"Hash mismatch for {label}: {path} expected={expected} actual={actual}")


def final_root(config: dict[str, Any]) -> Path:
    return resolve_repo_path(config["output_root"], label="output_root")


def final_root_rel(config: dict[str, Any]) -> Path:
    return reject_machine_absolute(config["output_root"], "output_root")


def stream_test_paths(config: dict[str, Any], stream_key: str) -> dict[str, Path]:
    stream = config[stream_key]
    sid = stream["sid_version"]
    root = final_root_rel(config) / sid
    data_root = Path("data/Amazon/sid_versions") / sid / CATEGORY
    out_root = Path("outputs/s4_sasrec_sid_valid") / CATEGORY / sid / "seed42/formal/valid/sft/final_checkpoint"
    return {
        "root": root,
        "checkpoint": out_root,
        "train_csv": data_root / "train.csv",
        "eval_csv": data_root / "test.csv",
        "info_file": data_root / "info.txt",
        "item2sid": data_root / "item2sid.json",
        "sid2items": data_root / "sid2items.json",
        "valid_sid_set": data_root / "valid_sid_set.json",
        "predictions": root / "generation/predictions.json",
        "candidates": root / "candidates/candidates.jsonl",
        "candidate_report": root / "candidates/candidate_report.json",
    }


def shell_quote(path: Path | str) -> str:
    text = str(path)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def command_plan(config: dict[str, Any]) -> dict[str, Any]:
    project_root_assignment = f'{PROJECT_ROOT_ENV}="${{{PROJECT_ROOT_ENV}:?set PROJECT_ROOT}}"'
    base_model_arg = f'"${{{BASE_MODEL_ENV}:?set BASE_MODEL}}"'
    common = (
        f"--category {CATEGORY} --sid-root data/Amazon/sid_versions "
        f"--baseline-version cf_k512_dedup --treatment-version sasrec_v3_k512_dedup "
        f"--base-model {base_model_arg} --results-root results/s4_sasrec_sid_valid "
        f"--outputs-root outputs/s4_sasrec_sid_valid --seed {config['seed']} --config-mode formal "
        f"--num-beams {config['num_beams']} --max-new-tokens {config['max_new_tokens']} "
        f"--max-pred-sids {config['max_pred_sids']} --max-candidates {config['max_candidates']}"
    )
    commands: dict[str, str] = {
        "env_preflight": f"{project_root_assignment} python3 scripts/s4_sasrec_sid_valid_pipeline.py --action env-preflight {common}",
        "parity": f"{project_root_assignment} python3 scripts/s4_sasrec_sid_valid_pipeline.py --action parity-audit {common}",
    }
    for label, stream_key in [("cf", "baseline_stream"), ("sasrec", "auxiliary_stream")]:
        paths = stream_test_paths(config, stream_key)
        commands[f"{label}_test_generation"] = (
            f"{project_root_assignment} python3 evaluate.py --base_model {shell_quote(paths['checkpoint'])} "
            f"--train_file {shell_quote(paths['train_csv'])} --info_file {shell_quote(paths['info_file'])} "
            f"--category {CATEGORY} --test_data_path {shell_quote(paths['eval_csv'])} "
            f"--result_json_data {shell_quote(paths['predictions'])} --batch_size 4 --seed {config['seed']} "
            f"--length_penalty 0.0 --max_new_tokens {config['max_new_tokens']} --num_beams {config['num_beams']}"
        )
        commands[f"{label}_test_candidate_eval"] = (
            f"{project_root_assignment} python3 evaluate_candidates.py --prediction-file {shell_quote(paths['predictions'])} "
            f"--eval-csv {shell_quote(paths['eval_csv'])} --eval-split test "
            f"--item2sid {shell_quote(paths['item2sid'])} --sid2items {shell_quote(paths['sid2items'])} "
            f"--valid-sid-set {shell_quote(paths['valid_sid_set'])} --output-jsonl {shell_quote(paths['candidates'])} "
            f"--output-report {shell_quote(paths['candidate_report'])} --topk 1 5 10 20 50 "
            f"--max-pred-sids {config['max_pred_sids']} --max-candidates {config['max_candidates']}"
        )
    commands["finalize"] = (
        f"{project_root_assignment} CONFIRM_FINAL_TEST=1 DRY_RUN=0 python3 scripts/s5_final_test_confirmation.py "
        f"--action finalize-test --config {shell_quote(CONFIG_PATH_REL)}"
    )
    return {
        "sequence": [
            "env_preflight",
            "parity",
            "cf_test_generation",
            "cf_test_candidate_eval",
            "sasrec_test_generation",
            "sasrec_test_candidate_eval",
            "finalize",
        ],
        "commands": commands,
    }


def resume_command_plan(config: dict[str, Any]) -> dict[str, Any]:
    full = command_plan(config)["commands"]
    project_root_assignment = f'{PROJECT_ROOT_ENV}="${{{PROJECT_ROOT_ENV}:?set PROJECT_ROOT}}"'
    commands = {
        "input_inventory_preflight": (
            f"{project_root_assignment} python3 scripts/s5_final_test_confirmation.py "
            f"--action audit-partial-final-test --config {shell_quote(CONFIG_PATH_REL)}"
        ),
        "sasrec_test_generation": full["sasrec_test_generation"],
        "sasrec_test_candidate_eval": full["sasrec_test_candidate_eval"],
        "finalize_resume": (
            f"{project_root_assignment} {CONFIRM_FINAL_TEST_RESUME_ENV}=1 DRY_RUN=0 "
            f"python3 scripts/s5_final_test_confirmation.py --action resume-final-test-after-cf "
            f"--config {shell_quote(CONFIG_PATH_REL)} --resume-manifest {shell_quote(PARTIAL_RESUME_TEMPLATE_REL)}"
        ),
    }
    return {
        "sequence": [
            "input_inventory_preflight",
            "frozen_hash_verification",
            "cf_completed_artifact_verification",
            "sasrec_test_generation",
            "sasrec_row_validation",
            "sasrec_test_candidate_eval",
            "deterministic_cf_sasrec_provenance_merge",
            "frozen_source_aware_rrf",
            "frozen_source_independent_projection",
            "frozen_ranker_inference",
            "final_metrics",
            "final_artifact_manifest",
            "protocol_closeout",
        ],
        "commands": commands,
        "forbidden": ["baseline generation", "baseline candidate evaluation"],
    }


def require_final_resume_confirmation() -> None:
    dry_run = os.environ.get("DRY_RUN", "1")
    confirm = os.environ.get(CONFIRM_FINAL_TEST_RESUME_ENV, "0")
    if dry_run != "0":
        raise SystemExit("Partial final-test resume is disabled because DRY_RUN is not 0.")
    if confirm != "1":
        raise SystemExit(f"Partial final-test resume requires {CONFIRM_FINAL_TEST_RESUME_ENV}=1.")


def require_final_confirmation() -> None:
    dry_run = os.environ.get("DRY_RUN", "1")
    confirm = os.environ.get("CONFIRM_FINAL_TEST", "0")
    if dry_run != "0":
        raise SystemExit("Final test execution is disabled because DRY_RUN is not 0.")
    if confirm != "1":
        raise SystemExit("Final test execution requires CONFIRM_FINAL_TEST=1.")


def protect_final_outputs(config: dict[str, Any], allow_overwrite: bool) -> None:
    if allow_overwrite:
        raise FileExistsError("--allow-overwrite cannot bypass the final-test one-shot contract.")
    root = final_root(config)
    complete = root / "final_test_complete_manifest.json"
    if complete.exists():
        raise FileExistsError(f"Refusing to overwrite completed final test artifacts: {complete}")
    for stream_key in ["baseline_stream", "auxiliary_stream"]:
        paths = stream_test_paths(config, stream_key)
        for key in ["predictions", "candidates", "candidate_report"]:
            path = resolve_repo_path(paths[key], label=key)
            if path.exists() and path.stat().st_size > 0:
                raise FileExistsError(f"Refusing to overwrite existing test artifact: {path}")


def read_csv_for_inventory(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    resolved = resolve_repo_path(path, label="csv inventory")
    assert_file(resolved)
    with open(resolved, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def require_nonempty_field(rows: list[dict[str, str]], field: str, path: Path) -> None:
    missing = [idx for idx, row in enumerate(rows) if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"{path} has empty {field} at rows {missing[:5]}")


def validate_test_csv_pair(cf_csv: Path, sas_csv: Path, expected_rows: int = EXPECTED_TEST_ROWS) -> dict[str, Any]:
    cf_columns, cf_rows = read_csv_for_inventory(cf_csv)
    sas_columns, sas_rows = read_csv_for_inventory(sas_csv)
    if len(cf_rows) != expected_rows:
        raise ValueError(f"CF test rows mismatch: {cf_csv} actual={len(cf_rows)} expected={expected_rows}")
    if len(sas_rows) != expected_rows:
        raise ValueError(f"SASRec test rows mismatch: {sas_csv} actual={len(sas_rows)} expected={expected_rows}")
    if cf_columns != sas_columns:
        raise ValueError(f"CF/SASRec test columns differ: cf={cf_columns} sasrec={sas_columns}")
    for field in ["item_id", "history_item_id"]:
        for idx, (cf_row, sas_row) in enumerate(zip(cf_rows, sas_rows)):
            if str(cf_row.get(field, "")) != str(sas_row.get(field, "")):
                raise ValueError(f"CF/SASRec test {field} mismatch at row {idx}")
    for field in ["item_sid", "history_item_sid"]:
        require_nonempty_field(cf_rows, field, cf_csv)
        require_nonempty_field(sas_rows, field, sas_csv)
    return {
        "cf_test_csv": cf_csv.as_posix(),
        "sasrec_test_csv": sas_csv.as_posix(),
        "rows": expected_rows,
        "columns": cf_columns,
        "item_id_aligned": True,
        "history_item_id_aligned": True,
        "cf_test_sha256": file_sha256(resolve_repo_path(cf_csv, label="cf test csv")),
        "sasrec_test_sha256": file_sha256(resolve_repo_path(sas_csv, label="sasrec test csv")),
    }


def input_inventory_preflight(config: dict[str, Any], expected_rows: int = EXPECTED_TEST_ROWS) -> dict[str, Any]:
    verify_immutable_config(config)
    verify_release_hashes(config)
    inventory: dict[str, Any] = {
        "schema": "s5_final_test_input_inventory.v1",
        "expected_test_rows": expected_rows,
        "streams": {},
        "frozen": {
            "config": {"path": CONFIG_PATH_REL.as_posix(), "sha256": file_sha256_rel(CONFIG_PATH_REL)},
            "release_manifest": {"path": RELEASE_MANIFEST_REL.as_posix(), "sha256": file_sha256_rel(RELEASE_MANIFEST_REL)},
            "projection_implementation": {
                "path": "scripts/s5_final_test_confirmation.py",
                "sha256": file_sha256_rel("scripts/s5_final_test_confirmation.py"),
            },
            "ranker": {
                "path": config["ranker"]["model_path"],
                "sha256": config["ranker"]["model_sha256"],
                "feature_schema_hash": config["ranker"]["feature_schema_hash"],
                "normalization_metadata_hash": config["ranker"]["normalization_metadata_hash"],
            },
        },
    }
    for stream_key in ["baseline_stream", "auxiliary_stream"]:
        paths = stream_test_paths(config, stream_key)
        stream_info: dict[str, Any] = {}
        for key in ["train_csv", "eval_csv", "info_file", "item2sid", "sid2items", "valid_sid_set"]:
            resolved = resolve_repo_path(paths[key], label=f"{stream_key} {key}")
            assert_file(resolved)
            stream_info[key] = {"path": paths[key].as_posix(), "sha256": file_sha256(resolved)}
        checkpoint = resolve_repo_path(paths["checkpoint"], label=f"{stream_key} checkpoint")
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        stream_info["checkpoint"] = {"path": paths["checkpoint"].as_posix(), "exists": True}
        inventory["streams"][stream_key] = stream_info
    inventory["test_csv_pair"] = validate_test_csv_pair(
        stream_test_paths(config, "baseline_stream")["eval_csv"],
        stream_test_paths(config, "auxiliary_stream")["eval_csv"],
        expected_rows,
    )
    return inventory


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return s5.read_jsonl(resolve_repo_path(path, label="jsonl rows"))


def validate_candidate_report(path: Path, expected_rows: int) -> dict[str, Any]:
    report_path = resolve_repo_path(path, label="candidate report")
    report = read_json(report_path)
    if not isinstance(report, dict):
        raise ValueError(f"candidate report must be a JSON object: {path}")
    if int(report.get("num_samples", -1)) != expected_rows:
        raise ValueError(f"candidate report row mismatch: {path}")
    candidate_count = report.get("candidate_count", {})
    if not isinstance(candidate_count, dict):
        raise ValueError(f"candidate report missing candidate_count: {path}")
    if int(candidate_count.get("min", 0)) < 0 or int(candidate_count.get("max", 0)) < int(candidate_count.get("min", 0)):
        raise ValueError(f"candidate report has invalid candidate_count: {path}")
    sid_validity = report.get("sid_validity", {})
    if isinstance(sid_validity, dict):
        for key in ["invalid_sid_count", "total_pred_sid_count"]:
            if key in sid_validity and int(sid_validity[key]) < 0:
                raise ValueError(f"candidate report has negative {key}: {path}")
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(report_path),
        "num_samples": report["num_samples"],
        "candidate_count": candidate_count,
        "sid_validity": sid_validity,
    }


def validate_prediction_artifact(path: Path, eval_csv: Path, expected_rows: int) -> dict[str, Any]:
    validation = validate_prediction_rows(path, eval_csv)
    if validation["actual"] != expected_rows:
        raise ValueError(f"prediction rows mismatch: {path} actual={validation['actual']} expected={expected_rows}")
    validation["sha256"] = file_sha256(resolve_repo_path(path, label="prediction artifact"))
    return validation


def validate_candidate_jsonl_artifact(path: Path, expected_rows: int) -> dict[str, Any]:
    rows = read_jsonl_rows(path)
    if len(rows) != expected_rows:
        raise ValueError(f"candidate rows mismatch: {path} actual={len(rows)} expected={expected_rows}")
    return {"path": path.as_posix(), "rows": len(rows), "sha256": file_sha256(resolve_repo_path(path, label="candidate artifact"))}


def load_resume_manifest(path: Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path, label="resume manifest")
    assert_file(resolved)
    data = read_json(resolved)
    if not isinstance(data, dict):
        raise ValueError("resume manifest must be a JSON object")
    return data


def expected_hash_from_manifest(manifest: dict[str, Any], key: str) -> str | None:
    hashes = manifest.get("expected_cf_artifact_sha256", {})
    value = hashes.get(key) if isinstance(hashes, dict) else None
    if not value or str(value).startswith("<"):
        return None
    return str(value)


def verify_expected_hash(manifest: dict[str, Any], key: str, actual: str, require_expected: bool) -> None:
    expected = expected_hash_from_manifest(manifest, key)
    if expected is None:
        if require_expected:
            raise ValueError(f"resume manifest missing concrete expected hash for {key}")
        return
    if expected != actual:
        raise ValueError(f"resume manifest hash mismatch for {key}: expected={expected} actual={actual}")


def partial_state_audit(
    config: dict[str, Any],
    resume_manifest_path: Path = PARTIAL_RESUME_TEMPLATE_REL,
    expected_rows: int = EXPECTED_TEST_ROWS,
    require_expected_hashes: bool = False,
) -> dict[str, Any]:
    verify_immutable_config(config)
    verify_release_hashes(config)
    resume_manifest = load_resume_manifest(resume_manifest_path)
    root = final_root(config)
    complete = root / "final_test_complete_manifest.json"
    metrics = root / "final_test_metrics.json"
    if complete.exists() or metrics.exists():
        raise FileExistsError("Final metrics or complete manifest already exists; resume is not allowed.")

    cf_paths = stream_test_paths(config, "baseline_stream")
    sas_paths = stream_test_paths(config, "auxiliary_stream")
    cf_prediction = validate_prediction_artifact(cf_paths["predictions"], cf_paths["eval_csv"], expected_rows)
    cf_candidates = validate_candidate_jsonl_artifact(cf_paths["candidates"], expected_rows)
    cf_report = validate_candidate_report(cf_paths["candidate_report"], expected_rows)
    verify_expected_hash(resume_manifest, "cf_predictions", cf_prediction["sha256"], require_expected_hashes)
    verify_expected_hash(resume_manifest, "cf_candidates", cf_candidates["sha256"], require_expected_hashes)
    verify_expected_hash(resume_manifest, "cf_candidate_report", cf_report["sha256"], require_expected_hashes)

    sas_forbidden = []
    for key in ["predictions", "candidates", "candidate_report"]:
        path = resolve_repo_path(sas_paths[key], label=f"sasrec {key}")
        if path.exists() and path.stat().st_size > 0:
            sas_forbidden.append(sas_paths[key].as_posix())
    if sas_forbidden:
        raise FileExistsError(f"SASRec partial artifacts already exist; resume state is not the expected CF-only partial state: {sas_forbidden}")

    return {
        "schema": "s5_partial_final_test_state_audit.v1",
        "state": "cf_complete_sasrec_missing",
        "expected_rows": expected_rows,
        "cf": {
            "predictions": cf_prediction,
            "candidates": cf_candidates,
            "candidate_report": cf_report,
        },
        "sasrec": {"artifacts_missing": True},
        "final_metrics_missing": True,
        "resume_manifest_path": resume_manifest_path.as_posix(),
        "expected_hashes_required": require_expected_hashes,
        "no_parameter_change": True,
    }


def validate_prediction_rows(prediction_file: Path, eval_csv: Path) -> dict[str, Any]:
    prediction_path = resolve_repo_path(prediction_file, label="prediction_file")
    eval_path = resolve_repo_path(eval_csv, label="eval_csv")
    predictions = read_json(prediction_path)
    with open(eval_path, "r", encoding="utf-8", newline="") as f:
        expected = sum(1 for _ in csv.DictReader(f))
    actual = len(predictions) if isinstance(predictions, list) else -1
    if actual != expected:
        raise ValueError(f"Prediction row mismatch: {prediction_file} actual={actual} expected={expected}")
    return {"prediction_file": prediction_file.as_posix(), "eval_csv": eval_csv.as_posix(), "actual": actual, "expected": expected}


def load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    return s5.read_jsonl(resolve_repo_path(path, label="candidate_rows"))


def finalize_test_core(config: dict[str, Any]) -> dict[str, Any]:
    verify_immutable_config(config)
    verify_release_hashes(config)
    root = final_root(config)
    complete = root / "final_test_complete_manifest.json"
    if complete.exists():
        raise FileExistsError(f"Final test already completed: {complete}")
    cf_paths = stream_test_paths(config, "baseline_stream")
    sas_paths = stream_test_paths(config, "auxiliary_stream")
    validations = [
        validate_prediction_rows(cf_paths["predictions"], cf_paths["eval_csv"]),
        validate_prediction_rows(sas_paths["predictions"], sas_paths["eval_csv"]),
    ]
    split_map = {}
    cf_rows = load_candidate_rows(cf_paths["candidates"])
    sas_rows = load_candidate_rows(sas_paths["candidates"])
    for idx in range(len(cf_rows)):
        split_map[str(idx)] = "final_test"
    samples = s5.build_samples(cf_rows, sas_rows, split_map)
    configs = {
        "cf_only": {"config_id": "cf_only_top50", "policy": "cf_only_top50"},
        "sasrec_only": {"config_id": "sasrec_only_top50", "policy": "sasrec_only_top50"},
        "fixed_rrf": {
            "config_id": config["selected_candidate_policy"],
            "policy": "source_aware_rrf",
            "lambda_sasrec": config["lambda_sasrec"],
            "source_bonus": config["source_bonus"],
        },
    }
    metrics = {
        key: s5.evaluate_policy(samples, cfg, "final_test", [1, 5, 10, 20, 50])
        for key, cfg in configs.items()
    }
    model = load_model()
    matrix, row_index = s5.load_embeddings(
        resolve_repo_path(f"data/Amazon/cs_embeddings/{CATEGORY}/{CATEGORY}.cf_emb.npy", label="cf embedding"),
        resolve_repo_path(f"data/Amazon/cs_embeddings/{CATEGORY}/{CATEGORY}.row_index.json", label="cf row index"),
    )
    popularity = s5.train_popularity(resolve_repo_path(f"data/Amazon/sid_versions/cf_k512_dedup/{CATEGORY}/train.csv", label="cf train csv"))
    max_log_pop = max((math.log1p(v) for v in popularity.values()), default=1.0)
    metrics["projected_ranker"] = s5.evaluate_frozen_ranker(
        samples,
        configs["fixed_rrf"],
        "final_test",
        [1, 5, 10, 20, 50],
        model,
        matrix,
        row_index,
        popularity,
        max_log_pop,
        "source_independent_projection",
    )
    out = {
        "schema": "s5_final_test_confirmation.v1",
        "config_sha256": file_sha256(resolve_repo_path(CONFIG_PATH_REL, label="config")),
        "validations": validations,
        "metrics": metrics,
        "test_result_must_not_change_parameters": True,
    }
    write_json(root / "final_test_metrics.json", out)
    write_json(complete, {"complete": True, "metrics": (final_root_rel(config) / "final_test_metrics.json").as_posix()})
    return out


def finalize_test(config: dict[str, Any], allow_overwrite: bool) -> dict[str, Any]:
    if allow_overwrite:
        raise FileExistsError("--allow-overwrite cannot bypass the final-test one-shot contract.")
    require_final_confirmation()
    return finalize_test_core(config)


def resume_final_test_after_cf(config: dict[str, Any], resume_manifest: Path, allow_overwrite: bool) -> dict[str, Any]:
    if allow_overwrite:
        raise FileExistsError("--allow-overwrite cannot bypass the partial-resume one-shot contract.")
    require_final_resume_confirmation()
    inventory = input_inventory_preflight(config)
    partial = partial_state_audit(config, resume_manifest, require_expected_hashes=True)
    sas_paths = stream_test_paths(config, "auxiliary_stream")
    for path_key in ["predictions", "candidates", "candidate_report"]:
        resolve_repo_path(sas_paths[path_key], label=path_key).parent.mkdir(parents=True, exist_ok=True)
    plan = resume_command_plan(config)
    run_commands([
        plan["commands"]["sasrec_test_generation"],
        plan["commands"]["sasrec_test_candidate_eval"],
    ])
    validate_prediction_artifact(sas_paths["predictions"], sas_paths["eval_csv"], EXPECTED_TEST_ROWS)
    out = finalize_test_core(config)
    resume_closeout = {
        "schema": "s5_partial_resume_closeout.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resume_start_stage": "sasrec_test_generation",
        "input_inventory": inventory,
        "partial_state_before_resume": partial,
        "final_metrics": (final_root_rel(config) / "final_test_metrics.json").as_posix(),
        "no_parameter_change": True,
    }
    write_json(final_root(config) / "partial_resume_closeout_manifest.json", resume_closeout)
    return out


def make_release_manifest(config: dict[str, Any]) -> dict[str, Any]:
    verify_immutable_config(config)
    verify_release_hashes(config)
    plan = command_plan(config)
    manifest = {
        "schema": "s5_1_release_manifest.v1",
        "config_path": CONFIG_PATH_REL.as_posix(),
        "config_sha256": None,
        "category": CATEGORY,
        "frozen_selected_config": {
            "candidate_policy": config["selected_candidate_policy"],
            "lambda_sasrec": config["lambda_sasrec"],
            "source_bonus": config["source_bonus"],
            "ranker_compatibility_mode": config["ranker_compatibility_mode"],
            "candidate_mode": config["candidate_mode"],
            "num_beams": config["num_beams"],
            "max_new_tokens": config["max_new_tokens"],
            "seed": config["seed"],
        },
        "ranker": config["ranker"],
        "projection_contract": config["projection_contract"],
        "hashes": {
            "selected_config_hash": config["selected_config_hash"],
            "feature_schema_hash": config["ranker"]["feature_schema_hash"],
            "normalization_metadata_hash": config["ranker"]["normalization_metadata_hash"],
            "projection_contract_hash": config["projection_contract_hash"],
            "projection_implementation_hash": file_sha256_rel("scripts/s5_final_test_confirmation.py"),
            "s5_0_valid_select_leaderboard_hash": config["s5_0_evidence"]["valid_select_leaderboard_sha256"],
            "s5_0_valid_report_metrics_hash": config["s5_0_evidence"]["valid_report_metrics_sha256"],
            "s5_0_selected_config_file_hash": config["s5_0_evidence"]["selected_config_sha256"],
        },
        "source_artifacts": {
            "baseline_stream": config["baseline_stream"],
            "auxiliary_stream": config["auxiliary_stream"],
        },
        "command_plan": plan,
        "contracts": {
            "local_test_read": False,
            "final_test_single_execution": True,
            "test_result_must_not_change_parameters": True,
            "environment_overrides_allowed": False,
            "overwrite_default": "refuse",
            "project_root_env_var": PROJECT_ROOT_ENV,
            "base_model_env_var": BASE_MODEL_ENV,
            "path_identity": "repository-relative path plus content SHA256; runtime absolute paths are resolved from PROJECT_ROOT and are not immutable identity",
        },
    }
    return manifest


def make_sync_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "s5_1_autodl_sync_manifest.v1",
        "local_to_autodl_minimum_files": [
            "scripts/s5_final_test_confirmation.py",
            "scripts/run_s5_final_test_confirmation.sh",
            "scripts/s5_auxiliary_fusion_conversion.py",
            "scripts/s4_sasrec_sid_valid_pipeline.py",
            "evaluate.py",
            "evaluate_candidates.py",
            "LogitProcessor.py",
            "utils_sid.py",
            "configs/s5_auxiliary_fusion/frozen_release_config.json",
            "docs/s5_1_frozen_release_and_test_protocol.md",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/s5_1_release_manifest.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/s5_1_autodl_sync_manifest.json",
            "tests/test_s5_final_test_confirmation.py",
        ],
        "required_existing_autodl_artifacts": [
            config["baseline_stream"]["formal_checkpoint_path"],
            config["auxiliary_stream"]["formal_checkpoint_path"],
            "data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/test.csv",
            "data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/test.csv",
            config["ranker"]["model_path"],
        ],
        "runtime_environment": {
            "project_root_env_var": PROJECT_ROOT_ENV,
            "base_model_env_var": BASE_MODEL_ENV,
            "path_identity_contract": "store repository-relative path and content SHA256; resolve runtime absolute paths from PROJECT_ROOT on the target machine",
        },
        "return_after_final_test": [
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/final_test_metrics.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/final_test_complete_manifest.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/*/candidates/candidate_report.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/*/summary/*.json",
        ],
        "do_not_return_by_default_due_to_size": [
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/*/generation/predictions.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/*/candidates/candidates.jsonl",
        ],
        "safe_cleanup_after_backup": [
            "rm -rf results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/*/generation",
            "rm -f results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/*/candidates/candidates.jsonl",
        ],
        "never_delete_without_backup": [
            config["baseline_stream"]["formal_checkpoint_path"],
            config["auxiliary_stream"]["formal_checkpoint_path"],
        ],
    }


def make_partial_resume_manifest_template(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "s5_1_partial_resume_manifest_template.v1",
        "original_final_test_status": "cf_complete_sasrec_test_csv_missing_before_sasrec_generation",
        "original_failure_reason": "data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/test.csv missing",
        "non_root_cause": "LogitProcessor step-5 warning was not the crash root cause",
        "original_runner_config_sha256": file_sha256_rel(CONFIG_PATH_REL),
        "resume_runner_sha256": file_sha256_rel("scripts/s5_final_test_confirmation.py"),
        "resume_start_stage": "sasrec_test_generation",
        "expected_cf_artifact_sha256": {
            "cf_predictions": "<fill from AutoDL partial-state audit>",
            "cf_candidates": "<fill from AutoDL partial-state audit>",
            "cf_candidate_report": "<fill from AutoDL partial-state audit>",
        },
        "rebuilt_sasrec_test_csv_sha256": "<fill after deterministic SASRec test.csv rebuild>",
        "frozen_selected_config_hash": config["selected_config_hash"],
        "projection_hash": config["projection_contract_hash"],
        "frozen_ranker_hash": config["ranker"]["model_sha256"],
        "frozen_parameters": {
            "lambda_sasrec": config["lambda_sasrec"],
            "source_bonus": config["source_bonus"],
            "ranker_compatibility_mode": config["ranker_compatibility_mode"],
            "num_beams": config["num_beams"],
            "max_new_tokens": config["max_new_tokens"],
            "seed": config["seed"],
        },
        "no_parameter_change": True,
        "cf_artifacts_must_not_be_deleted_or_overwritten": True,
    }


def make_partial_resume_sync_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "s5_1_partial_resume_sync_manifest.v1",
        "local_to_autodl_minimum_files": [
            "scripts/s5_final_test_confirmation.py",
            "scripts/run_s5_final_test_confirmation.sh",
            "scripts/s5_auxiliary_fusion_conversion.py",
            "evaluate.py",
            "evaluate_candidates.py",
            "LogitProcessor.py",
            "utils_sid.py",
            "configs/s5_auxiliary_fusion/frozen_release_config.json",
            "docs/s5_1_partial_final_test_resume.md",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/s5_1_release_manifest.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/s5_1_partial_resume_manifest_template.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/s5_1_partial_resume_sync_manifest.json",
            "tests/test_s5_final_test_confirmation.py",
        ],
        "required_autodl_partial_state": [
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/cf_k512_dedup/generation/predictions.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/cf_k512_dedup/candidates/candidates.jsonl",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/cf_k512_dedup/candidates/candidate_report.json",
        ],
        "required_rebuilt_input": [
            "data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/test.csv",
        ],
        "forbidden_existing_before_resume": [
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/sasrec_v3_k512_dedup/generation/predictions.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/sasrec_v3_k512_dedup/candidates/candidates.jsonl",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/sasrec_v3_k512_dedup/candidates/candidate_report.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/final_test_metrics.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/final_test_complete_manifest.json",
        ],
        "return_after_resume": [
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/final_test_metrics.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/final_test_complete_manifest.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/partial_resume_closeout_manifest.json",
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/*/candidates/candidate_report.json",
        ],
        "do_not_delete": [
            "results/s5_auxiliary_fusion/Industrial_and_Scientific/final_frozen_test/cf_k512_dedup",
            config["baseline_stream"]["formal_checkpoint_path"],
            config["auxiliary_stream"]["formal_checkpoint_path"],
        ],
    }


def make_partial_resume_doc(config: dict[str, Any], template: dict[str, Any], sync: dict[str, Any]) -> str:
    plan = resume_command_plan(config)
    return "\n".join([
        "# S5-1 Partial Final-Test Resume",
        "",
        "## Root Cause",
        "",
        "1. `sasrec_v3_k512_dedup` test.csv was missing when SASRec test generation was about to start.",
        "2. The S5 dry-run/preflight did not check both baseline and treatment test.csv files before GPU generation.",
        "",
        "The LogitProcessor step-5 warning is retained as a warning but is not the crash root cause.",
        "",
        "## Allowed Partial State",
        "",
        "- CF generation and CF candidate evaluation complete.",
        "- SASRec generation/candidates/report absent.",
        "- final metrics and complete manifest absent.",
        "- CF artifacts are never deleted, quarantined, or overwritten by the resume path.",
        "",
        "## Resume Sequence",
        "",
        "```text",
        *plan["sequence"],
        "```",
        "",
        "## AutoDL Audit",
        "",
        "```bash",
        "export PROJECT_ROOT=\"$(pwd)\"",
        "export BASE_MODEL=\"<AutoDL base model path>\"",
        "DRY_RUN=1 bash scripts/run_s5_final_test_confirmation.sh --action audit-partial-final-test",
        "```",
        "",
        "## AutoDL Resume Dry-Run",
        "",
        "```bash",
        "DRY_RUN=1 bash scripts/run_s5_final_test_confirmation.sh --action resume-final-test-after-cf",
        "```",
        "",
        "## Unique Resume Command",
        "",
        "Run only after the partial-state audit, rebuilt SASRec test.csv validation, and concrete CF artifact hashes match the resume manifest:",
        "",
        "```bash",
        f"{CONFIRM_FINAL_TEST_RESUME_ENV}=1 DRY_RUN=0 bash scripts/run_s5_final_test_confirmation.sh --action resume-final-test-after-cf",
        "```",
        "",
        "## Return Files",
        "",
        "```text",
        *sync["return_after_resume"],
        "```",
        "",
        "No lambda, source bonus, projection, ranker, beam, token budget, seed, checkpoint, or SID parameter may be changed during resume.",
    ]) + "\n"


def make_doc(config: dict[str, Any], manifest: dict[str, Any], sync: dict[str, Any]) -> str:
    fc = manifest["frozen_selected_config"]
    zeroed = config["projection_contract"]["source_specific_features_zeroed_after_normalization"]
    retained = config["projection_contract"]["retained_normalized_features"]
    plan = manifest["command_plan"]["commands"]
    return "\n".join(
        [
            "# S5-1 Frozen Release and Final Test Protocol",
            "",
            "## Scope",
            "",
            "This release freezes the S5-0 valid-only auxiliary conversion result. Local work prepares manifests and dry-run commands only; it does not read or evaluate test labels.",
            "",
            "## Frozen Selected Config",
            "",
            f"- candidate policy: `{fc['candidate_policy']}`",
            f"- lambda_sasrec: `{fc['lambda_sasrec']}`",
            f"- source_bonus: `{fc['source_bonus']}`",
            f"- ranker compatibility: `{fc['ranker_compatibility_mode']}`",
            f"- candidate mode: `{fc['candidate_mode']}`",
            f"- num_beams: `{fc['num_beams']}`",
            f"- max_new_tokens: `{fc['max_new_tokens']}`",
            f"- seed: `{fc['seed']}`",
            "",
            "## Projection Contract",
            "",
            "Zeroing occurs **after normalization**. A zero value therefore means the frozen training-distribution mean for that feature, not raw missing/false.",
            "",
            f"- zeroed normalized source-specific features: `{zeroed}`",
            f"- retained normalized features: `{retained}`",
            f"- ordered feature schema hash: `{config['ranker']['feature_schema_hash']}`",
            f"- normalization metadata hash: `{config['ranker']['normalization_metadata_hash']}`",
            f"- ranker model hash: `{config['ranker']['model_sha256']}`",
            "",
            "## AutoDL Dry-Run Commands",
            "",
            "```bash",
            "export PROJECT_ROOT=\"$(pwd)\"",
            "export BASE_MODEL=\"<AutoDL base model path>\"",
            "DRY_RUN=1 bash scripts/run_s5_final_test_confirmation.sh --action dry-run",
            "python3 scripts/s5_final_test_confirmation.py --action audit",
            "```",
            "",
            "## AutoDL Final Test Command",
            "",
            "Run exactly once after dry-run and parity pass:",
            "",
            "```bash",
            "export PROJECT_ROOT=\"$(pwd)\"",
            "export BASE_MODEL=\"<AutoDL base model path>\"",
            "CONFIRM_FINAL_TEST=1 DRY_RUN=0 bash scripts/run_s5_final_test_confirmation.sh --action run-final-test",
            "```",
            "",
            "The final test result must not be used to change lambda, source bonus, projection, ranker checkpoint, or candidate policy.",
            "",
            "## Command Plan",
            "",
            "```text",
            *[f"{key}: {value}" for key, value in plan.items()],
            "```",
            "",
            "## Return Files",
            "",
            "```text",
            *sync["return_after_final_test"],
            "```",
            "",
            "## Safe Cleanup",
            "",
            "Only after the return bundle is verified:",
            "",
            "```bash",
            *sync["safe_cleanup_after_backup"],
            "```",
            "",
            "Do not delete formal final checkpoints unless the final test is complete and checkpoint backups are verified.",
        ]
    ) + "\n"


def freeze_release() -> dict[str, Any]:
    config = release_config()
    write_json(CONFIG_PATH, config)
    manifest = make_release_manifest(config)
    manifest["config_sha256"] = file_sha256_rel(CONFIG_PATH_REL)
    write_json(RELEASE_MANIFEST, manifest)
    sync = make_sync_manifest(config)
    write_json(SYNC_MANIFEST, sync)
    DOC_PATH.write_text(make_doc(config, manifest, sync), encoding="utf-8")
    partial_template = make_partial_resume_manifest_template(config)
    partial_sync = make_partial_resume_sync_manifest(config)
    write_json(PARTIAL_RESUME_TEMPLATE, partial_template)
    write_json(PARTIAL_RESUME_SYNC, partial_sync)
    PARTIAL_RESUME_DOC.write_text(make_partial_resume_doc(config, partial_template, partial_sync), encoding="utf-8")
    return {
        "config": CONFIG_PATH_REL.as_posix(),
        "release_manifest": RELEASE_MANIFEST_REL.as_posix(),
        "sync_manifest": SYNC_MANIFEST_REL.as_posix(),
        "doc": DOC_PATH_REL.as_posix(),
        "partial_resume_doc": PARTIAL_RESUME_DOC_REL.as_posix(),
        "partial_resume_template": PARTIAL_RESUME_TEMPLATE_REL.as_posix(),
        "partial_resume_sync_manifest": PARTIAL_RESUME_SYNC_REL.as_posix(),
    }


def load_config(path: Path) -> dict[str, Any]:
    config_path = path if path.is_absolute() else resolve_repo_path(path, label="config")
    if path.is_absolute() and not path.resolve().is_relative_to(PROJECT_ROOT):
        raise ValueError(f"config resolved outside PROJECT_ROOT: {path}")
    config = read_json(config_path)
    verify_immutable_config(config)
    return config


def run_commands(commands: list[str]) -> None:
    for command in commands:
        subprocess.run(command, shell=True, check=True, cwd=PROJECT_ROOT)


def run_final_test(config: dict[str, Any], allow_overwrite: bool) -> dict[str, Any]:
    require_final_confirmation()
    verify_release_hashes(config)
    protect_final_outputs(config, allow_overwrite)
    root = final_root(config)
    for stream_key in ["baseline_stream", "auxiliary_stream"]:
        paths = stream_test_paths(config, stream_key)
        for path_key in ["predictions", "candidates", "candidate_report"]:
            resolve_repo_path(paths[path_key], label=path_key).parent.mkdir(parents=True, exist_ok=True)
    plan = command_plan(config)
    sequence = [
        plan["commands"]["env_preflight"],
        plan["commands"]["parity"],
        plan["commands"]["cf_test_generation"],
        plan["commands"]["cf_test_candidate_eval"],
        plan["commands"]["sasrec_test_generation"],
        plan["commands"]["sasrec_test_candidate_eval"],
    ]
    run_commands(sequence)
    return finalize_test(config, allow_overwrite)


def main() -> None:
    args = parse_args()
    if args.action == "freeze-release":
        print(json.dumps(freeze_release(), indent=2, sort_keys=True))
        return
    config = load_config(args.config)
    if args.action in {"dry-run", "audit"}:
        verify_release_hashes(config)
        payload = {
            "action": args.action,
            "dry_run": True,
            "config": args.config.as_posix() if not args.config.is_absolute() else args.config.relative_to(PROJECT_ROOT).as_posix(),
            "config_sha256": file_sha256(resolve_repo_path(args.config, label="config") if not args.config.is_absolute() else args.config),
            "command_plan": command_plan(config),
            "final_test_requires": {"DRY_RUN": "0", "CONFIRM_FINAL_TEST": "1"},
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.action == "audit-partial-final-test":
        payload = {
            "action": args.action,
            "dry_run": True,
            "input_inventory": input_inventory_preflight(config),
            "partial_state": partial_state_audit(config, args.resume_manifest),
            "resume_command_plan": resume_command_plan(config),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.action == "resume-final-test-after-cf" and os.environ.get("DRY_RUN", "1") != "0":
        payload = {
            "action": args.action,
            "dry_run": True,
            "resume_command_plan": resume_command_plan(config),
            "final_resume_requires": {"DRY_RUN": "0", CONFIRM_FINAL_TEST_RESUME_ENV: "1"},
            "forbidden": ["baseline generation", "baseline candidate evaluation", "parameter changes", "overwrite"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.action == "run-final-test":
        print(json.dumps(run_final_test(config, args.allow_overwrite), indent=2, sort_keys=True))
        return
    if args.action == "finalize-test":
        print(json.dumps(finalize_test(config, args.allow_overwrite), indent=2, sort_keys=True))
        return
    if args.action == "resume-final-test-after-cf":
        print(json.dumps(resume_final_test_after_cf(config, args.resume_manifest, args.allow_overwrite), indent=2, sort_keys=True))
        return
    raise ValueError(args.action)


if __name__ == "__main__":
    main()
