#!/usr/bin/env python3
"""Build split-safe SASRec v3 Behavior-SID artifacts.

This is an orchestration layer around the existing MiniOneRec SID contract:
Residual MiniBatchKMeans, ``<a_i><b_j><c_k>`` tokens, optional ``<d_i>``
dedup suffix, and train/valid CSV rewriting.  It intentionally refuses test
paths by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils_sid import load_json, parse_sid_tokens, save_json  # noqa: E402


DEFAULT_CATEGORY = "Industrial_and_Scientific"
DEFAULT_SASREC_VERSION = "formal_v3_finite_maskfix_seed42"
DEFAULT_SID_VERSION = "sasrec_v3_k512_dedup"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SASRec v3 Behavior-SID artifacts without reading test.")
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--config-mode", choices=["smoke", "formal"], default="formal")
    parser.add_argument("--sasrec-dir", type=Path, default=None)
    parser.add_argument("--sid-version", default=DEFAULT_SID_VERSION)
    parser.add_argument("--output-root", type=Path, default=Path("data/Amazon/sid_versions"))
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--valid-csv", type=Path, default=None)
    parser.add_argument("--item-json", type=Path, default=None)
    parser.add_argument("--num-levels", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dedup-mode", choices=["append"], default="append")
    parser.add_argument("--expected-num-items", type=int, default=3686)
    parser.add_argument("--expected-dim", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def path_has_forbidden_split(path: Path) -> bool:
    return any(part.lower() in {"test", "test.csv"} for part in path.parts)


def assert_not_test_path(path: Path, label: str) -> None:
    if path_has_forbidden_split(path):
        raise ValueError(f"{label} must not point at test data: {path}")


def assert_not_v2_path(path: Path, label: str) -> None:
    if "formal_v2_no_leakage" in path.as_posix():
        raise ValueError(f"{label} must use SASRec v3, not v2: {path}")


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    category = args.category
    sasrec_dir = args.sasrec_dir or Path("data/Amazon/behavior_embeddings/sasrec") / category / DEFAULT_SASREC_VERSION
    return {
        "sasrec_dir": sasrec_dir,
        "embedding": sasrec_dir / f"{category}.sasrec_item_emb.npy",
        "item_order": sasrec_dir / f"{category}.item_order.json",
        "row_index": sasrec_dir / f"{category}.row_index.json",
        "train_item_count": sasrec_dir / f"{category}.train_item_count.json",
        "artifact_manifest": sasrec_dir / "artifact_manifest.json",
        "config": sasrec_dir / "config.json",
        "metrics_summary": sasrec_dir / "metrics_summary.json",
        "cold_item_report": sasrec_dir / f"{category}.cold_item_report.json",
        "train_csv": args.train_csv or Path("data/Amazon/train") / f"{category}_5_2016-10-2018-11.csv",
        "valid_csv": args.valid_csv or Path("data/Amazon/valid") / f"{category}_5_2016-10-2018-11.csv",
        "item_json": args.item_json or Path("data/Amazon/index") / f"{category}.item.json",
        "output_dir": args.output_root / args.sid_version / category,
    }


def validate_item_mapping(item_order: list[str], row_index: dict[str, int], num_rows: int) -> None:
    if len(item_order) != num_rows:
        raise ValueError(f"item_order length={len(item_order)} does not match embedding rows={num_rows}")
    if len(set(item_order)) != len(item_order):
        raise ValueError("item_order contains duplicate item ids")
    if set(row_index.keys()) != set(item_order):
        raise ValueError("row_index keys do not exactly match item_order")
    expected_rows = set(range(num_rows))
    actual_rows = set(int(value) for value in row_index.values())
    if actual_rows != expected_rows:
        raise ValueError("row_index is not contiguous from 0 to num_items-1")
    for row, item_id in enumerate(item_order):
        if int(row_index[item_id]) != row:
            raise ValueError(f"row_index mismatch for item_id={item_id}: expected {row}, got {row_index[item_id]}")


def validate_manifest(paths: dict[str, Path], manifest: dict[str, Any], expected_num_items: int, expected_dim: int) -> list[str]:
    warnings: list[str] = []
    if manifest.get("embedding_source") != "train_only_sasrec":
        raise ValueError(f"Unexpected embedding_source: {manifest.get('embedding_source')}")
    if int(manifest.get("dimension", -1)) != expected_dim:
        raise ValueError(f"Unexpected manifest dimension: {manifest.get('dimension')}")
    if int(manifest.get("num_items", -1)) != expected_num_items:
        raise ValueError(f"Unexpected manifest num_items: {manifest.get('num_items')}")
    if manifest.get("normalization") != "l2":
        raise ValueError(f"Unexpected normalization: {manifest.get('normalization')}")
    policy = manifest.get("train_only_policy", {})
    if policy.get("test_read") is not False:
        raise ValueError("artifact_manifest train_only_policy.test_read must be false")

    hashes = manifest.get("hashes", {})
    expected_hash_keys = {
        "embedding": paths["embedding"],
        "item_order": paths["item_order"],
        "row_index": paths["row_index"],
        "train_item_count": paths["train_item_count"],
        "config": paths["config"],
        "metrics_summary": paths["metrics_summary"],
    }
    for key, path in expected_hash_keys.items():
        expected = hashes.get(key)
        if expected is None:
            raise ValueError(f"artifact_manifest missing hash for {key}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {key}: manifest={expected} actual={actual}")
    if "cold_item_report" in hashes and not paths["cold_item_report"].exists():
        warnings.append("artifact_manifest references cold_item_report, but local file is missing; deriving cold stats from train_item_count")
    return warnings


def validate_sasrec_input(paths: dict[str, Path], expected_num_items: int, expected_dim: int) -> dict[str, Any]:
    for label in ["sasrec_dir", "embedding", "item_order", "row_index", "train_item_count", "artifact_manifest", "config", "metrics_summary", "train_csv", "valid_csv", "item_json"]:
        assert_not_test_path(paths[label], label)
        assert_not_v2_path(paths[label], label)
        if not paths[label].exists():
            raise FileNotFoundError(f"Required {label} path is missing: {paths[label]}")

    matches = sorted(paths["sasrec_dir"].glob("*.sasrec_item_emb.npy"))
    if matches != [paths["embedding"]]:
        raise ValueError(f"Expected exactly one SASRec embedding at {paths['embedding']}, found: {matches}")

    embedding = np.load(paths["embedding"])
    if tuple(embedding.shape) != (expected_num_items, expected_dim):
        raise ValueError(f"Expected embedding shape {(expected_num_items, expected_dim)}, got {tuple(embedding.shape)}")
    if embedding.dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError(f"Unexpected embedding dtype: {embedding.dtype}")
    if not np.isfinite(embedding).all():
        raise ValueError("SASRec embedding contains NaN/Inf")
    norms = np.linalg.norm(embedding, axis=1)
    zero_rows = np.where(norms <= 1e-12)[0].tolist()
    if zero_rows:
        raise ValueError(f"SASRec embedding contains all-zero rows: {zero_rows[:20]}")

    item_order = [str(item) for item in load_json(paths["item_order"])]
    row_index = {str(item): int(row) for item, row in load_json(paths["row_index"]).items()}
    validate_item_mapping(item_order, row_index, embedding.shape[0])
    manifest = load_json(paths["artifact_manifest"])
    warnings = validate_manifest(paths, manifest, expected_num_items, expected_dim)
    counts = {str(item): int(count) for item, count in load_json(paths["train_item_count"]).items()}
    cold_items = [item for item in item_order if int(counts.get(item, 0)) == 0]

    return {
        "embedding_shape": [int(x) for x in embedding.shape],
        "embedding_dtype": str(embedding.dtype),
        "embedding_sha256": sha256(paths["embedding"]),
        "finite": bool(np.isfinite(embedding).all()),
        "zero_rows": len(zero_rows),
        "l2_norm": {"min": float(norms.min()), "mean": float(norms.mean()), "max": float(norms.max())},
        "item_order": item_order,
        "row_index": row_index,
        "train_item_count": counts,
        "cold_items": cold_items,
        "manifest": manifest,
        "warnings": warnings,
    }


def refuse_existing_formal_outputs(output_dir: Path, overwrite: bool) -> None:
    formal_outputs = [
        output_dir / "index.json",
        output_dir / "item2sid.json",
        output_dir / "sid2items.json",
        output_dir / "train.csv",
        output_dir / "valid.csv",
        output_dir / "reports" / "s3_static_quality_report.json",
    ]
    existing = [path for path in formal_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory already contains formal artifacts under {output_dir}; "
            f"refusing to overwrite: {[p.as_posix() for p in existing[:8]]}"
        )


def run_command(cmd: list[str]) -> None:
    print("Command:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def prefix_from_tokens(tokens: list[str], levels: int = 3) -> str:
    return "".join(tokens[:levels])


def entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts:
        if count:
            p = count / total
            out -= p * math.log2(p)
    return out


def level_stats(codes: np.ndarray) -> list[dict[str, Any]]:
    stats = []
    for level in range(codes.shape[1]):
        level_codes = codes[:, level]
        distribution = Counter(int(value) for value in level_codes.tolist())
        ent = entropy(list(distribution.values()))
        stats.append(
            {
                "level": level,
                "effective_codes": len(distribution),
                "entropy": ent,
                "perplexity": float(2**ent),
                "max_bucket_size": max(distribution.values()) if distribution else 0,
                "assignment_distribution": {str(k): int(v) for k, v in sorted(distribution.items())},
            }
        )
    return stats


def collision_stats(item2sid: dict[str, str], prefix_levels: int = 3) -> dict[str, Any]:
    full = defaultdict(list)
    prefix = defaultdict(list)
    dedup_count = 0
    for item, sid in item2sid.items():
        tokens = parse_sid_tokens(sid)
        full[sid].append(item)
        prefix[prefix_from_tokens(tokens, prefix_levels)].append(item)
        dedup_count += int(any(token.startswith("<d_") for token in tokens))
    full_buckets = [items for items in full.values() if len(items) > 1]
    prefix_buckets = [items for items in prefix.values() if len(items) > 1]
    return {
        "full_unique": len(full),
        "full_collision_items": sum(len(items) for items in full_buckets),
        "full_collision_buckets": len(full_buckets),
        "prefix_unique": len(prefix),
        "prefix_collision_items": sum(len(items) for items in prefix_buckets),
        "prefix_collision_buckets": len(prefix_buckets),
        "dedup_token_items": dedup_count,
        "dedup_token_rate": dedup_count / len(item2sid) if item2sid else 0.0,
        "max_full_collision_bucket": max((len(items) for items in full.values()), default=0),
        "max_prefix_collision_bucket": max((len(items) for items in prefix.values()), default=0),
    }


def cold_item_analysis(
    embedding: np.ndarray,
    item_order: list[str],
    cold_items: list[str],
    item2sid: dict[str, str],
    codes: np.ndarray,
    train_counts: dict[str, int],
    fallback_type: str,
) -> dict[str, Any]:
    cold_rows = [item_order.index(item) for item in cold_items]
    cold_matrix = embedding[cold_rows] if cold_rows else np.zeros((0, embedding.shape[1]), dtype=embedding.dtype)
    unique_rows = np.unique(cold_matrix, axis=0).shape[0] if cold_rows else 0
    rounded_unique_rows = np.unique(np.round(cold_matrix, decimals=7), axis=0).shape[0] if cold_rows else 0
    exact_pairs = 0
    approx_pairs = 0
    for i in range(len(cold_rows)):
        for j in range(i + 1, len(cold_rows)):
            exact_pairs += int(np.array_equal(cold_matrix[i], cold_matrix[j]))
            approx_pairs += int(np.allclose(cold_matrix[i], cold_matrix[j], rtol=1e-6, atol=1e-7))

    cold_item2sid = {item: item2sid[item] for item in cold_items}
    cold_full = collision_stats(cold_item2sid) if cold_item2sid else {}
    prefix_groups = defaultdict(list)
    for item, sid in cold_item2sid.items():
        prefix_groups[prefix_from_tokens(parse_sid_tokens(sid), 3)].append(item)
    only_dedup_items = sum(1 for items in prefix_groups.values() if len(items) > 1 for _ in items)

    warm_rows = [idx for idx, item in enumerate(item_order) if int(train_counts.get(item, 0)) > 0]
    def usage(rows: list[int]) -> list[int]:
        if not rows:
            return [0 for _ in range(codes.shape[1])]
        return [int(len(set(codes[rows, level].tolist()))) for level in range(codes.shape[1])]

    return {
        "cold_fallback_type": fallback_type,
        "cold_item_count": len(cold_items),
        "cold_item_ids": cold_items,
        "cold_embedding_unique_rows_exact": int(unique_rows),
        "cold_embedding_unique_rows_rounded_1e_7": int(rounded_unique_rows),
        "cold_embedding_exact_duplicate_pairs": exact_pairs,
        "cold_embedding_approx_duplicate_pairs": approx_pairs,
        "cold_full_sid_collision_items": cold_full.get("full_collision_items", 0),
        "cold_full_sid_collision_buckets": cold_full.get("full_collision_buckets", 0),
        "cold_prefix_unique": cold_full.get("prefix_unique", 0),
        "cold_prefix_collision_items": cold_full.get("prefix_collision_items", 0),
        "cold_prefix_collision_buckets": cold_full.get("prefix_collision_buckets", 0),
        "cold_items_only_distinguished_by_dedup": only_dedup_items,
        "warm_code_usage_by_level": usage(warm_rows),
        "cold_code_usage_by_level": usage(cold_rows),
    }


def validate_rewritten_csv(input_csv: Path, output_csv: Path, item2sid: dict[str, str]) -> dict[str, Any]:
    assert_not_test_path(input_csv, "input_csv")
    assert_not_test_path(output_csv, "output_csv")
    with open(input_csv, "r", encoding="utf-8", newline="") as f:
        input_rows = list(csv.DictReader(f))
    with open(output_csv, "r", encoding="utf-8", newline="") as f:
        output_rows = list(csv.DictReader(f))
    if len(input_rows) != len(output_rows):
        raise ValueError(f"row count mismatch for {output_csv}: {len(input_rows)} != {len(output_rows)}")
    bad_rows = []
    for idx, (src, out) in enumerate(zip(input_rows, output_rows)):
        for col in ["user_id", "history_item_id", "item_id"]:
            if src.get(col) != out.get(col):
                bad_rows.append({"row": idx, "column": col})
                break
        if out.get("item_sid") != item2sid.get(str(out.get("item_id"))):
            bad_rows.append({"row": idx, "column": "item_sid"})
            break
        history_items = _parse_literal_list(out.get("history_item_id", "[]"))
        history_sids = _parse_literal_list(out.get("history_item_sid", "[]"))
        if len(history_items) != len(history_sids):
            bad_rows.append({"row": idx, "column": "history_length"})
            break
        for item, sid in zip(history_items, history_sids):
            if item2sid.get(str(item)) != str(sid):
                bad_rows.append({"row": idx, "column": "history_item_sid"})
                break
    if bad_rows:
        raise ValueError(f"CSV rewrite validation failed for {output_csv}: {bad_rows[:5]}")
    return {"input_csv": input_csv.as_posix(), "output_csv": output_csv.as_posix(), "rows": len(output_rows), "ok": True}


def _parse_literal_list(text: Any) -> list[Any]:
    import ast

    value = "" if text is None else str(text).strip()
    if not value:
        return []
    parsed = ast.literal_eval(value)
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def build_static_report(
    args: argparse.Namespace,
    paths: dict[str, Path],
    input_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    embedding = np.load(paths["embedding"])
    item_order = input_report["item_order"]
    train_counts = input_report["train_item_count"]
    item2sid = {str(k): str(v) for k, v in load_json(output_dir / "item2sid.json").items()}
    codes = np.load(output_dir / "codes.npy")
    generation_report = load_json(output_dir / "reports" / "generation_report.json")
    cf_report_path = Path("data/Amazon/sid_versions/cf_k512_dedup") / args.category / "reports" / "generation_report.json"
    cf_report = load_json(cf_report_path) if cf_report_path.exists() else None
    fallback_type = "pure_sasrec_mean_cold_baseline" if len(input_report["cold_items"]) else "not_applicable"
    cold = cold_item_analysis(
        embedding=embedding,
        item_order=item_order,
        cold_items=input_report["cold_items"],
        item2sid=item2sid,
        codes=codes,
        train_counts=train_counts,
        fallback_type=fallback_type,
    )
    collisions = collision_stats(item2sid)
    report = {
        "category": args.category,
        "sid_version": args.sid_version,
        "method": "sasrec_v3_residual_minibatch_kmeans",
        "source_embedding": "formal_v3_finite_maskfix_seed42",
        "split_policy": {"test_read": False, "rewritten_splits": ["train", "valid"]},
        "seed": args.seed,
        "command_line": " ".join(sys.argv),
        "git": git_info(),
        "inputs": {
            key: path.as_posix()
            for key, path in paths.items()
            if key not in {"sasrec_dir", "output_dir"}
        },
        "input_sha256": {
            "embedding": sha256(paths["embedding"]),
            "item_order": sha256(paths["item_order"]),
            "row_index": sha256(paths["row_index"]),
            "train_item_count": sha256(paths["train_item_count"]),
            "artifact_manifest": sha256(paths["artifact_manifest"]),
        },
        "num_items": len(item2sid),
        "num_levels": args.num_levels,
        "codebook_size": args.codebook_size,
        "dedup_mode": args.dedup_mode,
        "level_stats": level_stats(codes),
        "generation_report": generation_report,
        "collision_stats": collisions,
        "reconstruction_error": {
            "reconstruction_mse": generation_report.get("reconstruction_mse"),
            "residual_norm_by_level": generation_report.get("residual_norm_by_level"),
        },
        "dedup": generation_report.get("dedup", {}),
        "cold_item_analysis": cold,
        "cf_k512_dedup_comparison": {
            "available": cf_report is not None,
            "cf_generation_report": cf_report_path.as_posix() if cf_report_path.exists() else "missing",
            "cf_num_levels": None if cf_report is None else cf_report.get("num_levels"),
            "cf_codebook_size": None if cf_report is None else cf_report.get("codebook_size"),
            "cf_pre_dedup": None if cf_report is None else cf_report.get("pre_dedup"),
            "cf_post_dedup": None if cf_report is None else cf_report.get("post_dedup"),
            "cf_dedup": None if cf_report is None else cf_report.get("dedup"),
            "cf_reconstruction_mse": None if cf_report is None else cf_report.get("reconstruction_mse"),
        },
        "output_sha256": {
            "item2sid": sha256(output_dir / "item2sid.json"),
            "sid2items": sha256(output_dir / "sid2items.json"),
            "index": sha256(output_dir / "index.json"),
            "codes": sha256(output_dir / "codes.npy"),
            "codebooks": sha256(output_dir / "codebooks.npz"),
        },
        "warnings": input_report["warnings"],
    }
    save_json(report, output_dir / "reports" / "s3_static_quality_report.json")
    write_text(output_dir / "reports" / "s3_static_quality_report.md", render_markdown_report(report))
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    cold = report["cold_item_analysis"]
    collision = report["collision_stats"]
    return "\n".join(
        [
            f"# SASRec v3 Behavior-SID Static Quality: {report['category']}",
            "",
            f"- sid_version: `{report['sid_version']}`",
            f"- num_items: `{report['num_items']}`",
            f"- levels/codebook: `{report['num_levels']} x {report['codebook_size']}`",
            f"- full_unique: `{collision['full_unique']}`",
            f"- full_collision_items: `{collision['full_collision_items']}`",
            f"- prefix_unique: `{collision['prefix_unique']}`",
            f"- prefix_collision_items: `{collision['prefix_collision_items']}`",
            f"- dedup_token_rate: `{collision['dedup_token_rate']}`",
            f"- reconstruction_mse: `{report['reconstruction_error']['reconstruction_mse']}`",
            "",
            "## Cold Items",
            "",
            f"- cold_item_count: `{cold['cold_item_count']}`",
            f"- fallback_type: `{cold['cold_fallback_type']}`",
            f"- cold_embedding_unique_rows_exact: `{cold['cold_embedding_unique_rows_exact']}`",
            f"- cold_prefix_collision_items: `{cold['cold_prefix_collision_items']}`",
            f"- cold_items_only_distinguished_by_dedup: `{cold['cold_items_only_distinguished_by_dedup']}`",
            "",
            "## Split Policy",
            "",
            "- train/valid rewritten only.",
            "- test is not read or generated in this S3 build.",
            "- pure SASRec mean cold fallback is reported as a baseline, not personalized behavior.",
            "",
        ]
    )


def git_info() -> dict[str, Any]:
    def run(parts: list[str]) -> str:
        try:
            return subprocess.check_output(parts, cwd=ROOT, text=True).strip()
        except Exception as exc:
            return f"not_available: {exc}"

    return {"commit": run(["git", "rev-parse", "--short", "HEAD"]), "status_short": run(["git", "status", "--short"])}


def write_s3_manifest(output_dir: Path, report: dict[str, Any], csv_validation: dict[str, Any]) -> None:
    manifest = {
        "sid_version": report["sid_version"],
        "category": report["category"],
        "source_embedding": report["source_embedding"],
        "test_read": False,
        "paths": {
            "output_dir": output_dir.as_posix(),
            "item2sid": (output_dir / "item2sid.json").as_posix(),
            "sid2items": (output_dir / "sid2items.json").as_posix(),
            "valid_sid_set": (output_dir / "valid_sid_set.json").as_posix(),
            "item_mapping": (output_dir / "item_mapping.json").as_posix(),
            "train_csv": (output_dir / "train.csv").as_posix(),
            "valid_csv": (output_dir / "valid.csv").as_posix(),
            "static_quality_report": (output_dir / "reports" / "s3_static_quality_report.json").as_posix(),
        },
        "csv_validation": csv_validation,
        "acceptance": {
            "post_dedup_collision_zero": report["collision_stats"]["full_collision_items"] == 0,
            "dedup_unique_item_mapping": report["collision_stats"]["full_unique"] == report["num_items"],
            "cold_collision_recorded": True,
            "test_not_read": True,
        },
        "output_sha256": {
            "item2sid": sha256(output_dir / "item2sid.json"),
            "train_csv": sha256(output_dir / "train.csv"),
            "valid_csv": sha256(output_dir / "valid.csv"),
            "static_quality_report": sha256(output_dir / "reports" / "s3_static_quality_report.json"),
        },
    }
    save_json(manifest, output_dir / "reports" / "s3_manifest.json")


def build(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_paths(args)
    input_report = validate_sasrec_input(paths, args.expected_num_items, args.expected_dim)
    refuse_existing_formal_outputs(paths["output_dir"], args.overwrite)
    max_iter = args.max_iter if args.max_iter is not None else (5 if args.config_mode == "smoke" else 100)
    planned = {
        "category": args.category,
        "config_mode": args.config_mode,
        "sid_version": args.sid_version,
        "output_dir": paths["output_dir"].as_posix(),
        "embedding": paths["embedding"].as_posix(),
        "embedding_shape": input_report["embedding_shape"],
        "embedding_dtype": input_report["embedding_dtype"],
        "cold_item_count": len(input_report["cold_items"]),
        "num_levels": args.num_levels,
        "codebook_size": args.codebook_size,
        "max_iter": max_iter,
        "dedup_mode": args.dedup_mode,
        "test_read": False,
        "warnings": input_report["warnings"],
    }
    if args.dry_run:
        print("S3 SASRec Behavior-SID DRY_RUN")
        print(json.dumps(planned, indent=2, sort_keys=True))
        return {"planned": planned, "dry_run": True}

    run_command(
        [
            sys.executable,
            "run_rqkmeans_with_emb.py",
            "--category",
            args.category,
            "--sid-version",
            args.sid_version,
            "--emb-path",
            paths["embedding"].as_posix(),
            "--item-order",
            paths["item_order"].as_posix(),
            "--row-index",
            paths["row_index"].as_posix(),
            "--item-json",
            paths["item_json"].as_posix(),
            "--output-root",
            args.output_root.as_posix(),
            "--num-levels",
            str(args.num_levels),
            "--codebook-size",
            str(args.codebook_size),
            "--max-iter",
            str(max_iter),
            "--seed",
            str(args.seed),
            "--dedup-mode",
            args.dedup_mode,
            *(["--batch-size", str(args.batch_size)] if args.batch_size else []),
            *(["--overwrite"] if args.overwrite else []),
        ]
    )
    output_dir = paths["output_dir"]
    item2sid = {str(k): str(v) for k, v in load_json(output_dir / "item2sid.json").items()}
    report_dir = output_dir / "reports"
    for split in ["train", "valid"]:
        run_command(
            [
                sys.executable,
                "rewrite_sid_csv.py",
                "--input-csv",
                paths[f"{split}_csv"].as_posix(),
                "--output-csv",
                (output_dir / f"{split}.csv").as_posix(),
                "--item2sid",
                (output_dir / "item2sid.json").as_posix(),
                "--report-path",
                (report_dir / f"rewrite_{split}_report.json").as_posix(),
                *(["--overwrite"] if args.overwrite else []),
            ]
        )
    static_report = build_static_report(args, paths, input_report, output_dir)
    csv_validation = {
        split: validate_rewritten_csv(paths[f"{split}_csv"], output_dir / f"{split}.csv", item2sid)
        for split in ["train", "valid"]
    }
    write_s3_manifest(output_dir, static_report, csv_validation)
    print("S3 SASRec Behavior-SID build completed")
    print(json.dumps({"output_dir": output_dir.as_posix(), "static_quality": static_report["collision_stats"]}, indent=2, sort_keys=True))
    return {"planned": planned, "static_report": static_report, "csv_validation": csv_validation}


def main() -> None:
    args = parse_args()
    build(args)


if __name__ == "__main__":
    main()
