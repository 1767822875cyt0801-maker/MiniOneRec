#!/usr/bin/env python3
"""Bounded S6 final artifact inventory.

Default mode is intentionally lightweight: it hashes only small allowlisted
files and records directories without recursive traversal. Use --full-hash
manually when a complete archive hash is explicitly needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific/s6_final_archive"
SMALL_FILE_LIMIT_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class ArtifactSpec:
    path: str
    stage: str
    status: str
    kind: str = "file"


ALLOWLIST = [
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/s6_checkpoint_compatibility_report.json", "S6-0", "canonical"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/s6_validation_split_manifest.json", "S6-0", "canonical"),
    ArtifactSpec("docs/s6_direct_sasrec_protocol.md", "S6-0", "canonical"),
    ArtifactSpec("export_sasrec_direct_candidates.py", "S6-1", "canonical"),
    ArtifactSpec("merge_direct_sasrec_candidates.py", "S6-1", "canonical"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/s6_2_smoke_audit_report.json", "S6-2", "canonical"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/s6_3_formal_direct_sasrec_audit_report.json", "S6-3", "canonical"),
    ArtifactSpec("incoming/s6_formal_direct_sasrec/s6_formal_direct_sasrec_formal_v1_bundle.tar.gz", "S6-3", "canonical"),
    ArtifactSpec("incoming/s6_formal_direct_sasrec/s6_formal_direct_sasrec_formal_v1_bundle.tar.gz.sha256", "S6-3", "canonical"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/cf_split_views/union_v1", "S6-4", "canonical", "directory"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/union/union_v1/valid_fit/k20", "S6-4", "canonical", "directory"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/union/union_v1/valid_select/k20", "S6-4", "canonical", "directory"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/union/union_v1/valid_gate/k20", "S6-4", "canonical", "directory"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/union/union_v1/valid_fit/k50", "S6-4", "ablation", "directory"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/union/union_v1/valid_fit/k100", "S6-4", "ablation", "directory"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/s6_4_union_validation_report.json", "S6-4", "canonical"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/s6_5_frozen_ranker_validation_report.json", "S6-5", "canonical"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/frozen_ranker/frozen_ranker_v1", "S6-5", "canonical", "directory"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/s6_6_lightweight_ranker_validation_report.json", "S6-6", "negative"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/lightweight_ranker/lightweight_v1", "S6-6", "ablation", "directory"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/s6_6r_direct_promotion_gate_report.json", "S6-6R", "negative"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/direct_promotion/promotion_v1", "S6-6R", "ablation", "directory"),
    ArtifactSpec("incoming/s6_qwen_cost_profile/cost_profile_qwen_sasrec_sid_v2_bundle.tar.gz", "S6-7B", "canonical"),
    ArtifactSpec("incoming/s6_qwen_cost_profile/cost_profile_qwen_sasrec_sid_v2_bundle.tar.gz.sha256", "S6-7B", "canonical"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/s6_cost_evidence_report.json", "S6-7B", "canonical"),
    ArtifactSpec("results/s6_cost_aware_aux/Industrial_and_Scientific/s6_stage_closeout_report.json", "S6-7", "canonical"),
    ArtifactSpec("docs/s6_stage_closeout.md", "S6-7", "canonical"),
    ArtifactSpec("docs/s6_cost_evidence_closeout.md", "S6-7B", "canonical"),
    ArtifactSpec("docs/s6_final", "S6-final", "canonical", "directory"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rel_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {value}")
    if any(part == ".." for part in path.parts):
        raise ValueError(f"path traversal is not allowed: {value}")
    lowered = [part.lower() for part in path.parts]
    if "test" in lowered or "final_test" in lowered or path.name.lower() == "test.csv":
        raise ValueError(f"S6 final inventory refuses test path: {value}")
    return path


def git_state(rel_path: str) -> dict[str, Any]:
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel_path], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    ignored = subprocess.run(["git", "check-ignore", "-q", rel_path], cwd=ROOT).returncode == 0
    return {"tracked": tracked, "ignored": ignored}


def directory_tree_hash(path: Path) -> dict[str, Any]:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for file_path in files:
        rel = file_path.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(sha256(file_path).encode("ascii") + b"\0")
    return {"file_count": len(files), "sha256_tree": digest.hexdigest()}


def inventory_entry(spec: ArtifactSpec, *, full_hash: bool) -> dict[str, Any]:
    rel = normalize_rel_path(spec.path)
    abs_path = ROOT / rel
    item: dict[str, Any] = {
        "relative_path": rel.as_posix(),
        "stage": spec.stage,
        "status": spec.status,
        "expected_kind": spec.kind,
        "exists": abs_path.exists(),
        "git": git_state(rel.as_posix()),
    }
    if not abs_path.exists():
        item.update({"type": "missing", "size_bytes": None, "sha256": None})
        return item
    if abs_path.is_file():
        size = abs_path.stat().st_size
        item.update({"type": "file", "size_bytes": size})
        item["sha256"] = sha256(abs_path) if full_hash or size <= SMALL_FILE_LIMIT_BYTES else None
        item["hash_policy"] = "full_hash" if full_hash else ("small_file" if item["sha256"] else "skipped_large_file")
        return item
    if abs_path.is_dir():
        item.update({"type": "directory", "size_bytes": None, "sha256": None, "hash_policy": "directory_not_recursed"})
        if full_hash:
            item.update(directory_tree_hash(abs_path))
            item["hash_policy"] = "full_hash_directory_tree"
        return item
    item.update({"type": "other", "size_bytes": None, "sha256": None, "hash_policy": "not_hashed"})
    return item


def build_inventory(*, full_hash: bool) -> dict[str, Any]:
    entries = [inventory_entry(spec, full_hash=full_hash) for spec in sorted(ALLOWLIST, key=lambda s: s.path)]
    return {
        "schema": "s6_final_artifact_inventory.v1",
        "full_hash": full_hash,
        "small_file_limit_bytes": SMALL_FILE_LIMIT_BYTES,
        "entry_count": len(entries),
        "entries": entries,
        "missing_count": sum(not entry["exists"] for entry in entries),
        "test_read": False,
    }


def write_outputs(out_dir: Path, inventory: dict[str, Any], *, overwrite: bool) -> None:
    paths = {
        "json": out_dir / "artifact_inventory.json",
        "md": out_dir / "artifact_inventory.md",
        "sha": out_dir / "small_files.sha256",
    }
    if not overwrite:
        existing = [path for path in paths.values() if path.exists() and path.stat().st_size > 0]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing inventory outputs: {[p.as_posix() for p in existing]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# S6 Final Artifact Inventory",
        "",
        f"- full_hash: `{inventory['full_hash']}`",
        f"- entries: `{inventory['entry_count']}`",
        f"- missing: `{inventory['missing_count']}`",
        "",
        "| Path | Stage | Status | Exists | Type | Hash policy | Git tracked | Git ignored |",
        "|---|---|---|---:|---|---|---:|---:|",
    ]
    for entry in inventory["entries"]:
        lines.append(
            f"| `{entry['relative_path']}` | {entry['stage']} | {entry['status']} | {entry['exists']} | {entry['type']} | {entry.get('hash_policy')} | {entry['git']['tracked']} | {entry['git']['ignored']} |"
        )
    paths["md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    sha_lines = [
        f"{entry['sha256']}  {entry['relative_path']}"
        for entry in inventory["entries"]
        if entry.get("sha256") and entry["type"] == "file"
    ]
    paths["sha"].write_text("\n".join(sha_lines) + ("\n" if sha_lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate bounded S6 final artifact inventory.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = build_inventory(full_hash=args.full_hash)
    write_outputs(args.out_dir, inventory, overwrite=args.overwrite)
    print(json.dumps({"inventory": (args.out_dir / "artifact_inventory.json").as_posix(), "entries": inventory["entry_count"], "missing": inventory["missing_count"], "full_hash": args.full_hash}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
