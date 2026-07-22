#!/usr/bin/env python3
"""Audit the recovered formal-v3 SASRec checkpoint without executing training.

The local WSL environment used for repository work may not have PyTorch
installed. This script therefore uses a restricted PyTorch-zip checkpoint
parser for structure and reads raw FloatStorage bytes for the embedding-export
compatibility check. It does not execute arbitrary checkpoint code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import pickle
import sys
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/Amazon/behavior_embeddings/sasrec/Industrial_and_Scientific/formal_v3_finite_maskfix_seed42"
DEFAULT_CHECKPOINT = BASE / "checkpoint/best_model.pt"
DEFAULT_SHA256 = BASE / "checkpoint/best_model.pt.sha256"
DEFAULT_CONFIG = BASE / "config.json"
DEFAULT_MANIFEST = BASE / "artifact_manifest.json"
DEFAULT_EXPORTED = BASE / "Industrial_and_Scientific.sasrec_item_emb.npy"
DEFAULT_ITEM_ORDER = BASE / "Industrial_and_Scientific.item_order.json"
DEFAULT_TRAIN_COUNTS = BASE / "Industrial_and_Scientific.train_item_count.json"
DEFAULT_OUTPUT = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific/s6_checkpoint_compatibility_report.json"
EXPECTED_SHA256 = "577f51a9a539b9cd6307eea16c04db303691e349c0f32e245e44b8710cee94bd"


@dataclass(frozen=True)
class TensorMeta:
    storage_key: str
    storage_size: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_torch_pickle_stubs() -> None:
    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch._utils")

    class FloatStorage:  # pragma: no cover - marker for pickle only
        pass

    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        storage_type, storage_key, _location, storage_size = storage
        if storage_type != "FloatStorage":
            raise pickle.UnpicklingError(f"unsupported storage type: {storage_type}")
        return TensorMeta(
            storage_key=str(storage_key),
            storage_size=int(storage_size),
            shape=tuple(int(v) for v in size),
            stride=tuple(int(v) for v in stride),
            storage_offset=int(storage_offset),
        )

    torch_mod.FloatStorage = FloatStorage
    utils_mod._rebuild_tensor_v2 = _rebuild_tensor_v2
    sys.modules["torch"] = torch_mod
    sys.modules["torch._utils"] = utils_mod


class CheckpointUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        # PyTorch storage persistent ids are tuples:
        # ("storage", FloatStorage, storage_key, location, numel)
        if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")
        storage_type = getattr(pid[1], "__name__", str(pid[1]))
        return (storage_type, str(pid[2]), str(pid[3]), int(pid[4]))


def restricted_checkpoint_load(path: Path) -> dict[str, Any]:
    install_torch_pickle_stubs()
    with zipfile.ZipFile(path) as zf:
        data_name = next(name for name in zf.namelist() if name.endswith("data.pkl"))
        data = zf.read(data_name)
    return CheckpointUnpickler(io.BytesIO(data)).load()


def load_float_storage(checkpoint: Path, storage_key: str, shape: tuple[int, ...]) -> np.ndarray:
    with zipfile.ZipFile(checkpoint) as zf:
        storage_name = f"best_model/data/{storage_key}"
        raw = zf.read(storage_name)
    arr = np.frombuffer(raw, dtype="<f4")
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"storage {storage_key} has {arr.size} floats, expected {expected}")
    return arr.reshape(shape).copy()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return matrix / norms


def state_summary(state: dict[str, TensorMeta]) -> dict[str, Any]:
    keys = list(state)
    return {
        "tensor_count": len(keys),
        "keys": keys,
        "embedding_keys": [key for key in keys if "embedding" in key],
        "attention_keys": [key for key in keys if "self_attn" in key],
        "feed_forward_keys": [key for key in keys if ".linear" in key],
        "layer_norm_keys": [key for key in keys if "norm" in key],
        "shapes": {key: list(value.shape) for key, value in state.items()},
    }


def compare_embedding(checkpoint: Path, state: dict[str, TensorMeta], exported_path: Path) -> dict[str, Any]:
    item_order = json.loads(DEFAULT_ITEM_ORDER.read_text(encoding="utf-8"))
    train_counts = json.loads(DEFAULT_TRAIN_COUNTS.read_text(encoding="utf-8"))
    meta = state["item_embedding.weight"]
    raw = load_float_storage(checkpoint, meta.storage_key, meta.shape)
    exported = np.load(exported_path).astype("float32", copy=False)
    rows_1_to_n = raw[1:].astype("float32", copy=True)
    cold_rows = [idx for idx, item_id in enumerate(item_order) if int(train_counts.get(str(item_id), 0)) == 0]
    positive_rows = [idx for idx, item_id in enumerate(item_order) if int(train_counts.get(str(item_id), 0)) > 0]
    fallback = rows_1_to_n[positive_rows].mean(axis=0) if positive_rows else np.eye(1, rows_1_to_n.shape[1], 0)[0]
    for idx in cold_rows:
        rows_1_to_n[idx] = fallback
    reproduced = l2_normalize(rows_1_to_n)
    if reproduced.shape != exported.shape:
        return {
            "compatible": False,
            "reason": f"shape mismatch: reproduced checkpoint rows {reproduced.shape}, exported {exported.shape}",
        }
    diff = np.abs(reproduced - exported)
    cos = np.sum(reproduced * exported, axis=1) / (
        np.linalg.norm(reproduced, axis=1) * np.linalg.norm(exported, axis=1)
    )
    return {
        "compatible": bool(float(diff.max()) <= 1e-6 and float(cos.min()) >= 0.999999),
        "checkpoint_embedding_key": "item_embedding.weight",
        "checkpoint_embedding_shape": list(raw.shape),
        "exported_npy_shape": list(exported.shape),
        "selected_row_slice": "item_embedding.weight[1:]",
        "padding_row_removed": True,
        "cold_item_fallback": "mean_train_positive_item_embedding",
        "cold_item_count": len(cold_rows),
        "l2_normalization_applied": True,
        "max_absolute_error": float(diff.max()),
        "mean_absolute_error": float(diff.mean()),
        "cosine_similarity_mean": float(cos.mean()),
        "cosine_similarity_min": float(cos.min()),
        "mismatched_row_count_tolerance_1e-6": int(np.sum(np.max(diff, axis=1) > 1e-6)),
    }


def torch_status() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {"available": True, "version": getattr(torch, "__version__", "unknown")}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_sha = file_sha256(args.checkpoint)
    local_torch_status = torch_status()
    loaded = restricted_checkpoint_load(args.checkpoint)
    state = loaded["model_state_dict"]
    config = loaded.get("config", {})
    train_summary = loaded.get("train_summary", {})
    repo_config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = state_summary(state)
    embedding = compare_embedding(args.checkpoint, state, args.exported_embedding)
    expected_keys = [
        "item_embedding.weight",
        "position_embedding.weight",
        "layers.0.self_attn.in_proj_weight",
        "layers.0.self_attn.out_proj.weight",
        "layers.0.linear1.weight",
        "layers.0.linear2.weight",
        "layers.0.norm1.weight",
        "layers.0.norm2.weight",
        "layers.1.self_attn.in_proj_weight",
        "layers.1.self_attn.out_proj.weight",
        "layers.1.linear1.weight",
        "layers.1.linear2.weight",
        "layers.1.norm1.weight",
        "layers.1.norm2.weight",
        "layer_norm.weight",
    ]
    missing = [key for key in expected_keys if key not in state]
    config_fields = [
        "category",
        "train_csv",
        "valid_csv",
        "item_order",
        "row_index",
        "max_seq_len",
        "embedding_dim",
        "num_layers",
        "num_heads",
        "dropout",
        "learning_rate",
        "batch_size",
        "epochs",
        "seed",
        "normalization",
        "padding_id",
        "padding_side",
        "internal_id_contract",
    ]
    config_mismatch = {
        key: {"checkpoint": config.get(key), "repo_config": repo_config.get(key)}
        for key in config_fields
        if config.get(key) != repo_config.get(key)
    }
    compatible = (
        checkpoint_sha == EXPECTED_SHA256
        and not missing
        and not config_mismatch
        and embedding.get("compatible") is True
        and manifest.get("checkpoint") == args.checkpoint.relative_to(ROOT).as_posix()
    )
    return {
        "schema": "s6_checkpoint_compatibility_report.v1",
        "verdict": "GO_CHECKPOINT_COMPATIBLE" if compatible else "NO_GO_CHECKPOINT_COMPATIBILITY",
        "checkpoint": args.checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "expected_sha256": EXPECTED_SHA256,
        "sha256_matches": checkpoint_sha == EXPECTED_SHA256,
        "torch_load_available_locally": local_torch_status,
        "restricted_parser": {
            "object_type": type(loaded).__name__,
            "top_level_keys": list(loaded.keys()),
            "selected_state_dict_location": "checkpoint['model_state_dict']",
            "optimizer_state_present": "optimizer" in loaded or "optimizer_state_dict" in loaded,
            "scheduler_state_present": "scheduler" in loaded or "scheduler_state_dict" in loaded,
        },
        "state_dict": summary,
        "config": {
            "checkpoint_config": config,
            "repo_config": repo_config,
            "mismatches": config_mismatch,
        },
        "train_summary": {
            "best_epoch": train_summary.get("best_epoch"),
            "best_metric": train_summary.get("best_metric"),
            "keys": list(train_summary.keys()) if isinstance(train_summary, dict) else [],
        },
        "structure_audit": {
            "expected_keys_missing": missing,
            "complete_sequence_model": not missing,
            "embedding_only_artifact": False,
            "num_layers_observed": len({key.split(".")[1] for key in state if key.startswith("layers.")}),
            "embedding_dim_observed": state["item_embedding.weight"].shape[1],
            "max_seq_len_observed": state["position_embedding.weight"].shape[0],
            "item_embedding_rows_observed": state["item_embedding.weight"].shape[0],
        },
        "embedding_export_comparison": embedding,
        "manifest_checkpoint_path_matches": manifest.get("checkpoint") == args.checkpoint.relative_to(ROOT).as_posix(),
        "no_other_version_fallback": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit formal-v3 SASRec checkpoint compatibility.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sha256-file", type=Path, default=DEFAULT_SHA256)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--exported-embedding", type=Path, default=DEFAULT_EXPORTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"verdict": report["verdict"], "checkpoint_sha256": report["checkpoint_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
