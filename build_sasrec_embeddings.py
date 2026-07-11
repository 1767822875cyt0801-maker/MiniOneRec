#!/usr/bin/env python3
"""Train/export a train-only SASRec item embedding artifact.

The implementation keeps all data contracts explicit:

- train CSV is used for fitting and train-only counts;
- valid CSV is used only for checkpoint selection/evaluation;
- test CSV is intentionally not accepted by the CLI;
- exported rows follow the canonical item_order/row_index exactly.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_TOPK = [10, 20, 50]
NOT_AVAILABLE = "not_available"


@dataclass
class SequenceExample:
    input_ids: list[int]
    target_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export train-only SASRec item embeddings.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--valid-csv", type=Path, required=True)
    parser.add_argument("--item-order", type=Path, required=True)
    parser.add_argument("--row-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-metric", choices=["ndcg@20", "hr@20"], default="ndcg@20")
    parser.add_argument("--topk", type=int, nargs="+", default=DEFAULT_TOPK)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-valid-rows", type=int, default=0)
    parser.add_argument("--normalization", choices=["l2", "none"], default="l2")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def read_csv_rows(path: Path, limit: int = 0) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = []
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if limit and idx >= limit:
                break
            rows.append(dict(row))
        return rows


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_test_path(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    return any(part == "test" or part == "test.csv" or part.startswith("test_") or part.endswith("_test.csv") for part in lowered_parts)


def load_canonical_mapping(item_order_path: Path, row_index_path: Path) -> tuple[list[str], dict[str, int]]:
    item_order = [str(item) for item in read_json(item_order_path)]
    row_index = {str(item): int(idx) for item, idx in read_json(row_index_path).items()}
    if len(item_order) != len(row_index):
        raise ValueError("item_order length does not match row_index length")
    for idx, item_id in enumerate(item_order):
        if row_index.get(item_id) != idx:
            raise ValueError(f"canonical row mismatch for item_id={item_id}: row_index={row_index.get(item_id)} idx={idx}")
    return item_order, row_index


def to_internal_id(item_id: Any, row_index: dict[str, int]) -> int | None:
    row = row_index.get(str(item_id))
    if row is None:
        return None
    return row + 1


def row_sequence(row: dict[str, str], row_index: dict[str, int], include_target: bool) -> list[int]:
    seq: list[int] = []
    for item in parse_list(row.get("history_item_id", "")):
        internal = to_internal_id(item, row_index)
        if internal is not None:
            seq.append(internal)
    if include_target:
        internal = to_internal_id(row.get("item_id", ""), row_index)
        if internal is not None:
            seq.append(internal)
    return seq


def shifted_example(sequence: list[int], max_seq_len: int) -> SequenceExample | None:
    if len(sequence) < 2:
        return None
    trimmed = sequence[-(max_seq_len + 1) :]
    inputs = trimmed[:-1]
    targets = trimmed[1:]
    pad = max_seq_len - len(inputs)
    if pad < 0:
        raise ValueError("shifted sequence exceeds max_seq_len")
    return SequenceExample(input_ids=[0] * pad + inputs, target_ids=[0] * pad + targets)


def valid_input(sequence: list[int], max_seq_len: int) -> list[int]:
    trimmed = sequence[-max_seq_len:]
    return [0] * (max_seq_len - len(trimmed)) + trimmed


def causal_attention_mask(max_seq_len: int) -> np.ndarray:
    return np.triu(np.ones((max_seq_len, max_seq_len), dtype=bool), k=1)


def build_examples(rows: list[dict[str, str]], row_index: dict[str, int], max_seq_len: int) -> list[SequenceExample]:
    examples: list[SequenceExample] = []
    for row in rows:
        example = shifted_example(row_sequence(row, row_index, include_target=True), max_seq_len)
        if example is not None:
            examples.append(example)
    return examples


def train_item_counts(rows: list[dict[str, str]], row_index: dict[str, int], item_order: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    valid_items = set(row_index)
    for row in rows:
        target = str(row.get("item_id", "")).strip()
        if target in valid_items:
            counts[target] += 1
        for item in parse_list(row.get("history_item_id", "")):
            item_id = str(item)
            if item_id in valid_items:
                counts[item_id] += 1
    return {item_id: int(counts.get(item_id, 0)) for item_id in item_order}


def prepare_export_matrix(
    raw_item_embeddings: np.ndarray,
    item_order: list[str],
    train_counts: dict[str, int],
    normalization: str = "l2",
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(raw_item_embeddings, dtype=np.float32).copy()
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D item embeddings, got shape={matrix.shape}")
    if matrix.shape[0] != len(item_order):
        raise ValueError(f"Embedding rows={matrix.shape[0]} do not match item_order length={len(item_order)}")
    if not np.isfinite(matrix).all():
        raise ValueError("Raw SASRec item embeddings contain NaN/Inf")

    train_positive_rows = [idx for idx, item_id in enumerate(item_order) if int(train_counts.get(item_id, 0)) > 0]
    cold_rows = [idx for idx, item_id in enumerate(item_order) if int(train_counts.get(item_id, 0)) == 0]
    if train_positive_rows:
        fallback = matrix[train_positive_rows].mean(axis=0)
    else:
        fallback = np.zeros((matrix.shape[1],), dtype=np.float32)
        fallback[0] = 1.0
    if not np.isfinite(fallback).all() or float(np.linalg.norm(fallback)) <= 1e-12:
        fallback = np.zeros((matrix.shape[1],), dtype=np.float32)
        fallback[0] = 1.0
    for row in cold_rows:
        matrix[row] = fallback

    if normalization == "l2":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        matrix = matrix / norms
    elif normalization != "none":
        raise ValueError(f"Unsupported normalization: {normalization}")
    if not np.isfinite(matrix).all():
        raise ValueError("Exported SASRec item embeddings contain NaN/Inf")

    report = {
        "num_items": len(item_order),
        "embedding_dim": int(matrix.shape[1]),
        "train_positive_items": len(train_positive_rows),
        "cold_items": len(cold_rows),
        "cold_item_ids": [item_order[idx] for idx in cold_rows],
        "fallback": "mean_train_positive_item_embedding" if train_positive_rows else "unit_vector_no_train_positive_items",
        "normalization": normalization,
    }
    return matrix.astype(np.float32, copy=False), report


def compute_metrics(ranks: list[int | None], topk: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"num_samples": len(ranks)}
    for k in sorted(set(topk)):
        hits = 0
        ndcg = 0.0
        for rank in ranks:
            if rank is not None and rank < k:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
        out[f"hr@{k}"] = 0.0 if not ranks else hits / len(ranks)
        out[f"ndcg@{k}"] = 0.0 if not ranks else ndcg / len(ranks)
    return out


def torch_available() -> tuple[bool, str]:
    try:
        import torch  # type: ignore

        return True, torch.__version__
    except Exception as exc:
        return False, repr(exc)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_torch_model(num_items: int, args: argparse.Namespace):
    import torch  # type: ignore
    from torch import nn

    class MiniSASRec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.item_embedding = nn.Embedding(num_items + 1, args.embedding_dim, padding_idx=0)
            self.position_embedding = nn.Embedding(args.max_seq_len, args.embedding_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=args.embedding_dim,
                nhead=args.num_heads,
                dim_feedforward=args.embedding_dim * 4,
                dropout=args.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=args.num_layers)
            self.layer_norm = nn.LayerNorm(args.embedding_dim)

        def forward(self, input_ids):
            batch, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
            hidden = self.item_embedding(input_ids) + self.position_embedding(positions)
            padding_mask = input_ids.eq(0)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1)
            hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
            hidden = self.layer_norm(hidden)
            logits = hidden @ self.item_embedding.weight[1:].t()
            return logits

    return MiniSASRec()


def batch_iter(examples: list[SequenceExample], batch_size: int, shuffle: bool, seed: int) -> Iterator[list[SequenceExample]]:
    order = list(range(len(examples)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[idx] for idx in order[start : start + batch_size]]


def train_model(examples: list[SequenceExample], valid_rows: list[dict[str, str]], row_index: dict[str, int], args: argparse.Namespace):
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    model = build_torch_model(len(row_index), args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_state = None
    best_metric = -1.0
    best_epoch = 0
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for batch in batch_iter(examples, args.batch_size, shuffle=True, seed=args.seed + epoch):
            inputs = torch.tensor([ex.input_ids for ex in batch], dtype=torch.long, device=args.device)
            targets = torch.tensor([ex.target_ids for ex in batch], dtype=torch.long, device=args.device)
            logits = model(inputs)
            mask = targets.gt(0)
            if not mask.any():
                continue
            loss = F.cross_entropy(logits[mask], targets[mask] - 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(mask.sum().item())
            total_tokens += int(mask.sum().item())
        metrics = evaluate_model(model, valid_rows, row_index, args)
        metric_value = float(metrics.get(args.checkpoint_metric, 0.0))
        if metric_value > best_metric:
            best_metric = metric_value
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_tokens if total_tokens else 0.0,
                **metrics,
            }
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_metric": best_metric, "history": history_rows}


def evaluate_model(model: Any, valid_rows: list[dict[str, str]], row_index: dict[str, int], args: argparse.Namespace) -> dict[str, Any]:
    import torch  # type: ignore

    model.eval()
    ranks: list[int | None] = []
    with torch.no_grad():
        for row in valid_rows:
            history_seq = row_sequence(row, row_index, include_target=False)
            target = to_internal_id(row.get("item_id", ""), row_index)
            if not history_seq or target is None:
                ranks.append(None)
                continue
            inputs = torch.tensor([valid_input(history_seq, args.max_seq_len)], dtype=torch.long, device=args.device)
            logits = model(inputs)[0, -1]
            target_score = logits[target - 1]
            rank = int((logits > target_score).sum().item())
            ranks.append(rank)
    return compute_metrics(ranks, args.topk)


def git_info() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception as exc:
        sha = f"not_available: {exc}"
    try:
        status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    except Exception as exc:
        status = f"not_available: {exc}"
    return {"commit": sha, "status_short": status}


def write_artifacts(
    output_dir: Path,
    category: str,
    matrix: np.ndarray,
    item_order: list[str],
    row_index: dict[str, int],
    train_counts: dict[str, int],
    cold_report: dict[str, Any],
    config: dict[str, Any],
    metrics: dict[str, Any],
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / f"{category}.sasrec_item_emb.npy"
    np.save(emb_path, matrix)
    paths = {
        "embedding": emb_path,
        "row_index": output_dir / f"{category}.row_index.json",
        "item_order": output_dir / f"{category}.item_order.json",
        "train_item_count": output_dir / f"{category}.train_item_count.json",
        "cold_item_report": output_dir / f"{category}.cold_item_report.json",
        "config": output_dir / "config.json",
        "metrics_summary": output_dir / "metrics_summary.json",
        "artifact_manifest": output_dir / "artifact_manifest.json",
    }
    write_json(paths["row_index"], row_index)
    write_json(paths["item_order"], item_order)
    write_json(paths["train_item_count"], train_counts)
    write_json(paths["cold_item_report"], cold_report)
    write_json(paths["config"], config)
    write_json(paths["metrics_summary"], metrics)
    manifest = {
        "category": category,
        "embedding_source": "train_only_sasrec",
        "embedding_path": paths["embedding"].as_posix(),
        "row_index": paths["row_index"].as_posix(),
        "item_order": paths["item_order"].as_posix(),
        "dimension": int(matrix.shape[1]),
        "num_items": int(matrix.shape[0]),
        "normalization": config["normalization"],
        "finite": bool(np.isfinite(matrix).all()),
        "checkpoint": checkpoint_path.as_posix() if checkpoint_path else NOT_AVAILABLE,
        "hashes": {
            key: file_sha256(path)
            for key, path in paths.items()
            if path.exists() and key != "artifact_manifest"
        },
        "git": git_info(),
        "train_only_policy": {
            "train_used_for_fitting": True,
            "valid_used_for_checkpoint_selection": True,
            "test_read": False,
            "cold_fallback_uses_valid_or_test": False,
        },
    }
    write_json(paths["artifact_manifest"], manifest)
    return {key: path.as_posix() for key, path in paths.items()}


def main() -> None:
    args = parse_args()
    if has_test_path(args.train_csv) or has_test_path(args.valid_csv):
        raise SystemExit("SASRec builder refuses test paths in train/valid arguments.")
    set_seed(args.seed)
    item_order, row_index = load_canonical_mapping(args.item_order, args.row_index)
    train_rows = read_csv_rows(args.train_csv, args.max_train_rows)
    valid_rows = read_csv_rows(args.valid_csv, args.max_valid_rows)
    examples = build_examples(train_rows, row_index, args.max_seq_len)
    counts = train_item_counts(train_rows, row_index, item_order)
    config = {
        "category": args.category,
        "train_csv": args.train_csv.as_posix(),
        "valid_csv": args.valid_csv.as_posix(),
        "item_order": args.item_order.as_posix(),
        "row_index": args.row_index.as_posix(),
        "max_seq_len": args.max_seq_len,
        "embedding_dim": args.embedding_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "checkpoint_metric": args.checkpoint_metric,
        "normalization": args.normalization,
        "max_train_rows": args.max_train_rows,
        "max_valid_rows": args.max_valid_rows,
        "num_train_examples": len(examples),
        "num_valid_rows": len(valid_rows),
        "padding_id": 0,
        "internal_id_contract": "internal_id = canonical_row_index[item_id] + 1",
    }
    available, torch_status = torch_available()
    print(
        "SASRec builder preflight: "
        f"category={args.category} train_examples={len(examples)} valid_rows={len(valid_rows)} "
        f"items={len(item_order)} torch={torch_status}"
    )
    if args.dry_run:
        print(f"DRY_RUN: output_dir={args.output_dir}")
        print("DRY_RUN complete. No files were written.")
        return
    if not available:
        raise SystemExit(f"PyTorch is required for SASRec training/export but is unavailable: {torch_status}")
    if not examples:
        raise SystemExit("No train sequence examples were built from train CSV.")

    model, train_summary = train_model(examples, valid_rows, row_index, args)
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"PyTorch disappeared after preflight: {exc}")

    checkpoint_dir = args.output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config, "train_summary": train_summary}, checkpoint_path)
    raw_matrix = model.item_embedding.weight.detach().cpu().numpy()[1:]
    export_matrix, cold_report = prepare_export_matrix(raw_matrix, item_order, counts, args.normalization)
    metrics = {
        "checkpoint_metric": args.checkpoint_metric,
        "train_summary": train_summary,
        "final_valid": evaluate_model(model, valid_rows, row_index, args),
    }
    paths = write_artifacts(
        args.output_dir,
        args.category,
        export_matrix,
        item_order,
        row_index,
        counts,
        cold_report,
        config,
        metrics,
        checkpoint_path,
    )
    print(f"Wrote SASRec embedding artifacts under: {args.output_dir}")
    print(json.dumps(paths, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
