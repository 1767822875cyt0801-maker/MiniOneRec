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
from typing import Any, Iterable, Iterator

import numpy as np


DEFAULT_TOPK = [1, 10, 20, 50]
NOT_AVAILABLE = "not_available"


@dataclass
class SequenceExample:
    input_ids: list[int]
    target_ids: list[int]


class NonFiniteTensorError(RuntimeError):
    pass


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
    parser.add_argument("--debug-valid-samples", type=int, default=10)
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
    return SequenceExample(input_ids=inputs + [0] * pad, target_ids=targets + [0] * pad)


def valid_input(sequence: list[int], max_seq_len: int) -> list[int]:
    trimmed = sequence[-max_seq_len:]
    return trimmed + [0] * (max_seq_len - len(trimmed))


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
    ranked = [rank for rank in ranks if rank is not None]
    out: dict[str, Any] = {
        "num_samples": len(ranks),
        "num_ranked_samples": len(ranked),
        "num_unranked_samples": len(ranks) - len(ranked),
        "unranked_samples": len(ranks) - len(ranked),
    }
    for k in sorted(set(topk)):
        hits = 0
        ndcg = 0.0
        for rank in ranks:
            if rank is not None and rank < k:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
        out[f"hr@{k}"] = 0.0 if not ranks else hits / len(ranks)
        out[f"ndcg@{k}"] = 0.0 if not ranks else ndcg / len(ranks)
    out["mrr"] = 0.0 if not ranks else sum(0.0 if rank is None else 1.0 / (rank + 1) for rank in ranks) / len(ranks)
    return out


def rank_from_scores(scores: Iterable[float], target_index: int) -> tuple[int | None, int, list[int]]:
    """Return a safe 0-based full-candidate rank and top-10 row indices.

    Ties are handled pessimistically so all-equal or tied scores cannot turn
    every target into rank 0. Non-finite rows are treated as unrankable instead
    of being silently counted as hits.
    """
    values = np.asarray(list(scores), dtype=np.float64)
    candidate_count = int(values.shape[0])
    if target_index < 0 or target_index >= candidate_count:
        return None, candidate_count, []
    if not np.isfinite(values).all():
        return None, candidate_count, []
    target_score = float(values[target_index])
    rank = int((values >= target_score).sum() - 1)
    order = np.lexsort((np.arange(candidate_count), -values))
    return rank, candidate_count, [int(idx) for idx in order[:10]]


def candidate_summary(candidate_counts: list[int], expected_count: int) -> dict[str, Any]:
    if not candidate_counts:
        return {
            "candidate_count_expected": expected_count,
            "candidate_count_min": 0,
            "candidate_count_max": 0,
            "candidate_count_all_full": False,
        }
    return {
        "candidate_count_expected": expected_count,
        "candidate_count_min": min(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "candidate_count_all_full": all(count == expected_count for count in candidate_counts),
    }


def build_valid_diagnostics(
    rows: list[dict[str, str]],
    row_index: dict[str, int],
    item_order: list[str],
    max_seq_len: int,
    ranks: list[int | None] | None = None,
    top10_rows: list[list[int]] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[: max(0, limit)]):
        history_items = [str(item) for item in parse_list(row.get("history_item_id", ""))]
        history_seq = row_sequence(row, row_index, include_target=False)
        target_item = str(row.get("item_id", "")).strip()
        target_internal = to_internal_id(target_item, row_index)
        input_ids = valid_input(history_seq, max_seq_len) if history_seq else [0] * max_seq_len
        history_length = sum(1 for value in input_ids if value != 0)
        target_in_input = target_internal in input_ids if target_internal is not None else False
        top10 = []
        if top10_rows is not None and idx < len(top10_rows):
            top10 = [item_order[row_idx] for row_idx in top10_rows[idx]]
        diagnostics.append(
            {
                "row": idx,
                "history": history_items,
                "target": target_item,
                "input": input_ids,
                "input_last": input_ids[-1] if input_ids else None,
                "input_last_valid": input_ids[history_length - 1] if history_length else None,
                "history_length": history_length,
                "target_in_input": target_in_input,
                "valid_candidate_count": len(row_index),
                "target_rank": None if ranks is None or idx >= len(ranks) else ranks[idx],
                "top10": top10,
            }
        )
    return diagnostics


def print_valid_diagnostics(diagnostics: list[dict[str, Any]], label: str) -> None:
    if diagnostics:
        print(f"{label}:")
        for sample in diagnostics:
            print(json.dumps(sample, ensure_ascii=False, sort_keys=True))


def evaluate_score_rows(
    score_rows: Iterable[Iterable[float] | None],
    valid_rows: list[dict[str, str]],
    row_index: dict[str, int],
    item_order: list[str],
    args: argparse.Namespace,
    debug_limit: int = 0,
) -> dict[str, Any]:
    ranks: list[int | None] = []
    candidate_counts: list[int] = []
    top10_rows: list[list[int]] = []
    finite_score_samples = 0
    nonfinite_score_samples = 0
    for score_row, row in zip(score_rows, valid_rows):
        target = to_internal_id(row.get("item_id", ""), row_index)
        if target is None or score_row is None:
            ranks.append(None)
            candidate_counts.append(len(row_index))
            top10_rows.append([])
            continue
        values = np.asarray(list(score_row), dtype=np.float64)
        if np.isfinite(values).all():
            finite_score_samples += 1
        else:
            nonfinite_score_samples += 1
        rank, candidate_count, top10 = rank_from_scores(values, target - 1)
        ranks.append(rank)
        candidate_counts.append(candidate_count)
        top10_rows.append(top10)
    metrics = compute_metrics(ranks, args.topk)
    metrics.update(candidate_summary(candidate_counts, len(row_index)))
    metrics["finite_score_samples"] = finite_score_samples
    metrics["nonfinite_score_samples"] = nonfinite_score_samples
    if debug_limit:
        diagnostics = build_valid_diagnostics(
            valid_rows,
            row_index,
            item_order,
            args.max_seq_len,
            ranks=ranks,
            top10_rows=top10_rows,
            limit=debug_limit,
        )
        metrics["debug_valid_samples"] = diagnostics
        print_valid_diagnostics(diagnostics, "VALID_DEBUG_SAMPLES")
    return metrics


def assert_valid_evaluation_complete(metrics: dict[str, Any], stage: str) -> None:
    nonfinite = int(metrics.get("nonfinite_score_samples", 0))
    unranked = int(metrics.get("unranked_samples", metrics.get("num_unranked_samples", 0)))
    candidate_count_all_full = bool(metrics.get("candidate_count_all_full", False))
    if nonfinite or unranked or not candidate_count_all_full:
        details = {
            "stage": stage,
            "nonfinite_score_samples": nonfinite,
            "unranked_samples": unranked,
            "candidate_count_min": metrics.get("candidate_count_min"),
            "candidate_count_max": metrics.get("candidate_count_max"),
            "candidate_count_expected": metrics.get("candidate_count_expected"),
            "candidate_count_all_full": candidate_count_all_full,
        }
        raise SystemExit("Invalid SASRec valid evaluation; refusing to export artifacts: " + json.dumps(details, sort_keys=True))


def random_baseline_metrics(
    valid_rows: list[dict[str, str]],
    row_index: dict[str, int],
    item_order: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    score_rows = (rng.standard_normal(len(row_index)) for _ in valid_rows)
    return evaluate_score_rows(score_rows, valid_rows, row_index, item_order, args)


def popularity_baseline_metrics(
    valid_rows: list[dict[str, str]],
    row_index: dict[str, int],
    item_order: list[str],
    train_counts: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    scores = np.asarray([float(train_counts.get(item_id, 0)) for item_id in item_order], dtype=np.float64)
    return evaluate_score_rows((scores for _ in valid_rows), valid_rows, row_index, item_order, args)


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


def finite_context(input_ids: Any, epoch: int | None = None, batch_index: int | None = None) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if epoch is not None:
        context["epoch"] = epoch
    if batch_index is not None:
        context["batch"] = batch_index
    try:
        lengths = input_ids.ne(0).sum(dim=1).detach().cpu().tolist()
        padding = input_ids.eq(0).sum(dim=1).detach().cpu().tolist()
        context["history_lengths"] = [int(value) for value in lengths]
        context["padding_counts"] = [int(value) for value in padding]
    except Exception:
        pass
    return context


def assert_torch_finite(tensor: Any, stage: str, context: dict[str, Any] | None = None) -> None:
    import torch  # type: ignore

    if torch.isfinite(tensor).all():
        return
    finite = torch.isfinite(tensor)
    bad = (~finite).nonzero(as_tuple=False)
    details: dict[str, Any] = {
        "stage": stage,
        "shape": list(tensor.shape),
        "nonfinite_count": int((~finite).sum().detach().cpu().item()),
    }
    if context:
        details.update(context)
    if bad.numel():
        details["first_nonfinite_index"] = [int(value) for value in bad[0].detach().cpu().tolist()]
    raise NonFiniteTensorError("Non-finite SASRec tensor: " + json.dumps(details, sort_keys=True))


def assert_model_parameters_finite(model: Any, stage: str, context: dict[str, Any] | None = None) -> None:
    for name, parameter in model.named_parameters():
        assert_torch_finite(parameter, f"{stage}.parameter.{name}", context)


def assert_model_gradients_finite(model: Any, stage: str, context: dict[str, Any] | None = None) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert_torch_finite(parameter.grad, f"{stage}.gradient.{name}", context)


def build_torch_model(num_items: int, args: argparse.Namespace):
    import torch  # type: ignore
    from torch import nn

    class MiniSASRec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.item_embedding = nn.Embedding(num_items + 1, args.embedding_dim, padding_idx=0)
            self.position_embedding = nn.Embedding(args.max_seq_len, args.embedding_dim)
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=args.embedding_dim,
                        nhead=args.num_heads,
                        dim_feedforward=args.embedding_dim * 4,
                        dropout=args.dropout,
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(args.num_layers)
                ]
            )
            self.layer_norm = nn.LayerNorm(args.embedding_dim)

        def encode(self, input_ids, check_finite=False, finite_context=None):
            batch, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
            item_hidden = self.item_embedding(input_ids)
            if check_finite:
                assert_torch_finite(item_hidden, "input_embedding", finite_context)
            position_hidden = self.position_embedding(positions)
            if check_finite:
                assert_torch_finite(position_hidden, "positional_embedding", finite_context)
            hidden = item_hidden + position_hidden
            padding_mask = input_ids.eq(0)
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1)
            for layer_index, layer in enumerate(self.layers):
                hidden = layer(hidden, src_mask=causal_mask, src_key_padding_mask=padding_mask)
                hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
                if check_finite:
                    assert_torch_finite(hidden, f"transformer_layer_{layer_index}_output", finite_context)
            hidden = self.layer_norm(hidden)
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            if check_finite:
                assert_torch_finite(hidden, "encoder_output", finite_context)
            return hidden

        def forward(self, input_ids, check_finite=False, finite_context=None):
            hidden = self.encode(input_ids, check_finite=check_finite, finite_context=finite_context)
            logits = hidden @ self.item_embedding.weight[1:].t()
            if check_finite:
                assert_torch_finite(logits, "logits", finite_context)
            return logits

        def last_valid_logits(self, input_ids, check_finite=False, finite_context=None):
            hidden = self.encode(input_ids, check_finite=check_finite, finite_context=finite_context)
            lengths = input_ids.ne(0).sum(dim=1)
            if torch.any(lengths <= 0):
                raise ValueError("last_valid_logits requires at least one non-padding input per row")
            gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
            last_hidden = hidden.gather(1, gather_index).squeeze(1)
            if check_finite:
                assert_torch_finite(last_hidden, "last_valid_hidden", finite_context)
            logits = last_hidden @ self.item_embedding.weight[1:].t()
            if check_finite:
                assert_torch_finite(logits, "logits", finite_context)
            return logits

    return MiniSASRec()


def batch_iter(examples: list[SequenceExample], batch_size: int, shuffle: bool, seed: int) -> Iterator[list[SequenceExample]]:
    order = list(range(len(examples)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[idx] for idx in order[start : start + batch_size]]


def train_model(
    examples: list[SequenceExample],
    valid_rows: list[dict[str, str]],
    row_index: dict[str, int],
    item_order: list[str],
    args: argparse.Namespace,
):
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    model = build_torch_model(len(row_index), args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_state = None
    best_metric = -1.0
    best_epoch = 0
    history_rows: list[dict[str, Any]] = []
    assert_model_parameters_finite(model, "initial")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for batch_index, batch in enumerate(batch_iter(examples, args.batch_size, shuffle=True, seed=args.seed + epoch), start=1):
            inputs = torch.tensor([ex.input_ids for ex in batch], dtype=torch.long, device=args.device)
            targets = torch.tensor([ex.target_ids for ex in batch], dtype=torch.long, device=args.device)
            context = finite_context(inputs, epoch=epoch, batch_index=batch_index)
            logits = model(inputs, check_finite=True, finite_context=context)
            mask = targets.gt(0)
            if not mask.any():
                continue
            loss = F.cross_entropy(logits[mask], targets[mask] - 1)
            assert_torch_finite(loss, "loss", context)
            optimizer.zero_grad()
            loss.backward()
            assert_model_gradients_finite(model, "backward", context)
            optimizer.step()
            assert_model_parameters_finite(model, "optimizer_step", context)
            total_loss += float(loss.detach().cpu()) * int(mask.sum().item())
            total_tokens += int(mask.sum().item())
        metrics = evaluate_model(model, valid_rows, row_index, item_order, args)
        assert_valid_evaluation_complete(metrics, f"epoch_{epoch}_valid")
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


def evaluate_model(
    model: Any,
    valid_rows: list[dict[str, str]],
    row_index: dict[str, int],
    item_order: list[str],
    args: argparse.Namespace,
    debug_limit: int = 0,
) -> dict[str, Any]:
    import torch  # type: ignore

    model.eval()
    score_rows: list[np.ndarray | None] = []
    with torch.no_grad():
        for row in valid_rows:
            history_seq = row_sequence(row, row_index, include_target=False)
            target = to_internal_id(row.get("item_id", ""), row_index)
            if not history_seq or target is None:
                score_rows.append(None)
                continue
            inputs = torch.tensor([valid_input(history_seq, args.max_seq_len)], dtype=torch.long, device=args.device)
            logits = model.last_valid_logits(
                inputs,
                check_finite=True,
                finite_context=finite_context(inputs),
            )[0]
            score_rows.append(logits.detach().cpu().numpy())
    return evaluate_score_rows(score_rows, valid_rows, row_index, item_order, args, debug_limit=debug_limit)


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
    baselines = {
        "random_untrained": random_baseline_metrics(valid_rows, row_index, item_order, args),
        "popularity": popularity_baseline_metrics(valid_rows, row_index, item_order, counts, args),
    }
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
        "padding_side": "right",
        "internal_id_contract": "internal_id = canonical_row_index[item_id] + 1",
        "evaluation_contract": {
            "split": "valid",
            "candidate_set": "all canonical item_order rows",
            "candidate_count": len(item_order),
            "valid_input_includes_target": False,
            "valid_hidden_state": "gather length-1 last non-padding history position",
            "rank_tie_policy": "pessimistic: rank = count(score >= target_score) - 1",
            "non_finite_scores": "unranked, never counted as hit",
        },
    }
    available, torch_status = torch_available()
    print(
        "SASRec builder preflight: "
        f"category={args.category} train_examples={len(examples)} valid_rows={len(valid_rows)} "
        f"items={len(item_order)} torch={torch_status}"
    )
    if args.dry_run:
        print(f"DRY_RUN: output_dir={args.output_dir}")
        print(f"DRY_RUN: valid_candidate_count={len(item_order)}")
        print_valid_diagnostics(
            build_valid_diagnostics(valid_rows, row_index, item_order, args.max_seq_len, limit=args.debug_valid_samples),
            "DRY_RUN_VALID_INPUT_DEBUG",
        )
        print("DRY_RUN_BASELINES:")
        print(json.dumps(baselines, indent=2, sort_keys=True))
        print("DRY_RUN complete. No files were written.")
        return
    if not available:
        raise SystemExit(f"PyTorch is required for SASRec training/export but is unavailable: {torch_status}")
    if not examples:
        raise SystemExit("No train sequence examples were built from train CSV.")

    model, train_summary = train_model(examples, valid_rows, row_index, item_order, args)
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"PyTorch disappeared after preflight: {exc}")

    final_valid = evaluate_model(
        model,
        valid_rows,
        row_index,
        item_order,
        args,
        debug_limit=args.debug_valid_samples,
    )
    assert_valid_evaluation_complete(final_valid, "final_valid")

    checkpoint_dir = args.output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config, "train_summary": train_summary}, checkpoint_path)
    raw_matrix = model.item_embedding.weight.detach().cpu().numpy()[1:]
    export_matrix, cold_report = prepare_export_matrix(raw_matrix, item_order, counts, args.normalization)
    metrics = {
        "checkpoint_metric": args.checkpoint_metric,
        "baselines": baselines,
        "train_summary": train_summary,
        "final_valid": final_valid,
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
