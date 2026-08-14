"""Thin adapters around the repository's frozen candidate/fusion/ranker code."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import evaluate_candidates as candidate_impl
from scripts import s5_auxiliary_fusion_conversion as s5_impl

from .config import ResolvedConfig
from .profiling import cpu_rss_bytes, cuda_snapshot
from .schemas import InputBundle, PostBudgetSample, PreBudgetSample, RankingSample


class AdapterError(ValueError):
    pass


@dataclass(frozen=True)
class RankerResources:
    model: dict[str, Any]
    matrix: Any
    row_index: dict[str, int]
    popularity: Any
    max_log_popularity: float


def _read_split_assignments(
    path: Path,
    selected_split: str,
    max_selected: int | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Read the existing P2 assignment without deriving or guessing a split."""

    split_map: dict[str, str] = {}
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for fallback, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = s5_impl.normalize_id(row.get("row_index", row.get("sample_id")), fallback)
            split = str(row.get("p2_split", ""))
            if split not in {"valid_fit", "valid_select"}:
                raise AdapterError(f"Unsupported or missing P2 split for sample {row_id}: {split!r}")
            if row_id in split_map and split_map[row_id] != split:
                raise AdapterError(f"Conflicting P2 split assignments for sample {row_id}")
            split_map[row_id] = split
            counts[split] = counts.get(split, 0) + 1
            if max_selected is not None:
                selected_count = sum(value == selected_split for value in split_map.values())
                if selected_count >= max_selected:
                    break
    selected_ids = {sample_id: split for sample_id, split in split_map.items() if split == selected_split}
    if not selected_ids:
        raise AdapterError(f"No {selected_split} assignments found in {path}")
    source_kind = (
        "existing_p2_history_ranker_assignment"
        if path.name == "history_reranked_candidates.jsonl"
        else "provided_split_manifest"
    )
    return selected_ids, {
        "source": str(path),
        "source_kind": source_kind,
        "selected_split": selected_split,
        "observed_counts": counts,
        "partial_read": max_selected is not None,
    }


def inspect_input_files(config: ResolvedConfig) -> dict[str, Any]:
    """Perform bounded dry-run checks without loading a model or full pipeline."""

    report: dict[str, Any] = {"files": {}, "errors": []}
    for section, fields in {
        "inputs": [
            "cf_prediction",
            "sasrec_prediction",
            "cf_sid_mapping",
            "sasrec_sid_mapping",
            "valid_data",
            "valid_split_manifest",
        ],
        "rerank": ["frozen_config", "frozen_model", "train_data", "item_embedding", "row_index"],
    }.items():
        for field in fields:
            path = Path(config.data[section][field])
            entry = {"path": str(path), "exists": path.is_file()}
            if path.is_file():
                stat = path.stat()
                entry.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            else:
                report["errors"].append(f"missing {section}.{field}: {path}")
            report["files"][f"{section}.{field}"] = entry
    if report["errors"]:
        raise AdapterError("; ".join(report["errors"]))

    for field in ["cf_prediction", "sasrec_prediction"]:
        path = Path(config.data["inputs"][field])
        with path.open("r", encoding="utf-8") as handle:
            prefix = handle.read(4096).lstrip()
        if not prefix.startswith("[") or '"predict"' not in prefix:
            raise AdapterError(f"{field} is not a legacy frozen prediction JSON list: {path}")
    for field in ["cf_sid_mapping", "sasrec_sid_mapping"]:
        mapping = candidate_impl.normalize_sid2items(Path(config.data["inputs"][field]))
        if not mapping:
            raise AdapterError(f"{field} contains no SID mappings")
        report["files"][f"inputs.{field}"]["sid_count"] = len(mapping)
    with Path(config.data["rerank"]["frozen_model"]).open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("model_type") != "history_aware_linear_pairwise_logistic":
        raise AdapterError("frozen model schema is not P2 history-aware linear pairwise logistic")
    report["ranker_model_type"] = model["model_type"]
    report["status"] = "PASS"
    return report


def load_input_bundle(config: ResolvedConfig, max_samples: int | None = None) -> tuple[InputBundle, dict[str, Any]]:
    inputs = config.data["inputs"]
    selected_split = config.data["evaluation"]["quality_split"]
    cf_predictions = candidate_impl.load_predictions(Path(inputs["cf_prediction"]))
    sasrec_predictions = candidate_impl.load_predictions(Path(inputs["sasrec_prediction"]))
    eval_rows = candidate_impl.read_csv_rows(Path(inputs["valid_data"]))
    if len(cf_predictions) != len(sasrec_predictions) or len(cf_predictions) != len(eval_rows):
        raise AdapterError(
            "prediction/evaluation row counts do not align: "
            f"cf={len(cf_predictions)} sasrec={len(sasrec_predictions)} eval={len(eval_rows)}"
        )
    split_map, split_evidence = _read_split_assignments(
        Path(inputs["valid_split_manifest"]), selected_split, max_samples
    )
    selected_indices: list[int] = []
    for sample_id in sorted(split_map, key=s5_impl.item_sort_key):
        try:
            idx = int(sample_id)
        except ValueError as exc:
            raise AdapterError(f"P2 split sample_id is not a row index: {sample_id}") from exc
        if idx < 0 or idx >= len(eval_rows):
            raise AdapterError(f"P2 split row index is out of range: {idx}")
        selected_indices.append(idx)
    if max_samples is not None:
        selected_indices = selected_indices[:max_samples]
    selected_split_map = {str(idx): selected_split for idx in selected_indices}
    bundle = InputBundle(
        prediction_rows_cf=cf_predictions,
        prediction_rows_sasrec=sasrec_predictions,
        eval_rows=eval_rows,
        selected_indices=selected_indices,
        split_map=selected_split_map,
        cf_sid2items=candidate_impl.normalize_sid2items(Path(inputs["cf_sid_mapping"])),
        sasrec_sid2items=candidate_impl.normalize_sid2items(Path(inputs["sasrec_sid_mapping"])),
    )
    return bundle, {
        "prediction_rows": len(cf_predictions),
        "selected_rows": len(selected_indices),
        "split_evidence": split_evidence,
    }


def _expand_stream(
    predictions: list[dict[str, Any]],
    eval_rows: list[dict[str, str]],
    selected_indices: Iterable[int],
    sid2items: dict[str, list[str]],
    max_pred_sids: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid_sid_set = set(sid2items)
    for idx in selected_indices:
        pred = predictions[idx]
        eval_row = eval_rows[idx]
        _, item_ids, details, pred_valid = candidate_impl.expand_candidates(
            pred["pred_sids"],
            valid_sid_set,
            sid2items,
            {},
            [],
            max_pred_sids,
            max_candidates,
            False,
        )
        rows.append(
            {
                "row_index": idx,
                "sample_id": str(idx),
                "target_item_id": str(eval_row.get("item_id", "")).strip(),
                "history_item_id": [str(value) for value in candidate_impl.parse_list(eval_row.get("history_item_id"))],
                "pred_sids": pred["pred_sids"][:max_pred_sids],
                "pred_sid_valid": pred_valid,
                "candidate_item_ids": item_ids,
                "candidate_details": details,
            }
        )
    return rows


def exact_candidate_expansion(bundle: InputBundle, config: ResolvedConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate = config.data["candidate"]
    max_pred_sids = int(candidate.get("max_pred_sids", 50))
    max_candidates = int(candidate.get("max_candidates_per_stream", 1000))
    cf_rows = _expand_stream(
        bundle.prediction_rows_cf,
        bundle.eval_rows,
        bundle.selected_indices,
        bundle.cf_sid2items,
        max_pred_sids,
        max_candidates,
    )
    sasrec_rows = _expand_stream(
        bundle.prediction_rows_sasrec,
        bundle.eval_rows,
        bundle.selected_indices,
        bundle.sasrec_sid2items,
        max_pred_sids,
        max_candidates,
    )
    return cf_rows, sasrec_rows


def source_aware_fusion(
    cf_rows: list[dict[str, Any]],
    sasrec_rows: list[dict[str, Any]],
    split_map: dict[str, str],
    config: ResolvedConfig,
) -> list[PreBudgetSample]:
    """Adapt schemas, then delegate dedup and RRF ordering to the frozen S5 implementation."""

    samples = s5_impl.build_samples(cf_rows, sasrec_rows, split_map)
    fusion = config.data["fusion"]
    out: list[PreBudgetSample] = []
    for sample in samples:
        ordered = s5_impl.rrf_order(
            sample,
            float(fusion["lambda_sasrec"]),
            float(fusion["source_bonus"]),
        )
        if len(ordered) != len(set(ordered)):
            raise AdapterError(f"fusion output contains duplicate items for sample {sample.sample_id}")
        out.append(
            PreBudgetSample(
                sample_id=sample.sample_id,
                target_item_id=sample.target_item_id,
                cf_candidate_count=len(sample.cf_items),
                sasrec_candidate_count=len(sample.sasrec_items),
                pre_dedup_count=len(sample.cf_items) + len(sample.sasrec_items),
                ordered_item_ids=tuple(ordered),
                source_sample=sample,
            )
        )
    return out


def load_ranker_resources(config: ResolvedConfig) -> RankerResources:
    rerank = config.data["rerank"]
    model = s5_impl.load_model(Path(rerank["frozen_model"]))
    matrix, row_index = s5_impl.load_embeddings(Path(rerank["item_embedding"]), Path(rerank["row_index"]))
    popularity = s5_impl.train_popularity(Path(rerank["train_data"]))
    max_log = max((math.log1p(value) for value in popularity.values()), default=0.0)
    return RankerResources(model, matrix, row_index, popularity, max_log)


def rerank_post_budget(
    samples: list[PostBudgetSample],
    resources: RankerResources,
    top_n: int,
    repeat: int,
    warmup: bool,
) -> tuple[list[RankingSample], list[dict[str, Any]]]:
    rankings: list[RankingSample] = []
    latencies: list[dict[str, Any]] = []
    for sample in samples:
        if sample.stage != "post_fusion_pre_rerank":
            raise AdapterError("P2 reranker only accepts post-budget candidates")
        started = time.perf_counter_ns()
        ranked = s5_impl.frozen_ranker_order(
            sample.source_sample,
            list(sample.ordered_item_ids),
            resources.model,
            resources.matrix,
            resources.row_index,
            resources.popularity,
            resources.max_log_popularity,
            "source_independent_projection",
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        cuda = cuda_snapshot()
        rankings.append(
            RankingSample(
                sample_id=sample.sample_id,
                target_item_id=sample.target_item_id,
                budget=sample.budget,
                candidate_item_ids=sample.ordered_item_ids,
                ranked_item_ids=tuple(ranked),
                top_item_ids=tuple(ranked[:top_n]),
            )
        )
        latencies.append(
            {
                "sample_id": sample.sample_id,
                "budget": str(sample.budget),
                "stage": "history_rerank",
                "elapsed_ms": elapsed_ms,
                "sample_count": 1,
                "input_candidate_count": len(sample.ordered_item_ids),
                "output_candidate_count": min(top_n, len(ranked)),
                "candidate_count_in": len(sample.ordered_item_ids),
                "candidate_count_out": min(top_n, len(ranked)),
                "repeat": repeat,
                "warmup": warmup,
                "cpu_rss_bytes": cpu_rss_bytes(),
                **cuda,
            }
        )
    return rankings, latencies
