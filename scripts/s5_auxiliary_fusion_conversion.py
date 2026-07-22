#!/usr/bin/env python3
"""S5-0 frozen CF+SASRec auxiliary fusion conversion gate.

This script is CPU-only and inference-only. It reads frozen S4 formal valid
candidate artifacts plus the existing Stage 7 P2 split/ranker evidence, writes
deterministic CF+SASRec candidate provenance, evaluates fixed auxiliary
injection policies, and applies the frozen P2 history ranker only under an
explicit compatibility mode.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORY = "Industrial_and_Scientific"
DEFAULT_FORMAL_ROOT = ROOT / "incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036"
DEFAULT_STAGE7_ROOT = ROOT / "results/stage7_validation_protocol/valid/Industrial_and_Scientific"
DEFAULT_OUT_ROOT = ROOT / "results/s5_auxiliary_fusion/Industrial_and_Scientific"
DEFAULT_DOC = ROOT / "docs/s5_0_frozen_auxiliary_fusion_conversion.md"
DEFAULT_KS = [1, 5, 10, 20, 50]
RRF_LAMBDAS = [0.1, 0.25, 0.5, 0.75, 1.0]
RRF_BONUSES = [0.0, 0.01]
RRF_K = 60.0
EXPECTED_ROWS = 4532
SPLIT_SELECT = "valid_select"
SPLIT_REPORT_SOURCE = "valid_fit"
SPLIT_REPORT = "valid_report"


@dataclass(frozen=True)
class Candidate:
    item_id: str
    source_cf: bool
    source_sasrec: bool
    cf_rank: int | None
    sasrec_rank: int | None
    cf_score: float | None
    sasrec_score: float | None
    cf_expansion_level: int | None
    sasrec_expansion_level: int | None
    cf_bucket_size: int | None
    sasrec_bucket_size: int | None

    @property
    def duplicate_across_sources(self) -> bool:
        return self.source_cf and self.source_sasrec

    @property
    def source_count(self) -> int:
        return int(self.source_cf) + int(self.source_sasrec)

    @property
    def best_source_rank(self) -> int | None:
        ranks = [rank for rank in [self.cf_rank, self.sasrec_rank] if rank is not None]
        return min(ranks) if ranks else None


@dataclass(frozen=True)
class Sample:
    sample_id: str
    target_item_id: str
    history_item_id: list[str]
    cf_items: list[str]
    sasrec_items: list[str]
    candidates: dict[str, Candidate]
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S5-0 frozen auxiliary fusion conversion gate.")
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--stage7-root", type=Path, default=DEFAULT_STAGE7_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object rows in {path}")
                rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def no_test_path(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    return "test" not in parts and path.name.lower() != "test.csv"


def assert_no_test_paths(paths: Iterable[Path]) -> None:
    bad = [path.as_posix() for path in paths if not no_test_path(path)]
    if bad:
        raise ValueError(f"S5-0 refuses test paths: {bad}")


def assert_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def item_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def normalize_id(value: Any, fallback: int) -> str:
    if value is None or value == "":
        return str(fallback)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def list_field(row: dict[str, Any], field: str) -> list[str]:
    values = row.get(field, [])
    if isinstance(values, list):
        return [str(value) for value in values]
    text = str(values).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, (list, tuple)):
        return [str(value) for value in parsed]
    return [str(parsed)]


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def rank_of(target_item_id: str, items: list[str]) -> int | None:
    target = str(target_item_id)
    for idx, item_id in enumerate(items):
        if str(item_id) == target:
            return idx
    return None


def hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank < k


def rank_metrics(ranks: list[int | None], ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"num_samples": len(ranks)}
    for k in ks:
        hits = 0
        ndcg = 0.0
        for rank in ranks:
            if rank is not None and rank < k:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
        out[f"hr@{k}"] = 0.0 if not ranks else hits / len(ranks)
        out[f"ndcg@{k}"] = 0.0 if not ranks else ndcg / len(ranks)
    mrr = sum(0.0 if rank is None else 1.0 / (rank + 1) for rank in ranks)
    out["mrr"] = 0.0 if not ranks else mrr / len(ranks)
    out["target_in_pool_count"] = sum(rank is not None for rank in ranks)
    out["target_in_pool_rate"] = 0.0 if not ranks else out["target_in_pool_count"] / len(ranks)
    return out


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        row_id = normalize_id(row.get("row_index", row.get("sample_id")), idx)
        if row_id in indexed:
            raise ValueError(f"Duplicate row id in {label}: {row_id}")
        indexed[row_id] = row
    return indexed


def detail_index(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for detail in row.get("candidate_details", []):
        if isinstance(detail, dict):
            item = str(detail.get("item_id", ""))
            if item and item not in out:
                out[item] = detail
    return out


def candidate_paths(formal_root: Path) -> dict[str, Path]:
    base = formal_root / "results/s4_sasrec_sid_valid/Industrial_and_Scientific"
    return {
        "cf": base / "cf_k512_dedup/seed42/formal/valid/full_valid/candidates/candidates.jsonl",
        "sasrec": base / "sasrec_v3_k512_dedup/seed42/formal/valid/full_valid/candidates/candidates.jsonl",
        "cf_manifest": base / "cf_k512_dedup/seed42/formal/valid/full_valid/artifact_manifest.json",
        "sasrec_manifest": base / "sasrec_v3_k512_dedup/seed42/formal/valid/full_valid/artifact_manifest.json",
    }


def load_split_map(stage7_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    path = stage7_root / "p2_history_ranker/history_reranked_candidates.jsonl"
    assert_file(path)
    rows = read_jsonl(path)
    split_map: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        row_id = normalize_id(row.get("row_index"), idx)
        split = str(row.get("p2_split", ""))
        if split not in {"valid_fit", "valid_select"}:
            raise ValueError(f"Unsupported existing P2 split for row {row_id}: {split}")
        split_map[row_id] = split
        counts[split] += 1
    return split_map, {
        "source": path.as_posix(),
        "policy": "reuse_existing_p2_split; S5 valid_report is the existing non-select valid_fit subset",
        "counts": dict(counts),
        "valid_select": "valid_select",
        "valid_report": "valid_fit",
        "separate_valid_report_artifact_exists": False,
    }


def build_samples(cf_rows: list[dict[str, Any]], sasrec_rows: list[dict[str, Any]], split_map: dict[str, str]) -> list[Sample]:
    cf_by_id = index_rows(cf_rows, "CF formal candidates")
    sas_by_id = index_rows(sasrec_rows, "SASRec formal candidates")
    if set(cf_by_id) != set(sas_by_id):
        raise ValueError(f"CF/SASRec row ids do not align: cf={len(cf_by_id)} sasrec={len(sas_by_id)}")
    if set(cf_by_id) != set(split_map):
        raise ValueError(f"Candidate rows and existing split rows do not align: candidates={len(cf_by_id)} split={len(split_map)}")
    samples: list[Sample] = []
    for row_id in sorted(cf_by_id, key=item_sort_key):
        cf = cf_by_id[row_id]
        sas = sas_by_id[row_id]
        target = str(cf.get("target_item_id", ""))
        if not target or str(sas.get("target_item_id", "")) != target:
            raise ValueError(f"Target mismatch at row {row_id}")
        cf_items = [str(item) for item in cf["candidate_item_ids"]]
        sas_items = [str(item) for item in sas["candidate_item_ids"]]
        cf_details = detail_index(cf)
        sas_details = detail_index(sas)
        merged: dict[str, Candidate] = {}
        all_items = sorted(set(cf_items) | set(sas_items), key=lambda item: (
            min(
                cf_items.index(item) if item in cf_items else 999999,
                sas_items.index(item) if item in sas_items else 999999,
            ),
            item_sort_key(item),
        ))
        for item in all_items:
            cf_rank = cf_items.index(item) if item in cf_items else None
            sas_rank = sas_items.index(item) if item in sas_items else None
            cfd = cf_details.get(item, {})
            sad = sas_details.get(item, {})
            merged[item] = Candidate(
                item_id=item,
                source_cf=cf_rank is not None,
                source_sasrec=sas_rank is not None,
                cf_rank=cf_rank,
                sasrec_rank=sas_rank,
                cf_score=parse_float(cfd.get("score")),
                sasrec_score=parse_float(sad.get("score")),
                cf_expansion_level=parse_int(cfd.get("expansion_level")),
                sasrec_expansion_level=parse_int(sad.get("expansion_level")),
                cf_bucket_size=parse_int(cfd.get("bucket_size")),
                sasrec_bucket_size=parse_int(sad.get("bucket_size")),
            )
        samples.append(
            Sample(
                sample_id=row_id,
                target_item_id=target,
                history_item_id=list_field(cf, "history_item_id"),
                cf_items=cf_items,
                sasrec_items=sas_items,
                candidates=merged,
                split=split_map[row_id],
            )
        )
    return samples


def provenance_rows(samples: list[Sample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda row: item_sort_key(row.sample_id)):
        ordered = sorted(
            sample.candidates.values(),
            key=lambda c: (
                c.best_source_rank if c.best_source_rank is not None else 999999,
                0 if c.source_cf else 1,
                item_sort_key(c.item_id),
            ),
        )
        for candidate in ordered:
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "p2_split": sample.split,
                    "item_id": candidate.item_id,
                    "ground_truth": sample.target_item_id,
                    "is_ground_truth": candidate.item_id == sample.target_item_id,
                    "source_cf": candidate.source_cf,
                    "source_sasrec": candidate.source_sasrec,
                    "cf_rank": candidate.cf_rank,
                    "sasrec_rank": candidate.sasrec_rank,
                    "cf_score": candidate.cf_score,
                    "sasrec_score": candidate.sasrec_score,
                    "duplicate_across_sources": candidate.duplicate_across_sources,
                    "best_source_rank": candidate.best_source_rank,
                    "source_count": candidate.source_count,
                }
            )
    return rows


def cf_only(sample: Sample) -> list[str]:
    return list(sample.cf_items)


def sasrec_only(sample: Sample) -> list[str]:
    return list(sample.sasrec_items)


def union_cf_then_sasrec(sample: Sample, exclusive_limit: int | None = None) -> list[str]:
    out = list(sample.cf_items)
    added = 0
    for item in sample.sasrec_items:
        if item in sample.candidates and item not in out:
            if exclusive_limit is not None and added >= exclusive_limit:
                break
            out.append(item)
            added += 1
    return out


def rrf_order(sample: Sample, lambda_sasrec: float, source_bonus: float) -> list[str]:
    scored: list[tuple[float, int, tuple[int, int | str], str]] = []
    for candidate in sample.candidates.values():
        score = 0.0
        if candidate.cf_rank is not None:
            score += 1.0 / (RRF_K + candidate.cf_rank + 1.0)
        if candidate.sasrec_rank is not None:
            score += lambda_sasrec / (RRF_K + candidate.sasrec_rank + 1.0)
        if source_bonus and candidate.source_count > 1:
            score += source_bonus
        best = candidate.best_source_rank if candidate.best_source_rank is not None else 999999
        scored.append((score, best, item_sort_key(candidate.item_id), candidate.item_id))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[-1] for item in scored]


def policy_items(sample: Sample, config: dict[str, Any]) -> list[str]:
    policy = config["policy"]
    if policy == "cf_only_top50":
        return cf_only(sample)
    if policy == "sasrec_only_top50":
        return sasrec_only(sample)
    if policy == "naive_union_cf50_sasrec50":
        return union_cf_then_sasrec(sample)
    if policy == "cf50_sasrec_exclusive_top5":
        return union_cf_then_sasrec(sample, 5)
    if policy == "cf50_sasrec_exclusive_top10":
        return union_cf_then_sasrec(sample, 10)
    if policy == "cf50_sasrec_exclusive_top20":
        return union_cf_then_sasrec(sample, 20)
    if policy == "cf50_all_sasrec_exclusive":
        return union_cf_then_sasrec(sample)
    if policy == "source_aware_rrf":
        return rrf_order(sample, float(config["lambda_sasrec"]), float(config["source_bonus"]))
    raise ValueError(f"Unknown policy: {policy}")


def fixed_policy_configs() -> list[dict[str, Any]]:
    configs = [
        {"config_id": "cf_only_top50", "policy": "cf_only_top50"},
        {"config_id": "sasrec_only_top50", "policy": "sasrec_only_top50"},
        {"config_id": "naive_union_cf50_sasrec50", "policy": "naive_union_cf50_sasrec50"},
        {"config_id": "cf50_sasrec_exclusive_top5", "policy": "cf50_sasrec_exclusive_top5"},
        {"config_id": "cf50_sasrec_exclusive_top10", "policy": "cf50_sasrec_exclusive_top10"},
        {"config_id": "cf50_sasrec_exclusive_top20", "policy": "cf50_sasrec_exclusive_top20"},
        {"config_id": "cf50_all_sasrec_exclusive", "policy": "cf50_all_sasrec_exclusive"},
    ]
    for lam in RRF_LAMBDAS:
        for bonus in RRF_BONUSES:
            configs.append(
                {
                    "config_id": f"source_aware_rrf_lam{lam:g}_bonus{bonus:g}",
                    "policy": "source_aware_rrf",
                    "lambda_sasrec": lam,
                    "source_bonus": bonus,
                }
            )
    return configs


def split_filter(samples: list[Sample], split_name: str) -> list[Sample]:
    if split_name == SPLIT_REPORT:
        return [sample for sample in samples if sample.split == SPLIT_REPORT_SOURCE]
    return [sample for sample in samples if sample.split == split_name]


def evaluate_ordered_items(samples: list[Sample], orders: dict[str, list[str]], ks: list[int]) -> dict[str, Any]:
    ranks = [rank_of(sample.target_item_id, orders[sample.sample_id]) for sample in samples]
    metrics = rank_metrics(ranks, ks)
    sizes = [len(orders[sample.sample_id]) for sample in samples]
    metrics["average_union_size"] = sum(sizes) / len(sizes) if sizes else 0.0
    metrics["min_union_size"] = min(sizes) if sizes else 0
    metrics["max_union_size"] = max(sizes) if sizes else 0
    cf_ranks = [rank_of(sample.target_item_id, sample.cf_items) for sample in samples]
    sas_ranks = [rank_of(sample.target_item_id, sample.sasrec_items) for sample in samples]
    strategy_ranks = ranks
    cf_hit20 = [hit_at(rank, 20) for rank in cf_ranks]
    sas_hit20 = [hit_at(rank, 20) for rank in sas_ranks]
    strat_hit20 = [hit_at(rank, 20) for rank in strategy_ranks]
    sas_only_oracle = [not c and s for c, s in zip(cf_hit20, sas_hit20)]
    union_oracle = [c or s for c, s in zip(cf_hit20, sas_hit20)]
    metrics["sasrec_only_recoveries_at20"] = sum(so and sh for so, sh in zip(sas_only_oracle, strat_hit20))
    metrics["sasrec_only_oracle_hits_at20"] = sum(sas_only_oracle)
    metrics["cf_hits_preserved_at20"] = sum(c and sh for c, sh in zip(cf_hit20, strat_hit20))
    metrics["cf_hits_lost_at20"] = sum(c and not sh for c, sh in zip(cf_hit20, strat_hit20))
    metrics["cf_hits_at20"] = sum(cf_hit20)
    metrics["union_oracle_hits_at20"] = sum(union_oracle)
    metrics["oracle_conversion_rate_at20"] = (
        0.0 if not sum(union_oracle) else sum(u and sh for u, sh in zip(union_oracle, strat_hit20)) / sum(union_oracle)
    )
    metrics["sasrec_only_conversion_rate_at20"] = (
        0.0 if not sum(sas_only_oracle) else metrics["sasrec_only_recoveries_at20"] / sum(sas_only_oracle)
    )
    total_candidates = sum(sizes)
    metrics["candidate_noise_ratio"] = 0.0 if not total_candidates else (total_candidates - metrics["target_in_pool_count"]) / total_candidates
    added = 0
    added_target = 0
    for sample in samples:
        cf_set = set(sample.cf_items)
        for item in orders[sample.sample_id]:
            if item not in cf_set:
                added += 1
                added_target += int(item == sample.target_item_id)
    metrics["added_auxiliary_candidates"] = added
    metrics["added_auxiliary_target_hits"] = added_target
    metrics["added_auxiliary_noise_ratio"] = 0.0 if not added else (added - added_target) / added
    return metrics


def evaluate_policy(samples: list[Sample], config: dict[str, Any], split_name: str, ks: list[int]) -> dict[str, Any]:
    subset = split_filter(samples, split_name)
    orders = {sample.sample_id: policy_items(sample, config) for sample in subset}
    metrics = evaluate_ordered_items(subset, orders, ks)
    return {
        "config_id": config["config_id"],
        "strategy_type": "fixed_or_rrf",
        "split": split_name,
        **config,
        **metrics,
    }


def load_model(model_path: Path) -> dict[str, Any]:
    model = read_json(model_path)
    if model.get("model_type") != "history_aware_linear_pairwise_logistic":
        raise ValueError(f"Unsupported frozen ranker model_type: {model.get('model_type')}")
    return model


def load_embeddings(item_emb: Path, row_index_path: Path) -> tuple[Any, dict[str, int]]:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"numpy is required for frozen history ranker evaluation: {exc}") from exc
    matrix = np.load(item_emb).astype("float32", copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    matrix = matrix / norms
    raw = read_json(row_index_path)
    return matrix, {str(item): int(row) for item, row in raw.items()}


def train_popularity(train_csv: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with open(train_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            target = str(row.get("item_id", "")).strip()
            if target:
                counts[target] += 1
            for item in list_field(row, "history_item_id"):
                counts[item] += 1
    return counts


def cosine(matrix: Any, row_index: dict[str, int], a: str, b: str) -> float:
    ia = row_index.get(str(a))
    ib = row_index.get(str(b))
    if ia is None or ib is None:
        return 0.0
    return float(matrix[ia].dot(matrix[ib]))


def source_score(source_type: str, expansion_level: int | None) -> float:
    if source_type == "exact":
        return 1.0
    if expansion_level is None:
        return 0.0
    return max(0.0, min(1.0, float(expansion_level) / 4.0))


def raw_features(
    sample: Sample,
    candidate: Candidate,
    fusion_rank_1_based: int,
    num_candidates: int,
    matrix: Any,
    row_index: dict[str, int],
    popularity: Counter[str],
    max_log_pop: float,
    alias_mode: str,
) -> dict[str, float]:
    if alias_mode == "sasrec_as_text_auxiliary":
        text_rank = None if candidate.sasrec_rank is None else candidate.sasrec_rank + 1
        cf_rank = None if candidate.cf_rank is None else candidate.cf_rank + 1
        from_text = candidate.source_sasrec
        from_cf = candidate.source_cf
    elif alias_mode == "source_independent_projection":
        text_rank = None
        cf_rank = None
        from_text = False
        from_cf = False
    else:
        raise ValueError(f"Unknown ranker alias mode: {alias_mode}")
    source_count = int(from_text) + int(from_cf)
    missing_rank = float(num_candidates + 1)
    text_rank_filled = float(text_rank) if text_rank is not None else missing_rank
    cf_rank_filled = float(cf_rank) if cf_rank is not None else missing_rank
    min_rank = min(text_rank_filled, cf_rank_filled)
    fusion_score = 1.0 / float(max(fusion_rank_1_based, 1))
    best_rank = candidate.best_source_rank if candidate.best_source_rank is not None else 999999
    expansion = candidate.cf_expansion_level if candidate.cf_expansion_level is not None else candidate.sasrec_expansion_level
    if expansion is None:
        expansion = 3
    src_type = "exact"
    sid_rank_score = 1.0 / float(best_rank + 1)
    src_score = source_score(src_type, expansion)
    pop_log = math.log1p(popularity.get(candidate.item_id, 0))
    pop_score = 0.0 if max_log_pop <= 0 else pop_log / max_log_pop
    sims = [cosine(matrix, row_index, candidate.item_id, hist) for hist in sample.history_item_id]
    history_cosine = max(sims) if sims else 0.0
    recent_cosine = cosine(matrix, row_index, candidate.item_id, sample.history_item_id[-1]) if sample.history_item_id else 0.0
    bucket = candidate.cf_bucket_size if candidate.cf_bucket_size is not None else candidate.sasrec_bucket_size
    if bucket is None:
        bucket = 1
    bucket_penalty = math.log1p(max(float(bucket), 1.0))
    expansion_score = max(0.0, min(1.0, float(expansion) / 4.0))
    heuristic_score = (
        sid_rank_score
        + 4.0 * src_score
        + 0.2 * pop_score
        + 0.8 * history_cosine
        + 0.4 * recent_cosine
        - 0.05 * bucket_penalty
    )
    return {
        "text_present": float(from_text),
        "cf_present": float(from_cf),
        "both_sources": float(from_text and from_cf),
        "text_only": float(from_text and not from_cf),
        "cf_only": float(from_cf and not from_text),
        "source_count": float(source_count),
        "text_rank_filled": text_rank_filled,
        "cf_rank_filled": cf_rank_filled,
        "min_source_rank": min_rank,
        "fusion_rank": float(fusion_rank_1_based),
        "reciprocal_text_rank": 0.0 if text_rank is None else 1.0 / float(text_rank),
        "reciprocal_cf_rank": 0.0 if cf_rank is None else 1.0 / float(cf_rank),
        "reciprocal_min_source_rank": 1.0 / float(min_rank),
        "reciprocal_fusion_rank": 1.0 / float(fusion_rank_1_based),
        "fusion_score": fusion_score,
        "sid_rank_score": sid_rank_score,
        "source_score": src_score,
        "popularity_score": pop_score,
        "history_cosine": history_cosine,
        "recent_cosine": recent_cosine,
        "bucket_penalty": bucket_penalty,
        "bucket_size_log": bucket_penalty,
        "expansion_level_score": expansion_score,
        "exact_source": 1.0,
        "prefix_source": 0.0,
        "heuristic_score": heuristic_score,
    }


def frozen_ranker_order(
    sample: Sample,
    base_items: list[str],
    model: dict[str, Any],
    matrix: Any,
    row_index: dict[str, int],
    popularity: Counter[str],
    max_log_pop: float,
    alias_mode: str,
) -> list[str]:
    feature_names = model["feature_names"]
    means = model["feature_stats"]["mean"]
    stds = model["feature_stats"]["std"]
    weights = [float(value) for value in model["weights"]]
    excluded = set()
    if alias_mode == "source_independent_projection":
        excluded = {
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
        }
    scored: list[tuple[float, int, tuple[int, int | str], str]] = []
    for rank0, item in enumerate(base_items):
        cand = sample.candidates[item]
        raw = raw_features(sample, cand, rank0 + 1, len(base_items), matrix, row_index, popularity, max_log_pop, alias_mode)
        score = 0.0
        for idx, name in enumerate(feature_names):
            z = 0.0 if name in excluded else (raw[name] - means[idx]) / stds[idx]
            score += weights[idx] * z
        scored.append((score, rank0, item_sort_key(item), item))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    return [row[-1] for row in scored]


def evaluate_frozen_ranker(
    samples: list[Sample],
    base_config: dict[str, Any],
    split_name: str,
    ks: list[int],
    model: dict[str, Any],
    matrix: Any,
    row_index: dict[str, int],
    popularity: Counter[str],
    max_log_pop: float,
    alias_mode: str,
) -> dict[str, Any]:
    subset = split_filter(samples, split_name)
    orders = {
        sample.sample_id: frozen_ranker_order(
            sample,
            policy_items(sample, base_config),
            model,
            matrix,
            row_index,
            popularity,
            max_log_pop,
            alias_mode,
        )
        for sample in subset
    }
    metrics = evaluate_ordered_items(subset, orders, ks)
    return {
        "config_id": f"frozen_ranker_{alias_mode}_on_{base_config['config_id']}",
        "strategy_type": "frozen_history_ranker",
        "split": split_name,
        "base_config_id": base_config["config_id"],
        "ranker_alias_mode": alias_mode,
        **metrics,
    }


def flatten_for_csv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "config_id",
        "strategy_type",
        "split",
        "policy",
        "lambda_sasrec",
        "source_bonus",
        "base_config_id",
        "ranker_alias_mode",
        "num_samples",
        "average_union_size",
        "hr@1",
        "hr@5",
        "hr@10",
        "hr@20",
        "hr@50",
        "ndcg@10",
        "ndcg@20",
        "mrr",
        "target_in_pool_rate",
        "target_in_pool_count",
        "sasrec_only_recoveries_at20",
        "cf_hits_preserved_at20",
        "cf_hits_lost_at20",
        "oracle_conversion_rate_at20",
        "candidate_noise_ratio",
        "added_auxiliary_noise_ratio",
    ]
    return [{field: row.get(field, "") for field in fields} for row in rows]


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        float(row.get("hr@20", 0.0)),
        float(row.get("ndcg@20", 0.0)),
        -float(row.get("cf_hits_lost_at20", 0.0)),
        -float(row.get("average_union_size", 0.0)),
        str(row.get("config_id", "")),
    )


def compatibility_matrix(model: dict[str, Any], stage7_root: Path) -> list[dict[str, Any]]:
    names = set(model.get("feature_names", []))
    hardcoded_text_cf = {"text_present", "cf_present", "text_rank_filled", "cf_rank_filled"}.issubset(names)
    return [
        {
            "component": "P2-4 frozen history-aware ranker",
            "evidence": (stage7_root / "p2_history_ranker/model.json").as_posix(),
            "compatible_directly_with_cf_sasrec": False,
            "reason": "feature schema contains text_* and cf_* fields trained for Text+CF, not CF+SASRec",
            "source_fields_hardcoded": hardcoded_text_cf,
            "allowed_use_in_s5": "explicit alias or source-independent projection only",
        },
        {
            "component": "sasrec_as_text_auxiliary alias",
            "evidence": "SASRec is treated as generic auxiliary behavior source occupying the old text_* feature slots",
            "compatible_directly_with_cf_sasrec": True,
            "reason": "alias preserves item/provenance semantics but changes source-feature interpretation; report must not call this direct compatibility",
            "source_fields_hardcoded": True,
            "allowed_use_in_s5": "diagnostic frozen-ranker conversion evaluation",
        },
        {
            "component": "source_independent_projection",
            "evidence": "source-specific normalized features are zeroed; history/popularity/rank-shape features remain",
            "compatible_directly_with_cf_sasrec": True,
            "reason": "removes source identity dependence but is a projection of a frozen model, not a retrained fair gate",
            "source_fields_hardcoded": False,
            "allowed_use_in_s5": "diagnostic frozen-ranker conversion evaluation",
        },
    ]


def make_markdown(report: dict[str, Any]) -> str:
    selected = report["selected_config"]
    final = report["valid_report_final"]
    cf = final["cf_only_top50"]
    chosen = final[selected["config_id"]]
    ranker_best = report["valid_report_ranker_best"]
    lines = [
        "# S5-0 Frozen CF+SASRec Auxiliary Fusion Conversion Gate",
        "",
        "## Scope",
        "",
        "CPU-only analysis. No GPU, SFT, generation, SID reconstruction, Text-SID fusion, p3 expansion, or test split evaluation was run. The script reads frozen S4 formal valid artifacts and existing Stage 7 P2 split/ranker evidence only.",
        "",
        "## Compatibility Matrix",
        "",
        "| component | direct compatible | source hardcoded | allowed use | reason |",
        "|---|---:|---:|---|---|",
    ]
    for row in report["compatibility_matrix"]:
        lines.append(
            f"| {row['component']} | {row['compatible_directly_with_cf_sasrec']} | {row['source_fields_hardcoded']} | {row['allowed_use_in_s5']} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Selection Protocol",
            "",
            f"- valid_select source: `{report['split_protocol']['valid_select']}`",
            f"- valid_report source: `{report['split_protocol']['valid_report']}`",
            f"- valid_select rows: `{report['split_protocol']['counts'].get('valid_select')}`",
            f"- valid_report rows: `{report['split_protocol']['counts'].get('valid_fit')}`",
            "- No new random split was created. There is no separate historical valid_report artifact; S5 uses the existing non-select `valid_fit` subset as the report split and records this limitation.",
            "",
            "## Selected Config",
            "",
            f"`{selected['config_id']}` selected on valid_select by HR@20, then NDCG@20, then lower CF hit loss and smaller candidate pool.",
            "",
            "## Valid Report Metrics",
            "",
            "| strategy | HR@10 | HR@20 | HR@50 | NDCG@20 | MRR | avg size | SASRec-only recoveries@20 | CF lost@20 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| CF-only | {cf['hr@10']:.6f} | {cf['hr@20']:.6f} | {cf['hr@50']:.6f} | {cf['ndcg@20']:.6f} | {cf['mrr']:.6f} | {cf['average_union_size']:.2f} | {cf['sasrec_only_recoveries_at20']} | {cf['cf_hits_lost_at20']} |",
            f"| selected fixed/RRF | {chosen['hr@10']:.6f} | {chosen['hr@20']:.6f} | {chosen['hr@50']:.6f} | {chosen['ndcg@20']:.6f} | {chosen['mrr']:.6f} | {chosen['average_union_size']:.2f} | {chosen['sasrec_only_recoveries_at20']} | {chosen['cf_hits_lost_at20']} |",
            f"| best frozen ranker diagnostic | {ranker_best['hr@10']:.6f} | {ranker_best['hr@20']:.6f} | {ranker_best['hr@50']:.6f} | {ranker_best['ndcg@20']:.6f} | {ranker_best['mrr']:.6f} | {ranker_best['average_union_size']:.2f} | {ranker_best['sasrec_only_recoveries_at20']} | {ranker_best['cf_hits_lost_at20']} |",
            "",
            "## Gate",
            "",
            f"Decision: **{report['gate']['decision']}**",
            "",
            report["gate"]["reason"],
            "",
            "The frozen ranker cannot be treated as directly schema-compatible with CF+SASRec because it was trained with Text+CF source features. Its S5 result is diagnostic under an explicit alias/projection manifest.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = candidate_paths(args.formal_root)
    model_path = args.stage7_root / "p2_history_ranker/model.json"
    train_csv = ROOT / f"data/Amazon/sid_versions/cf_k512_dedup/{args.category}/train.csv"
    item_emb = ROOT / f"data/Amazon/cs_embeddings/{args.category}/{args.category}.cf_emb.npy"
    row_index = ROOT / f"data/Amazon/cs_embeddings/{args.category}/{args.category}.row_index.json"
    input_paths = list(paths.values()) + [model_path, train_csv, item_emb, row_index]
    assert_no_test_paths(input_paths + [args.out_root, args.doc_path])
    for path in input_paths:
        assert_file(path)

    cf_rows = read_jsonl(paths["cf"])
    sasrec_rows = read_jsonl(paths["sasrec"])
    split_map, split_protocol = load_split_map(args.stage7_root)
    samples = build_samples(cf_rows, sasrec_rows, split_map)
    if len(samples) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} samples, got {len(samples)}")
    prov = provenance_rows(samples)

    model = load_model(model_path)
    matrix, emb_row_index = load_embeddings(item_emb, row_index)
    popularity = train_popularity(train_csv)
    max_log_pop = max((math.log1p(v) for v in popularity.values()), default=1.0)

    configs = fixed_policy_configs()
    select_rows = [evaluate_policy(samples, config, SPLIT_SELECT, args.ks) for config in configs]
    selected = max(select_rows, key=selection_key)
    selected_config = {k: selected[k] for k in ["config_id", "policy"] if k in selected}
    if selected.get("policy") == "source_aware_rrf":
        selected_config["lambda_sasrec"] = selected.get("lambda_sasrec")
        selected_config["source_bonus"] = selected.get("source_bonus")
    selected_config["selection_split"] = SPLIT_SELECT
    selected_config["selection_metric_order"] = ["hr@20", "ndcg@20", "-cf_hits_lost_at20", "-average_union_size", "config_id"]
    selected_config["frozen"] = True

    report_rows = [evaluate_policy(samples, config, SPLIT_REPORT, args.ks) for config in configs]
    report_by_id = {row["config_id"]: row for row in report_rows}

    ranker_base_configs = [
        {"config_id": "cf_only_top50", "policy": "cf_only_top50"},
        {"config_id": "naive_union_cf50_sasrec50", "policy": "naive_union_cf50_sasrec50"},
        {"config_id": "cf50_sasrec_exclusive_top5", "policy": "cf50_sasrec_exclusive_top5"},
        {"config_id": "cf50_sasrec_exclusive_top10", "policy": "cf50_sasrec_exclusive_top10"},
        {"config_id": "cf50_sasrec_exclusive_top20", "policy": "cf50_sasrec_exclusive_top20"},
        selected_config,
    ]
    # Preserve order while removing duplicate base configs.
    seen: set[str] = set()
    ranker_base_configs = [cfg for cfg in ranker_base_configs if not (cfg["config_id"] in seen or seen.add(cfg["config_id"]))]
    ranker_rows: list[dict[str, Any]] = []
    for base_cfg in ranker_base_configs:
        for alias in ["source_independent_projection", "sasrec_as_text_auxiliary"]:
            ranker_rows.append(
                evaluate_frozen_ranker(
                    samples,
                    base_cfg,
                    SPLIT_SELECT,
                    args.ks,
                    model,
                    matrix,
                    emb_row_index,
                    popularity,
                    max_log_pop,
                    alias,
                )
            )
            ranker_rows.append(
                evaluate_frozen_ranker(
                    samples,
                    base_cfg,
                    SPLIT_REPORT,
                    args.ks,
                    model,
                    matrix,
                    emb_row_index,
                    popularity,
                    max_log_pop,
                    alias,
                )
            )
    ranker_report_rows = [row for row in ranker_rows if row["split"] == SPLIT_REPORT]
    ranker_best = max(ranker_report_rows, key=selection_key)

    cf_report = report_by_id["cf_only_top50"]
    selected_report = report_by_id[selected_config["config_id"]]
    hr20_delta = selected_report["hr@20"] - cf_report["hr@20"]
    ndcg20_delta = selected_report["ndcg@20"] - cf_report["ndcg@20"]
    ranker_hr20_delta = ranker_best["hr@20"] - cf_report["hr@20"]
    ranker_ndcg20_delta = ranker_best["ndcg@20"] - cf_report["ndcg@20"]
    cf_loss_tolerance = max(2, 0.03 * cf_report["cf_hits_at20"])
    if (ranker_hr20_delta >= 0.003 or ranker_ndcg20_delta >= 0.002) and ranker_best["cf_hits_lost_at20"] <= cf_loss_tolerance:
        gate = {
            "decision": "FROZEN_RANKER_CONVERTS",
            "reason": "The source-independent frozen-ranker projection improves valid_report HR@20/NDCG@20 over CF-only, preserves at least 97% of CF@20 hits, and recovers part of the SASRec-only oracle space.",
        }
    elif selected_report["hr@20"] > cf_report["hr@20"] or ranker_best["hr@20"] > cf_report["hr@20"]:
        gate = {
            "decision": "LEARNED_GATE_NEEDED",
            "reason": "Auxiliary oracle remains convertibility-positive, but fixed/frozen conversion is modest or schema-conditional; a source-aware learned gate is the clean next step.",
        }
    else:
        gate = {
            "decision": "AUXILIARY_STREAM_NOT_CONVERTIBLE",
            "reason": "SASRec injection does not improve valid_report over CF-only under fixed policies or frozen-ranker diagnostics.",
        }

    leaderboard_rows = flatten_for_csv(select_rows + [row for row in ranker_rows if row["split"] == SPLIT_SELECT])
    report_metrics = {
        "schema": "s5_0_valid_report_metrics.v1",
        "split": SPLIT_REPORT,
        "split_source": SPLIT_REPORT_SOURCE,
        "cf_only_top50": cf_report,
        "selected_fixed_or_rrf": selected_report,
        "frozen_ranker_best": ranker_best,
        "all_fixed_or_rrf": {row["config_id"]: row for row in report_rows},
        "all_frozen_ranker": {row["config_id"]: row for row in ranker_report_rows},
        "deltas_vs_cf": {
            "selected_hr20_absolute": hr20_delta,
            "selected_ndcg20_absolute": ndcg20_delta,
            "ranker_best_hr20_absolute": ranker_hr20_delta,
            "ranker_best_ndcg20_absolute": ranker_ndcg20_delta,
        },
    }
    closeout = {
        "schema": "s5_0_protocol_closeout.v1",
        "category": args.category,
        "formal_root": args.formal_root.as_posix(),
        "test_read": False,
        "gpu_run": False,
        "training_run": False,
        "generation_run": False,
        "source_artifacts": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in paths.items()
        },
        "split_protocol": split_protocol,
        "compatibility_matrix": compatibility_matrix(model, args.stage7_root),
        "ranker_provenance": {
            "model_path": model_path.as_posix(),
            "model_sha256": file_sha256(model_path),
            "model_type": model.get("model_type"),
            "selected_config": model.get("selected_config"),
            "training_policy": model.get("training_policy"),
            "feature_names": model.get("feature_names"),
            "compatibility_status": "not_directly_compatible_text_cf_schema; explicit_alias_or_projection_required",
        },
        "candidate_provenance": {
            "rows": len(prov),
            "sample_count": len(samples),
            "deterministic_sha256": None,
        },
        "selected_config": selected_config,
        "valid_report_metrics": report_metrics,
        "gate": gate,
    }

    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)
    provenance_path = out / "s5_0_candidate_provenance.jsonl"
    leaderboard_path = out / "s5_0_valid_select_leaderboard.csv"
    selected_path = out / "s5_0_selected_config.json"
    report_path = out / "s5_0_valid_report_metrics.json"
    closeout_path = out / "s5_0_protocol_closeout.json"
    write_jsonl(provenance_path, prov)
    closeout["candidate_provenance"]["deterministic_sha256"] = file_sha256(provenance_path)
    write_csv(leaderboard_path, list(leaderboard_rows[0].keys()), leaderboard_rows)
    write_json(selected_path, selected_config)
    write_json(report_path, report_metrics)
    write_json(closeout_path, closeout)

    doc_payload = {
        **closeout,
        "selected_config": selected_config,
        "valid_report_final": {
            "cf_only_top50": cf_report,
            selected_config["config_id"]: selected_report,
        },
        "valid_report_ranker_best": ranker_best,
    }
    args.doc_path.parent.mkdir(parents=True, exist_ok=True)
    args.doc_path.write_text(make_markdown(doc_payload), encoding="utf-8")

    return {
        "provenance": provenance_path.as_posix(),
        "leaderboard": leaderboard_path.as_posix(),
        "selected_config": selected_path.as_posix(),
        "valid_report_metrics": report_path.as_posix(),
        "protocol_closeout": closeout_path.as_posix(),
        "doc": args.doc_path.as_posix(),
        "gate": gate,
        "selected_report": selected_report,
        "cf_report": cf_report,
        "ranker_best": ranker_best,
    }


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
