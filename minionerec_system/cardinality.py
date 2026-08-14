"""Candidate-cardinality audit and the post-fusion/pre-rerank budget gate."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .schemas import BudgetValue, PostBudgetSample, PreBudgetSample, budget_label


@dataclass(frozen=True)
class CardinalityAudit:
    per_sample: list[dict[str, Any]]
    summary: dict[str, Any]


class CardinalityGateError(ValueError):
    def __init__(self, audit: CardinalityAudit):
        self.audit = audit
        super().__init__("; ".join(audit.summary["gate"]["errors"]))


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def apply_budget(samples: list[PreBudgetSample], budget: BudgetValue) -> list[PostBudgetSample]:
    before_ids = {sample.sample_id for sample in samples}
    out: list[PostBudgetSample] = []
    for sample in samples:
        items = sample.ordered_item_ids if budget == "all" else sample.ordered_item_ids[:budget]
        if budget != "all" and len(items) > budget:
            raise ValueError(f"post-budget count exceeds K={budget} for sample {sample.sample_id}")
        out.append(
            PostBudgetSample(
                sample_id=sample.sample_id,
                target_item_id=sample.target_item_id,
                budget=budget,
                ordered_item_ids=tuple(items),
                source_sample=sample.source_sample,
            )
        )
    if {sample.sample_id for sample in out} != before_ids:
        raise ValueError("sample set changed across candidate budget")
    return out


def audit_cardinality(
    samples: list[PreBudgetSample],
    budgets: list[BudgetValue],
    top_n: int,
    fail_if_inactive: bool,
    warn_affected_fraction_below: float,
    raise_on_error: bool = True,
) -> CardinalityAudit:
    errors: list[str] = []
    warnings: list[str] = []
    finite = [value for value in budgets if isinstance(value, int)]
    for value in finite:
        if value < top_n:
            errors.append(f"K={value} is smaller than Top-{top_n}")
    if "all" not in budgets:
        errors.append("the unbounded all baseline is missing")

    rows: list[dict[str, Any]] = []
    vectors: dict[str, list[int]] = {budget_label(value): [] for value in budgets}
    for sample in samples:
        if len(sample.ordered_item_ids) != len(set(sample.ordered_item_ids)):
            errors.append(f"fusion output contains duplicate item for sample {sample.sample_id}")
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "cf_candidate_count": sample.cf_candidate_count,
            "sasrec_candidate_count": sample.sasrec_candidate_count,
            "pre_dedup_count": sample.pre_dedup_count,
            "post_dedup_count": sample.post_dedup_count,
        }
        for value in budgets:
            label = budget_label(value)
            effective = sample.post_dedup_count if value == "all" else min(sample.post_dedup_count, value)
            vectors[label].append(effective)
            row[f"effective_count_at_{label}"] = effective
            if value != "all":
                row[f"truncated_at_{label}"] = sample.post_dedup_count > value
                if effective > value:
                    errors.append(f"post-budget count exceeds K={value} for sample {sample.sample_id}")
        rows.append(row)

    all_vector = vectors.get("all", [])
    for value in finite:
        label = budget_label(value)
        truncated = sum(1 for before, after in zip(all_vector, vectors[label]) if after < before)
        if truncated == 0 and fail_if_inactive:
            errors.append(f"K={value} truncates zero samples")
        fraction = 0.0 if not samples else truncated / len(samples)
        if 0.0 < fraction < warn_affected_fraction_below:
            warnings.append(
                f"K={value} affects only {fraction:.6f} of samples; threshold={warn_affected_fraction_below:.6f}"
            )
        if vectors[label] == all_vector:
            errors.append(f"finite K={value} has the same effective-count vector as all")
    for left_idx, left in enumerate(finite):
        for right in finite[left_idx + 1 :]:
            if vectors[budget_label(left)] == vectors[budget_label(right)]:
                errors.append(f"K={left} and K={right} have identical effective-count vectors")

    counts = [sample.post_dedup_count for sample in samples]
    budget_aggregates: dict[str, Any] = {}
    for value in budgets:
        label = budget_label(value)
        effective = vectors[label]
        removed = [before - after for before, after in zip(counts, effective)]
        truncated = sum(amount > 0 for amount in removed)
        budget_aggregates[label] = {
            "total_candidates_after_budget": sum(effective),
            "fraction_truncated": 0.0 if not samples else truncated / len(samples),
            "mean_removed": 0.0 if not samples else sum(removed) / len(samples),
            "effective_count_vector": effective,
        }
    summary = {
        "schema": "course_candidate_cardinality.v1",
        "budget_position": "post_fusion_pre_rerank",
        "sample_count": len(samples),
        "min": min(counts) if counts else 0,
        "mean": statistics.fmean(counts) if counts else 0.0,
        "median": statistics.median(counts) if counts else 0.0,
        "p75": _percentile(counts, 0.75),
        "p95": _percentile(counts, 0.95),
        "max": max(counts) if counts else 0,
        "total_candidates_before_budget": sum(counts),
        "candidate_count_histogram": {str(key): value for key, value in sorted(Counter(counts).items())},
        "budgets": budget_aggregates,
        "gate": {
            "status": "FAIL" if errors else "PASS",
            "errors": sorted(set(errors)),
            "warnings": sorted(set(warnings)),
            "fail_if_inactive": fail_if_inactive,
            "warn_affected_fraction_below": warn_affected_fraction_below,
        },
    }
    audit = CardinalityAudit(rows, summary)
    if errors and raise_on_error:
        raise CardinalityGateError(audit)
    return audit
