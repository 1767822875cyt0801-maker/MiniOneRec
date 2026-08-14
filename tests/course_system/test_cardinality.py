from __future__ import annotations

from dataclasses import dataclass

import pytest

from minionerec_system.cardinality import CardinalityGateError, apply_budget, audit_cardinality
from minionerec_system.schemas import PreBudgetSample


@dataclass(frozen=True)
class DummySource:
    marker: str = "fusion_complete"


def _sample(sample_id: str, count: int, target: str = "0") -> PreBudgetSample:
    return PreBudgetSample(
        sample_id=sample_id,
        target_item_id=target,
        cf_candidate_count=50,
        sasrec_candidate_count=50,
        pre_dedup_count=100,
        ordered_item_ids=tuple(str(idx) for idx in range(count)),
        source_sample=DummySource(),
    )


def test_k20_k50_k75_k90_all_counts_and_gate_pass() -> None:
    audit = audit_cardinality(
        [_sample("0", 100), _sample("1", 95)], [20, 50, 75, 90, "all"], 20, True, 0.10
    )
    first = audit.per_sample[0]
    assert [first[f"effective_count_at_{key}"] for key in ["k20", "k50", "k75", "k90", "all"]] == [
        20,
        50,
        75,
        90,
        100,
    ]
    vectors = audit.summary["budgets"]
    assert vectors["k90"]["effective_count_vector"] != vectors["k75"]["effective_count_vector"]
    assert vectors["k90"]["effective_count_vector"] != vectors["all"]["effective_count_vector"]
    assert audit.summary["gate"]["status"] == "PASS"


def test_identical_effective_vectors_are_rejected() -> None:
    with pytest.raises(CardinalityGateError) as exc_info:
        audit_cardinality([_sample("0", 50), _sample("1", 50)], [20, 50, 75, 90, "all"], 20, True, 0.10)
    messages = " ".join(exc_info.value.audit.summary["gate"]["errors"])
    assert "identical effective-count vectors" in messages or "same effective-count vector as all" in messages


def test_zero_truncation_finite_k_is_rejected() -> None:
    with pytest.raises(CardinalityGateError) as exc_info:
        audit_cardinality([_sample("0", 70)], [20, 50, 75, 90, "all"], 20, True, 0.10)
    assert "K=75 truncates zero samples" in exc_info.value.audit.summary["gate"]["errors"]


def test_ground_truth_is_not_specially_retained() -> None:
    pre = _sample("0", 90, target="89")
    post = apply_budget([pre], 20)[0]
    assert len(post.ordered_item_ids) == 20
    assert "89" not in post.ordered_item_ids
    assert post.stage == "post_fusion_pre_rerank"


def test_duplicate_fusion_items_are_rejected() -> None:
    bad = PreBudgetSample("0", "0", 2, 2, 4, ("0", "1", "1"), DummySource())
    with pytest.raises(CardinalityGateError):
        audit_cardinality([bad], [20, 50, 75, 90, "all"], 20, True, 0.10)
