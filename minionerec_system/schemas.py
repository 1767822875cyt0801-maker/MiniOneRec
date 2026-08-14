"""Small data contracts shared by the course-system adapters and pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


BudgetValue = int | Literal["all"]


def budget_label(value: BudgetValue) -> str:
    return "all" if value == "all" else f"k{value}"


@dataclass(frozen=True)
class PreBudgetSample:
    """A stable, deduplicated fusion output before any candidate budget."""

    sample_id: str
    target_item_id: str
    cf_candidate_count: int
    sasrec_candidate_count: int
    pre_dedup_count: int
    ordered_item_ids: tuple[str, ...]
    source_sample: Any

    @property
    def post_dedup_count(self) -> int:
        return len(self.ordered_item_ids)


@dataclass(frozen=True)
class PostBudgetSample:
    """The only candidate contract accepted by the course-system reranker."""

    sample_id: str
    target_item_id: str
    budget: BudgetValue
    ordered_item_ids: tuple[str, ...]
    source_sample: Any
    stage: str = "post_fusion_pre_rerank"


@dataclass(frozen=True)
class RankingSample:
    sample_id: str
    target_item_id: str
    budget: BudgetValue
    candidate_item_ids: tuple[str, ...]
    ranked_item_ids: tuple[str, ...]
    top_item_ids: tuple[str, ...]


@dataclass
class InputBundle:
    prediction_rows_cf: list[dict[str, Any]]
    prediction_rows_sasrec: list[dict[str, Any]]
    eval_rows: list[dict[str, str]]
    selected_indices: list[int]
    split_map: dict[str, str]
    cf_sid2items: dict[str, list[str]]
    sasrec_sid2items: dict[str, list[str]]

    def subset(self, size: int) -> "InputBundle":
        indices = self.selected_indices[: max(size, 0)]
        return InputBundle(
            prediction_rows_cf=self.prediction_rows_cf,
            prediction_rows_sasrec=self.prediction_rows_sasrec,
            eval_rows=self.eval_rows,
            selected_indices=indices,
            split_map={str(idx): self.split_map[str(idx)] for idx in indices},
            cf_sid2items=self.cf_sid2items,
            sasrec_sid2items=self.sasrec_sid2items,
        )
