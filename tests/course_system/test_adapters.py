from __future__ import annotations

from pathlib import Path

import pytest

from minionerec_system import adapters
from minionerec_system.config import PROJECT_ROOT, load_course_config
from minionerec_system.schemas import PostBudgetSample


CONFIG = load_course_config(PROJECT_ROOT / "configs/course_system/smoke_cf_sasrec_exact.yaml", PROJECT_ROOT)


def _candidate_row(items: list[str]) -> dict:
    return {
        "row_index": 0,
        "target_item_id": "2",
        "history_item_id": [],
        "candidate_item_ids": items,
        "candidate_details": [
            {"item_id": item, "source_type": "exact", "sid_rank_0_based": rank, "expansion_level": 3, "bucket_size": 1}
            for rank, item in enumerate(items)
        ],
    }


def test_dual_stream_duplicate_is_stably_deduplicated() -> None:
    fused = adapters.source_aware_fusion(
        [_candidate_row(["1", "2"])],
        [_candidate_row(["2", "3"])],
        {"0": "valid_select"},
        CONFIG,
    )
    assert len(fused) == 1
    assert set(fused[0].ordered_item_ids) == {"1", "2", "3"}
    assert len(fused[0].ordered_item_ids) == len(set(fused[0].ordered_item_ids))


def test_reranker_rejects_non_post_budget_contract() -> None:
    invalid = PostBudgetSample("0", "0", 20, ("0",), object(), stage="fusion_only")
    with pytest.raises(adapters.AdapterError, match="post-budget"):
        adapters.rerank_post_budget([invalid], None, 20, 0, False)  # type: ignore[arg-type]
