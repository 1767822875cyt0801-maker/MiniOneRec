"""Quality adapter that delegates ranking metrics to the frozen S5 implementation."""

from __future__ import annotations

from typing import Any

from scripts import s5_auxiliary_fusion_conversion as s5_impl

from .schemas import RankingSample


def evaluate_rankings(rankings: list[RankingSample]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pool_ranks = [s5_impl.rank_of(row.target_item_id, list(row.candidate_item_ids)) for row in rankings]
    ranked_ranks = [s5_impl.rank_of(row.target_item_id, list(row.ranked_item_ids)) for row in rankings]
    pool_metrics = s5_impl.rank_metrics(pool_ranks, [10, 20])
    ranking_metrics = s5_impl.rank_metrics(ranked_ranks, [10, 20])
    per_sample: list[dict[str, Any]] = []
    for row, pool_rank, ranked_rank in zip(rankings, pool_ranks, ranked_ranks):
        per_sample.append(
            {
                "sample_id": row.sample_id,
                "budget": str(row.budget),
                "target_item_id": row.target_item_id,
                "candidate_count": len(row.candidate_item_ids),
                "candidate_hit_rank_0_based": "" if pool_rank is None else pool_rank,
                "ranked_hit_rank_0_based": "" if ranked_rank is None else ranked_rank,
                "candidate_pool_hit": pool_rank is not None,
                "hit_at_10": ranked_rank is not None and ranked_rank < 10,
                "hit_at_20": ranked_rank is not None and ranked_rank < 20,
            }
        )
    return {
        "candidate_recall": pool_metrics["target_in_pool_rate"],
        "hr@10": ranking_metrics["hr@10"],
        "hr@20": ranking_metrics["hr@20"],
        "ndcg@10": ranking_metrics["ndcg@10"],
        "ndcg@20": ranking_metrics["ndcg@20"],
        "sample_count": len(rankings),
    }, per_sample
