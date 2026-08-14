"""Frozen CF+SASRec downstream course pipeline orchestration."""

from __future__ import annotations

import json
import shlex
import time
from pathlib import Path
from typing import Any

from . import adapters
from .artifacts import ArtifactRun, FULL_ARTIFACTS, COMMON_ARTIFACTS
from .cardinality import CardinalityGateError, apply_budget, audit_cardinality
from .config import ConfigError, ResolvedConfig, verify_formal_execution_guard
from .metrics import evaluate_rankings
from .profiling import PROFILING_LABEL, PROFILING_SCOPE, StageProfiler, reset_cuda_peak_if_available
from .schemas import BudgetValue, InputBundle, RankingSample, budget_label


def build_command_plan(config: ResolvedConfig, run_mode: str) -> dict[str, Any]:
    stages = [
        {"name": "input_load", "loads_model": run_mode == "full_pipeline"},
        {"name": "candidate_expansion", "implementation": "evaluate_candidates.expand_candidates"},
        {"name": "fusion_and_dedup", "implementation": "s5_auxiliary_fusion_conversion.build_samples/rrf_order"},
        {"name": "cardinality_gate", "position": "post_fusion_pre_rerank"},
    ]
    if run_mode == "full_pipeline":
        stages.extend(
            [
                {"name": "candidate_budget", "values": config.data["budget"]["values"]},
                {"name": "history_rerank", "implementation": "s5_auxiliary_fusion_conversion.frozen_ranker_order"},
                {"name": "evaluation", "implementation": "s5_auxiliary_fusion_conversion.rank_metrics"},
            ]
        )
    return {
        "schema": "course_command_plan.v1",
        "experiment": config.data["experiment"]["name"],
        "run_mode": run_mode,
        "dataset": config.data["experiment"]["dataset"],
        "split": config.data["experiment"]["split"],
        "quality_split": config.data["evaluation"]["quality_split"],
        "retrieval_streams": ["CF-SID", "SASRec-SID"],
        "candidate_mode": "exact",
        "fusion": {
            "method": "source_aware_rrf",
            "lambda_sasrec": 0.75,
            "source_bonus": 0.01,
        },
        "budget_order_proof": ["fusion", "stable_dedup", "candidate_budget", "history_aware_rerank"],
        "profiling_scope": PROFILING_SCOPE,
        "profiling_label": PROFILING_LABEL,
        "stages": stages,
        "expected_artifacts": list(FULL_ARTIFACTS if run_mode == "full_pipeline" else COMMON_ARTIFACTS),
        "formal_execution": config.is_formal,
    }


def _candidate_total(rows: list[Any]) -> int:
    return sum(len(row.ordered_item_ids) for row in rows)


class CoursePipeline:
    def __init__(self, config: ResolvedConfig) -> None:
        self.config = config
        self.current_stage = "initialization"

    def dry_run(self, max_samples: int | None) -> dict[str, Any]:
        inspection = adapters.inspect_input_files(self.config)
        result: dict[str, Any] = {
            "status": "PASS",
            "mode": "dry_run",
            "formal_experiment_run": False,
            "input_inspection": inspection,
            "command_plan": build_command_plan(self.config, "full_pipeline"),
        }
        if max_samples is not None:
            bundle, input_meta = adapters.load_input_bundle(self.config, max_samples=max_samples)
            cf_rows, sasrec_rows = adapters.exact_candidate_expansion(bundle, self.config)
            fused = adapters.source_aware_fusion(cf_rows, sasrec_rows, bundle.split_map, self.config)
            audit = audit_cardinality(
                fused,
                self.config.budgets,
                int(self.config.data["budget"]["top_n"]),
                bool(self.config.data["budget"]["fail_if_inactive"]),
                float(self.config.data["budget"]["warn_affected_fraction_below"]),
            )
            result.update({"input_meta": input_meta, "sample_cardinality": audit.summary})
        return result

    def _prepare_pre_budget(
        self,
        bundle: InputBundle,
        profiler: StageProfiler,
        *,
        repeat: int,
        warmup: bool,
        budget: str,
    ) -> list[Any]:
        sample_count = len(bundle.selected_indices)
        self.current_stage = "candidate_expansion"
        cf_rows, sasrec_rows = profiler.call(
            "candidate_expansion",
            lambda: adapters.exact_candidate_expansion(bundle, self.config),
            sample_count=sample_count,
            repeat=repeat,
            warmup=warmup,
            budget=budget,
        )
        input_count = sum(len(row["candidate_item_ids"]) for row in cf_rows + sasrec_rows)
        self.current_stage = "fusion_and_dedup"
        fused = profiler.call(
            "fusion_and_dedup",
            lambda: adapters.source_aware_fusion(cf_rows, sasrec_rows, bundle.split_map, self.config),
            sample_count=sample_count,
            input_candidate_count=input_count,
            output_candidate_count=0,
            repeat=repeat,
            warmup=warmup,
            budget=budget,
        )
        profiler.records[-1]["output_candidate_count"] = _candidate_total(fused)
        return fused

    def _execute_budget(
        self,
        bundle: InputBundle,
        resources: adapters.RankerResources,
        budget: BudgetValue,
        profiler: StageProfiler,
        *,
        repeat: int,
        warmup: bool,
    ) -> tuple[list[Any], list[RankingSample], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        label = budget_label(budget)
        sample_count = len(bundle.selected_indices)
        e2e_start = time.perf_counter_ns()
        fused = self._prepare_pre_budget(bundle, profiler, repeat=repeat, warmup=warmup, budget=label)
        self.current_stage = "candidate_budget"
        post_budget = profiler.call(
            "candidate_budget",
            lambda: apply_budget(fused, budget),
            sample_count=sample_count,
            input_candidate_count=_candidate_total(fused),
            output_candidate_count=0,
            repeat=repeat,
            warmup=warmup,
            budget=label,
        )
        profiler.records[-1]["output_candidate_count"] = _candidate_total(post_budget)
        self.current_stage = "history_rerank"
        rankings, per_sample_latency = profiler.call(
            "history_rerank",
            lambda: adapters.rerank_post_budget(
                post_budget,
                resources,
                int(self.config.data["budget"]["top_n"]),
                repeat,
                warmup,
            ),
            sample_count=sample_count,
            input_candidate_count=_candidate_total(post_budget),
            output_candidate_count=sample_count * int(self.config.data["budget"]["top_n"]),
            repeat=repeat,
            warmup=warmup,
            budget=label,
        )
        self.current_stage = "evaluation"
        quality, per_sample_quality = profiler.call(
            "evaluation",
            lambda: evaluate_rankings(rankings),
            sample_count=sample_count,
            input_candidate_count=sum(len(row.ranked_item_ids) for row in rankings),
            output_candidate_count=sum(len(row.top_item_ids) for row in rankings),
            repeat=repeat,
            warmup=warmup,
            budget=label,
        )
        elapsed_ms = (time.perf_counter_ns() - e2e_start) / 1_000_000.0
        profiler.record_elapsed(
            "downstream_end_to_end",
            elapsed_ms,
            sample_count=sample_count,
            input_candidate_count=sum(row.cf_candidate_count + row.sasrec_candidate_count for row in fused),
            output_candidate_count=sum(len(row.top_item_ids) for row in rankings),
            repeat=repeat,
            warmup=warmup,
            budget=label,
        )
        return post_budget, rankings, quality, per_sample_quality, per_sample_latency

    def run(
        self,
        *,
        run_mode: str,
        run_id: str | None,
        max_samples: int | None,
        command: str,
    ) -> Path:
        if run_mode not in {"cardinality_audit", "full_pipeline"}:
            raise ValueError(run_mode)
        formal_guard = verify_formal_execution_guard(self.config) if self.config.is_formal else None
        plan = build_command_plan(self.config, run_mode)
        run = ArtifactRun(self.config, run_id, run_mode, command, plan, formal_guard=formal_guard)
        profiler = StageProfiler(bool(self.config.data["profiling"]["synchronize_cuda_if_used"]))
        sample_count: int | None = None
        try:
            self.current_stage = "input_load"
            load_started = time.perf_counter_ns()
            bundle, input_meta = adapters.load_input_bundle(self.config, max_samples=max_samples)
            sample_count = len(bundle.selected_indices)
            if formal_guard is not None:
                expected_count = int(formal_guard["expected_sample_count"])
                if sample_count != expected_count:
                    raise ConfigError(
                        f"formal sample count guard failed: selected {sample_count}, expected {expected_count}"
                    )
                if input_meta["split_evidence"].get("selected_split") != "valid_select":
                    raise ConfigError("formal selected split guard failed: expected valid_select")
            resources = adapters.load_ranker_resources(self.config) if run_mode == "full_pipeline" else None
            profiler.record_elapsed(
                "input_load",
                (time.perf_counter_ns() - load_started) / 1_000_000.0,
                sample_count=sample_count,
                measurement_kind="cold_start",
            )
            run.append_log(f"input_load selected_samples={sample_count}")

            self.current_stage = "candidate_cardinality_gate"
            pre_budget = self._prepare_pre_budget(bundle, profiler, repeat=0, warmup=False, budget="audit")
            try:
                audit = audit_cardinality(
                    pre_budget,
                    self.config.budgets,
                    int(self.config.data["budget"]["top_n"]),
                    bool(self.config.data["budget"]["fail_if_inactive"]),
                    float(self.config.data["budget"]["warn_affected_fraction_below"]),
                )
            except CardinalityGateError as exc:
                run.write_csv("cardinality_per_sample.csv", exc.audit.per_sample)
                run.write_json("cardinality_summary.json", exc.audit.summary)
                raise
            run.write_csv("cardinality_per_sample.csv", audit.per_sample)
            run.write_json("cardinality_summary.json", audit.summary)
            run.append_log(f"candidate_cardinality_gate status={audit.summary['gate']['status']}")
            if run_mode == "cardinality_audit":
                run.complete(
                    sample_count,
                    {
                        "cardinality_gate": audit.summary["gate"]["status"],
                        "split_evidence": input_meta["split_evidence"],
                    },
                )
                return run.run_dir

            assert resources is not None
            warmup_samples = int(self.config.data["profiling"]["warmup_samples"])
            repeats = int(self.config.data["profiling"]["repeats"])
            if warmup_samples:
                warm_bundle = bundle.subset(min(warmup_samples, sample_count))
                for budget in self.config.budgets:
                    self._execute_budget(warm_bundle, resources, budget, profiler, repeat=-1, warmup=True)
            cuda_peak_reset = reset_cuda_peak_if_available()

            candidate_rows: list[dict[str, Any]] = []
            ranking_rows: list[dict[str, Any]] = []
            quality_rows: list[dict[str, Any]] = []
            latency_rows: list[dict[str, Any]] = []
            quality_summary: dict[str, Any] = {"scope": PROFILING_SCOPE, "budgets": {}}
            for repeat in range(repeats):
                for budget in self.config.budgets:
                    post, rankings, quality, per_quality, per_latency = self._execute_budget(
                        bundle, resources, budget, profiler, repeat=repeat, warmup=False
                    )
                    latency_rows.extend(per_latency)
                    if repeat != 0:
                        continue
                    label = budget_label(budget)
                    quality_summary["budgets"][label] = quality
                    quality_rows.extend(per_quality)
                    for sample in post:
                        candidate_rows.append(
                            {
                                "sample_id": sample.sample_id,
                                "budget": str(sample.budget),
                                "budget_position": sample.stage,
                                "target_item_id": sample.target_item_id,
                                "candidate_count": len(sample.ordered_item_ids),
                                "candidate_item_ids": list(sample.ordered_item_ids),
                            }
                        )
                    for row in rankings:
                        ranking_rows.append(
                            {
                                "sample_id": row.sample_id,
                                "budget": str(row.budget),
                                "target_item_id": row.target_item_id,
                                "candidate_count": len(row.candidate_item_ids),
                                "ranked_item_ids": list(row.ranked_item_ids),
                                "top20_item_ids": list(row.top_item_ids),
                            }
                        )
            run.write_jsonl("candidates.jsonl", candidate_rows)
            run.write_jsonl("rankings.jsonl", ranking_rows)
            run.write_csv("per_sample_quality.csv", quality_rows)
            run.write_csv("per_sample_latency.csv", latency_rows)
            run.write_json("quality_summary.json", quality_summary)
            run.write_json("latency_summary.json", profiler.summaries())
            run.write_json("resource_summary.json", profiler.resource_summary())
            run.complete(
                sample_count,
                {
                    "cardinality_gate": audit.summary["gate"]["status"],
                    "split_evidence": input_meta["split_evidence"],
                    "profiling_records": len(profiler.records),
                    "cuda_peak_reset": cuda_peak_reset,
                },
            )
            return run.run_dir
        except Exception as exc:
            run.fail(self.current_stage, exc, sample_count)
            raise


def command_text(argv: list[str]) -> str:
    return shlex.join(argv)
