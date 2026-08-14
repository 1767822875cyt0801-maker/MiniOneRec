"""Profiling primitives for frozen-prediction downstream pipeline latency."""

from __future__ import annotations

import math
import resource
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar


T = TypeVar("T")
PROFILING_SCOPE = "frozen_prediction_downstream"
PROFILING_LABEL = "frozen-prediction downstream pipeline latency"
STAGES = (
    "input_load",
    "candidate_expansion",
    "fusion_and_dedup",
    "candidate_budget",
    "history_rerank",
    "evaluation",
    "downstream_end_to_end",
)


def cpu_rss_bytes() -> int:
    # Linux ru_maxrss is KiB; the course AutoDL target is Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def cuda_snapshot() -> dict[str, int | str]:
    try:
        import torch
    except Exception:
        return {
            "cuda_measurement": "not_applicable",
            "cuda_allocated_bytes": "not_applicable",
            "cuda_reserved_bytes": "not_applicable",
            "cuda_peak_allocated_bytes": "not_applicable",
            "cuda_peak_reserved_bytes": "not_applicable",
        }
    if not torch.cuda.is_available():
        return {
            "cuda_measurement": "not_applicable",
            "cuda_allocated_bytes": "not_applicable",
            "cuda_reserved_bytes": "not_applicable",
            "cuda_peak_allocated_bytes": "not_applicable",
            "cuda_peak_reserved_bytes": "not_applicable",
        }
    return {
        "cuda_measurement": "available",
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def reset_cuda_peak_if_available() -> str:
    try:
        import torch
    except Exception:
        return "not_applicable"
    if not torch.cuda.is_available():
        return "not_applicable"
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return "reset"


def _sync_cuda(enabled: bool, cuda_used: bool) -> None:
    if not enabled or not cuda_used:
        return
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


@dataclass
class StageProfiler:
    synchronize_cuda_if_used: bool = True
    records: list[dict[str, Any]] = field(default_factory=list)

    def call(
        self,
        stage: str,
        function: Callable[[], T],
        *,
        sample_count: int,
        input_candidate_count: int = 0,
        output_candidate_count: int = 0,
        repeat: int = 0,
        warmup: bool = False,
        budget: str = "not_applicable",
        cuda_used: bool = False,
        measurement_kind: str = "steady_state",
    ) -> T:
        if stage not in STAGES:
            raise ValueError(f"Unknown profiling stage: {stage}")
        _sync_cuda(self.synchronize_cuda_if_used, cuda_used)
        started = time.perf_counter_ns()
        result = function()
        _sync_cuda(self.synchronize_cuda_if_used, cuda_used)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        self.record_elapsed(
            stage,
            elapsed_ms,
            sample_count=sample_count,
            input_candidate_count=input_candidate_count,
            output_candidate_count=output_candidate_count,
            repeat=repeat,
            warmup=warmup,
            budget=budget,
            measurement_kind=measurement_kind,
        )
        return result

    def record_elapsed(
        self,
        stage: str,
        elapsed_ms: float,
        *,
        sample_count: int,
        input_candidate_count: int = 0,
        output_candidate_count: int = 0,
        repeat: int = 0,
        warmup: bool = False,
        budget: str = "not_applicable",
        measurement_kind: str = "steady_state",
    ) -> None:
        cuda = cuda_snapshot()
        self.records.append(
            {
                "scope": PROFILING_SCOPE,
                "latency_label": PROFILING_LABEL,
                "stage": stage,
                "elapsed_ms": elapsed_ms,
                "sample_count": sample_count,
                "input_candidate_count": input_candidate_count,
                "output_candidate_count": output_candidate_count,
                "repeat": repeat,
                "warmup": warmup,
                "budget": budget,
                "measurement_kind": measurement_kind,
                "cpu_rss_bytes": cpu_rss_bytes(),
                **cuda,
            }
        )

    def summaries(self) -> dict[str, Any]:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in self.records:
            if row["warmup"]:
                continue
            grouped.setdefault((row["stage"], row["budget"], row["measurement_kind"]), []).append(row)
        result: dict[str, Any] = {}
        for (stage, budget, measurement_kind), rows in grouped.items():
            elapsed = [float(row["elapsed_ms"]) for row in rows]
            total_samples = sum(int(row["sample_count"]) for row in rows)
            total_candidates = sum(int(row["input_candidate_count"]) for row in rows)
            total_seconds = sum(elapsed) / 1000.0
            key = f"{stage}:{budget}:{measurement_kind}"
            result[key] = {
                "stage": stage,
                "budget": budget,
                "measurement_kind": measurement_kind,
                "mean": statistics.fmean(elapsed),
                "std": statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0,
                "p50": _percentile(elapsed, 0.50),
                "p95": _percentile(elapsed, 0.95),
                "p99": _percentile(elapsed, 0.99),
                "samples_per_second": 0.0 if total_seconds <= 0 else total_samples / total_seconds,
                "candidate_scores_per_second": 0.0 if total_seconds <= 0 else total_candidates / total_seconds,
                "measured_records": len(rows),
            }
        return {"scope": PROFILING_SCOPE, "latency_label": PROFILING_LABEL, "stages": result}

    def resource_summary(self) -> dict[str, Any]:
        cuda_rows = [row for row in self.records if row["cuda_measurement"] == "available"]
        return {
            "scope": PROFILING_SCOPE,
            "peak_cpu_rss": max((int(row["cpu_rss_bytes"]) for row in self.records), default=cpu_rss_bytes()),
            "cuda_measurement": "available" if cuda_rows else "not_applicable",
            "peak_cuda_allocated": max((int(row["cuda_peak_allocated_bytes"]) for row in cuda_rows), default="not_applicable"),
            "peak_cuda_reserved": max((int(row["cuda_peak_reserved_bytes"]) for row in cuda_rows), default="not_applicable"),
        }
