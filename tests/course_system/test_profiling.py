from __future__ import annotations

import sys
from types import SimpleNamespace

from minionerec_system.profiling import PROFILING_LABEL, PROFILING_SCOPE, StageProfiler, cuda_snapshot


def test_cuda_unavailable_is_not_applicable(monkeypatch) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    snapshot = cuda_snapshot()
    assert snapshot["cuda_measurement"] == "not_applicable"
    assert snapshot["cuda_allocated_bytes"] == "not_applicable"
    assert snapshot["cuda_peak_allocated_bytes"] == "not_applicable"


def test_downstream_scope_and_summary_names_are_exact() -> None:
    profiler = StageProfiler()
    profiler.call("candidate_budget", lambda: None, sample_count=2, input_candidate_count=180, output_candidate_count=40)
    summary = profiler.summaries()
    assert summary["scope"] == PROFILING_SCOPE == "frozen_prediction_downstream"
    assert summary["latency_label"] == PROFILING_LABEL == "frozen-prediction downstream pipeline latency"
