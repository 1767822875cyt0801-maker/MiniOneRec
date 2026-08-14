from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "results/course_system/valid/Industrial_and_Scientific/course-valid-cardinality-v1/cardinality_per_sample.csv"


def test_k90_is_active_on_read_only_v1_distribution() -> None:
    if not V1.is_file():
        pytest.skip("local read-only v1 cardinality artifact is not present")
    with V1.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = [int(row["post_dedup_count"]) for row in rows]
    effective = [min(count, 90) for count in counts]
    removed = [before - after for before, after in zip(counts, effective)]
    assert len(rows) == 1360
    assert len({row["sample_id"] for row in rows}) == 1360
    assert sum(amount > 0 for amount in removed) == 1351
    assert sum(effective) == 122399
    assert math.isclose(sum(removed) / len(rows), 5.815441176470588)
    assert effective != [min(count, 75) for count in counts]
    assert effective != counts
