# S6-6 Lightweight CF-Preserving Ranker Validation

Verdict: **REVISE_LIGHTWEIGHT_RANKER**

S6-6 trained and evaluated a small validation-only residual ranker for the CF + direct-SASRec K20 union. The goal was to improve CF top-20 preservation over the frozen S5 projected ranker while retaining at least 95% of its HR@20/NDCG@20 uplift and at least 90% of its direct recovery.

No test data, Qwen generation, direct SASRec inference, union regeneration, K re-selection, or S5 frozen-ranker modification was performed. `valid_gate` was not inspected because no predeclared configuration passed the `valid_fit`/`valid_select` gates.

Machine-readable report:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/s6_6_lightweight_ranker_validation_report.json
```

## Metric-Definition Reconciliation

S6-4 and S6-5 used related but different denominator concepts:

| Name | Formula |
|---|---|
| S6-4 direct-only targets beyond CF pool | target absent from the CF candidate pool but present in direct K20 pool |
| S6-5 direct-only recovered denominator | target missed by CF top-20 baseline but present in direct top-20 ordering |
| non-CF-top20 recoverable targets | target not hit by CF top20 but available somewhere in the union pool |
| direct-source-only pool targets | target in direct K20 and not in CF candidate pool |

These names are non-overlapping in the S6-6 report. They are intentionally not collapsed into one metric.

## Input Integrity

Inputs:

- K20 `union_v1` candidates for `valid_fit` and `valid_select`;
- S6-5 frozen-ranker validation report;
- S6-4 union validation report;
- frozen CF split views;
- direct-SASRec score sidecars embedded in union provenance;
- S5 frozen model as a read-only baseline.

Input hashes are recorded in:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/lightweight_ranker/lightweight_v1/train_manifest.json
```

The training manifest records feature hashes for `valid_fit` and `valid_select` only. Because no configuration passed fit/select, `valid_gate` was not opened.

## Feature Contract

Feature manifest:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/lightweight_ranker/lightweight_v1/train_manifest.json
```

Feature families:

- frozen-ranker score/rank anchors;
- CF/direct source membership;
- overlap;
- CF and direct ranks;
- reciprocal ranks;
- CF top-1/5/10/20 indicators;
- direct-SASRec score sidecar features;
- S5-compatible history, popularity, bucket, expansion, and heuristic features.

Direct features used:

- `sasrec_direct_score`
- `sasrec_direct_zscore`
- `sasrec_direct_top1_margin`
- `sasrec_direct_score_minus_topk_mean`

Forbidden feature audit:

- target item identity is not a feature;
- candidate label is not a feature;
- post-ranking outcome is not a feature;
- valid_select and valid_gate labels are not used during fitting;
- test-derived statistics are not used.

Normalization:

```text
mean/std fitted on valid_fit candidates only
```

Missing values:

```text
missing ranks -> candidate_count + 1
missing direct scores -> 0
```

## Model Family

Model:

```text
linear_pairwise_logistic_residual
```

Score:

```text
final_score = frozen_ranker_score + alpha * clipped_residual_score
```

Preservation mechanism:

```text
deterministic_cf_head_quota
```

The quota moves the highest-scoring CF top-20 candidates into the head of the ranking up to the configured quota, but does not lock all 20 CF candidates.

Predeclared grid:

- `alpha`: `{0.1, 0.25, 0.5, 1.0}`
- `l2`: `{0.001, 0.01}`
- `residual_clip`: `{0.25, 0.5}`
- `cf_head_quota`: `{0, 10, 15, 18}`

Total configurations:

```text
64
```

All configurations were written before selection to:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/lightweight_ranker/lightweight_v1/predeclared_config_grid.json
```

## Training Protocol

Training split:

```text
valid_fit only
```

Labels:

```text
1 iff candidate_item_id == target_item_id, else 0
```

Training objective:

```text
pairwise logistic loss over positive-negative candidate pairs within the same query/sample
```

No cross-query pairs are formed.

Training summary:

| L2 | Pairs | Epochs | LR | Wall seconds |
|---:|---:|---:|---:|---:|
| 0.001 | 350790 | 80 | 0.05 | recorded in report |
| 0.01 | 350790 | 80 | 0.05 | recorded in report |

## Fit/Select Results

No predeclared configuration passed all fit/select gates.

Best retention/highest HR configuration:

```text
alpha1_l20.001_clip0.5_cfq0
```

| Split | HR@20 | NDCG@20 | hits@20 | CF preservation | CF lost | direct recovered | HR retention | NDCG retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valid_fit | 0.155263 | 0.095161 | 413 | 97.21% | 8 | 112 | 1.145 | 1.089 |
| valid_select | 0.137072 | 0.084272 | 132 | 94.74% | 5 | 37 | 1.194 | 1.095 |

Failure reason:

```text
valid_select CF preservation < 97%
```

Best CF-preserving configuration:

```text
alpha0.1_l20.001_clip0.25_cfq18
```

| Split | HR@20 | NDCG@20 | hits@20 | CF preservation | CF lost | direct recovered | HR retention | NDCG retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valid_fit | 0.133459 | 0.085736 | 355 | 98.26% | 5 | 66 | 0.618 | 0.652 |
| valid_select | 0.116303 | 0.076331 | 112 | 97.89% | 2 | 19 | 0.548 | 0.707 |

Failure reason:

```text
HR/NDCG uplift retention and direct-recovery retention below required gates
```

## Valid Gate

`valid_gate` was not inspected. No configuration passed the fit/select gates, so the protocol returned `REVISE_LIGHTWEIGHT_RANKER` before opening the one-time gate split.

## Runtime

| Quantity | Value |
|---|---:|
| total S6-6 wall seconds | 87.801 |
| training wall seconds | 12.292 |
| selected model size | 0 bytes, no model selected |
| direct-SASRec model inference | 24.967018 s / 4532 samples |
| direct-SASRec formal wall | 34.233973 s |
| direct-SASRec peak CUDA allocated | 13,118,464 bytes |

The direct-versus-Qwen cost ratio remains pending because comparable validation-side SASRec-SID Qwen runtime evidence was not available.

## Decision

Final verdict:

```text
REVISE_LIGHTWEIGHT_RANKER
```

Interpretation:

- Direct-SASRec candidate complementarity remains strong.
- A lightweight residual ranker can exceed the frozen projected ranker in HR/NDCG, but the high-gain configurations lose too many CF hits on `valid_select`.
- Strong CF-head quota configurations meet preservation but lose too much ranking uplift and direct recovery.
- The current predeclared model family and preservation mechanism do not satisfy all gates simultaneously.

Recommended next gate:

Revise the lightweight ranker design on validation only. Candidate directions include a softer CF-preservation regularizer or pairwise constraints that specifically protect CF top-20 positives without forcing a large deterministic head quota. Do not use test data.

## Outputs

```text
configs/s6_cost_aware_aux/lightweight_ranker_protocol.json
scripts/s6_lightweight_ranker_validation.py
tests/test_s6_lightweight_ranker_validation.py
results/s6_cost_aware_aux/Industrial_and_Scientific/lightweight_ranker/lightweight_v1/predeclared_config_grid.json
results/s6_cost_aware_aux/Industrial_and_Scientific/lightweight_ranker/lightweight_v1/train_manifest.json
results/s6_cost_aware_aux/Industrial_and_Scientific/lightweight_ranker/lightweight_v1/exit_code.txt
results/s6_cost_aware_aux/Industrial_and_Scientific/s6_6_lightweight_ranker_validation_report.json
docs/s6_6_lightweight_cf_preserving_ranker.md
```

## Commands

Dry-run:

```bash
cd /root/autodl-tmp/projects/MiniOneRec
python3 scripts/s6_lightweight_ranker_validation.py --dry-run
```

Formal validation:

```bash
cd /root/autodl-tmp/projects/MiniOneRec
python3 scripts/s6_lightweight_ranker_validation.py
```

## Rollback

Remove only S6-6 derived outputs:

```bash
python3 - <<'PY'
from pathlib import Path
import shutil
root = Path("results/s6_cost_aware_aux/Industrial_and_Scientific")
for path in [root / "lightweight_ranker/lightweight_v1", root / "s6_6_lightweight_ranker_validation_report.json"]:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
PY
```
