# S6-5 Frozen Ranker Validation

Verdict: **GO_LIGHTWEIGHT_RANKER_VALIDATION**

S6-5 evaluates whether the frozen S5 history-aware `source_independent_projection` ranker can convert the low-cost CF + direct-SASRec K20 candidate pool into top-20 ranking gains without retraining and without using direct-SASRec raw score features.

No test data, Qwen generation, direct SASRec inference, ranker training, union regeneration, or parameter tuning was performed.

Machine-readable report:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/s6_5_frozen_ranker_validation_report.json
```

Run id:

```text
frozen_ranker_v1
```

## Frozen Ranker Compatibility

Compatibility verdict: **GO_FROZEN_RANKER_COMPATIBLE**

Frozen model:

```text
results/stage7_validation_protocol/valid/Industrial_and_Scientific/p2_history_ranker/model.json
```

SHA256:

```text
6a3bca7cc714d295572ce4a3953d60719a1c72757895835554a83d747dca65ce
```

Frozen protocol:

| Field | Value |
|---|---|
| model type | `history_aware_linear_pairwise_logistic` |
| compatibility mode | `source_independent_projection` |
| candidate policy | `source_aware_rrf_lam0.75_bonus0.01` |
| `lambda_sasrec` | 0.75 |
| `source_bonus` | 0.01 |
| feature dimension | 26 |
| tie breaker | score descending, original rank ascending, item id ascending |

The model feature order exactly matches the frozen S5 release config.

## Projection Mapping

Projection manifest:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/frozen_ranker/frozen_ranker_v1/projection_manifest.json
```

The projection uses only the frozen S5 source-independent feature surface:

- CF/direct source membership is preserved in provenance.
- CF rank and direct auxiliary rank are available to construct source-independent best-rank and fusion-rank features.
- Source identity features are zeroed after normalization.
- Overlap provenance is preserved.
- History, popularity, bucket, expansion, and heuristic features use the frozen S5 construction.

Explicitly excluded from the frozen ranker:

- `sasrec_direct_score`
- `sasrec_direct_zscore`
- `sasrec_direct_top1_margin`
- `sasrec_direct_score_minus_topk_mean`

These features were not present during S5 ranker training and are reserved for a possible S6 lightweight ranker.

## Split Metrics

### valid_fit

| Method | HR@20 | NDCG@20 | MRR | hits@20 | CF preserved | CF lost | direct-only recovered | avg pool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CF-only | 0.107895 | 0.071681 | 0.061706 | 287 | 287 | 0 | 0 | 67.24 |
| direct order | 0.134586 | 0.104068 | 0.095333 | 358 | 196 | 91 | 162 | 67.24 |
| fixed RRF | 0.114286 | 0.082575 | 0.076096 | 304 | 282 | 5 | 22 | 67.24 |
| projected ranker | 0.149248 | 0.093252 | 0.079197 | 397 | 279 | 8 | 102 | 67.24 |

Ranking gate:

- HR@20 uplift retention: 2.292, pass.
- NDCG@20 uplift retention: 3.777, pass.
- CF hit preservation: 279 / 287 = 97.21%, pass.

### valid_select

| Method | HR@20 | NDCG@20 | MRR | hits@20 | CF preserved | CF lost | direct-only recovered | avg pool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CF-only | 0.098650 | 0.061860 | 0.055043 | 95 | 95 | 0 | 0 | 67.39 |
| direct order | 0.127726 | 0.094707 | 0.085376 | 123 | 65 | 30 | 58 | 67.39 |
| fixed RRF | 0.100727 | 0.073357 | 0.069310 | 97 | 93 | 2 | 4 | 67.39 |
| projected ranker | 0.130841 | 0.082337 | 0.068327 | 126 | 91 | 4 | 34 | 67.39 |

Ranking gate:

- HR@20 uplift retention: 3.875, pass.
- NDCG@20 uplift retention: 9.335, pass.
- CF hit preservation: 91 / 95 = 95.79%, **fails** the 97% gate.

### valid_gate

`valid_gate` was inspected only after fit/select evaluation and did not revise parameters.

| Method | HR@20 | NDCG@20 | MRR | hits@20 | CF preserved | CF lost | direct-only recovered | avg pool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CF-only | 0.090209 | 0.060987 | 0.056563 | 82 | 82 | 0 | 0 | 67.51 |
| direct order | 0.114411 | 0.089078 | 0.081472 | 104 | 56 | 26 | 48 | 67.51 |
| fixed RRF | 0.094609 | 0.069548 | 0.066250 | 86 | 80 | 2 | 6 | 67.51 |
| projected ranker | 0.130913 | 0.080346 | 0.071662 | 119 | 81 | 1 | 35 | 67.51 |

Confirmation:

- HR@20 uplift retention: 3.364, pass.
- NDCG@20 uplift retention: 5.041, pass.
- CF hit preservation: 81 / 82 = 98.78%, pass.

## Aggregate Validation Metrics

| Method | HR@1 | HR@5 | HR@10 | HR@20 | HR@50 | NDCG@20 | MRR | hits@20 | target pool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CF-only | 0.046337 | 0.067961 | 0.080980 | 0.102383 | 0.138791 | 0.067449 | 0.058812 | 464 | 629 |
| direct order | 0.078994 | 0.102383 | 0.114960 | 0.129082 | 0.129082 | 0.099072 | 0.090437 | 585 | 585 |
| fixed RRF | 0.058473 | 0.082524 | 0.091792 | 0.107458 | 0.173213 | 0.078003 | 0.072212 | 487 | 865 |
| projected ranker | 0.058473 | 0.090026 | 0.107899 | 0.141659 | 0.182921 | 0.088344 | 0.075089 | 642 | 865 |

Aggregate decomposition for projected ranker:

| Quantity | Value |
|---|---:|
| union target-in-pool | 865 / 4532 |
| CF hits@20 | 464 |
| projected hits@20 | 642 |
| CF hits preserved@20 | 451 |
| CF hits lost@20 | 13 |
| direct-only targets available@20 | 268 |
| direct-only targets recovered@20 | 171 |
| direct-only recovery rate@20 | 63.81% |
| average candidate pool size | 67.33 |

## Fixed RRF Versus Projected Ranker

The frozen projected ranker materially improves over fixed RRF:

| Metric | fixed RRF | projected ranker | change |
|---|---:|---:|---:|
| HR@20 | 0.107458 | 0.141659 | +0.034201 |
| NDCG@20 | 0.078003 | 0.088344 | +0.010341 |
| hits@20 | 487 | 642 | +155 |
| direct-only recovered@20 | 32 | 171 | +139 |
| CF hits lost@20 | 9 | 13 | +4 |

The conversion is strong, but the `valid_select` CF-preservation gate misses the frozen 97% threshold.

## Dual-Qwen Uplift-Retention Comparison

The reference is the frozen S5 dual-Qwen validation path on the same row splits.

Aggregate S5 reference:

| Method | HR@20 | NDCG@20 | hits@20 |
|---|---:|---:|---:|
| S5 CF-only | 0.102383 | 0.067449 | 464 |
| S5 fixed RRF | 0.106134 | 0.066942 | 481 |
| S5 projected ranker | 0.117167 | 0.072037 | 531 |

S6 projected ranker exceeds S5 projected validation uplift in HR@20 and NDCG@20, but it does not satisfy the CF-preservation gate on `valid_select`.

## Cost Status

Frozen direct-SASRec runtime evidence:

| Quantity | Value |
|---|---:|
| direct model inference | 24.967018 seconds / 4532 samples |
| direct formal wall time | 34.233973 seconds |
| peak CUDA allocated | 13,118,464 bytes |

Comparable validation-side SASRec-SID Qwen runtime evidence was not found in frozen validation artifacts. The numerical cost-ratio gate remains pending and does not block this ranking verdict.

## Decision

Final verdict: **GO_LIGHTWEIGHT_RANKER_VALIDATION**

Reason:

- Candidate complementarity is strong.
- The frozen projected ranker converts many direct-only candidates into top-20 hits.
- However, frozen projected ranker CF hit preservation on `valid_select` is 95.79%, below the required 97%.
- Projection compatibility is not the blocker; the frozen S5 ranker lacks direct-SASRec score features and was not trained for this source.

Recommended next gate:

Train or validate a lightweight S6 ranker on validation splits only, with an explicit schema that may include direct-SASRec score features. Do not use test data for tuning.

## Outputs

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/frozen_ranker/frozen_ranker_v1/projection_manifest.json
results/s6_cost_aware_aux/Industrial_and_Scientific/frozen_ranker/frozen_ranker_v1/execution_log.json
results/s6_cost_aware_aux/Industrial_and_Scientific/frozen_ranker/frozen_ranker_v1/exit_code.txt
results/s6_cost_aware_aux/Industrial_and_Scientific/frozen_ranker/frozen_ranker_v1/<split>/ranked_candidates.jsonl
results/s6_cost_aware_aux/Industrial_and_Scientific/frozen_ranker/frozen_ranker_v1/<split>/metrics.json
results/s6_cost_aware_aux/Industrial_and_Scientific/s6_5_frozen_ranker_validation_report.json
```

## Commands

Dry-run:

```bash
cd /root/autodl-tmp/projects/MiniOneRec
python3 scripts/s6_frozen_ranker_validation.py --dry-run
```

Formal validation:

```bash
cd /root/autodl-tmp/projects/MiniOneRec
python3 scripts/s6_frozen_ranker_validation.py
```

## Rollback

Remove only S6-5 derived outputs:

```bash
rm -rf results/s6_cost_aware_aux/Industrial_and_Scientific/frozen_ranker/frozen_ranker_v1
rm -f results/s6_cost_aware_aux/Industrial_and_Scientific/s6_5_frozen_ranker_validation_report.json
```

Do not modify frozen S4/S5/S6 input artifacts.
