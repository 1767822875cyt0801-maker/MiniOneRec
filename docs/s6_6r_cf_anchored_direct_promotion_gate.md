# S6-6R CF-Anchored Direct Promotion Gate

## Verdict

`REVISE_STAGE_CLOSEOUT`

No predeclared CF-anchored direct-promotion policy passed the frozen `valid_fit` and `valid_select` gates. `valid_gate` was not opened. S6-5 frozen projected ranker remains the strongest validated S6 ranking result.

## Boundary

This validation is CPU-only and validation-only. It does not read test data, rerun CF Qwen, rerun SASRec-SID Qwen, rerun direct SASRec, regenerate `union_v1`, reconsider K, modify S5/S6-5 artifacts, or use final-test evidence.

## Metric Reconciliation

S6-6R freezes the following definitions:

- `CF-candidate-pool-exclusive target`: target exists in the direct pool but not in the CF candidate pool.
- `Non-CF-top20 target`: target is not hit by the CF-only top-20 baseline but is available in the union/direct top-20 evidence.
- `Direct-source-exclusive candidate`: candidate has `from_sasrec_direct=true` and `from_cf=false`.
- `Promotion-recoverable target`: non-CF-top20 target that can enter top-20 by promoting a direct-source-exclusive candidate.
- `CF-hit preservation`: a CF-only top-20 hit remains a hit after promotion.

The frozen full-validation aggregate reconciles the previously confusing counts:

| Definition | Count |
|---|---:|
| `union_target_in_pool_count - cf_target_in_pool_count` | 236 |
| `direct_only_targets_available_at20` | 268 |

Raw preselection decomposition was recomputed only on `valid_fit` and `valid_select` to avoid opening `valid_gate` before a policy passed:

| Split | CF-pool-exclusive target | non-CF-top20 direct-top20 target | non-CF-top20 union target |
|---|---:|---:|---:|
| valid_fit | 140 | 162 | 239 |
| valid_select | 54 | 58 | 87 |

## Input Integrity

Inputs are immutable validation artifacts:

- `union_v1` K20 candidates.
- CF split views and CF-only ranking.
- direct score sidecars embedded in union candidate details.
- S6-5 frozen projected ranker report as comparison baseline.
- S6-6 residual-ranker report as negative evidence.
- S6 validation split manifest.

The run wrote a machine-readable input manifest:

`results/s6_cost_aware_aux/Industrial_and_Scientific/direct_promotion/promotion_v1/train_manifest.json`

It records protocol, split, union, frozen-ranker, S6-6 negative-evidence, feature, normalization, and training-pair provenance hashes. `test_read=false`.

## Ranking Semantics

For each sample:

1. Start from the frozen CF-only candidate order.
2. Preserve the internal order of retained CF candidates.
3. Consider only direct-source-exclusive candidates for promotion.
4. Score direct candidates against the current CF top-20 tail boundary.
5. Promote only if the probability threshold and margin criterion pass.
6. Replace only CF tail candidates.
7. Emit retained CF candidates first, then promoted direct candidates sorted by probability, margin, direct rank, and item id.
8. Keep overlap-source candidates at their CF-anchored position.

This is not a global reranking of the union pool.

## Model and Training

Model family: pairwise logistic direct-promotion classifier.

Training split: `valid_fit` only.

Features include direct rank/score sidecars, frozen projected score, source-independent history/popularity signals, CF boundary rank/score signals, and candidate-versus-boundary differences. Target equality is used only inside valid_fit loss construction and validation metrics.

Training summary:

| Field | Value |
|---|---:|
| positive promotion queries | 140 |
| negative protection queries | 287 |
| pair count | 2835 |
| positive pairs | 1400 |
| negative pairs | 1435 |
| epochs | 120 |
| L2 | 0.001 |

## Frozen Policy Grid

Thresholds were calibrated from valid_fit positive-pair probabilities only. The final predeclared grid had 12 policies:

`max_promotions in {1,2,3}` x four fit-only thresholds.

Grid artifact:

`results/s6_cost_aware_aux/Industrial_and_Scientific/direct_promotion/promotion_v1/predeclared_policy_grid.json`

## Fit/Select Results

No policy passed all gates.

Best valid-select HR policy:

| Config | Split | HR@20 | NDCG@20 | CF preserved/lost | CF preservation | Direct recovered | Avg promotions |
|---|---|---:|---:|---:|---:|---:|---:|
| `mp3_thr0.800224` | valid_fit | 0.129699 | 0.076776 | 271 / 16 | 94.43% | 74 | 2.940 |
| `mp3_thr0.800224` | valid_select | 0.119418 | 0.066684 | 90 / 5 | 94.74% | 25 | 2.922 |

Best CF-preserving policy:

| Config | Split | HR@20 | NDCG@20 | CF preserved/lost | CF preservation | Direct recovered | Avg promotions |
|---|---|---:|---:|---:|---:|---:|---:|
| `mp2_thr0.958302` | valid_fit | 0.125188 | 0.075633 | 286 / 1 | 99.65% | 47 | 0.347 |
| `mp2_thr0.958302` | valid_select | 0.115265 | 0.065666 | 95 / 0 | 100.00% | 16 | 0.363 |

The first policy recovers more direct-only targets but violates the 97% CF-preservation gate. The second preserves CF hits but fails the HR/NDCG/direct-recovery retention gates versus S6-5.

## Comparison

S6-5 frozen projected ranker:

| Split | HR@20 | NDCG@20 | CF preserved/lost | Direct recovered |
|---|---:|---:|---:|---:|
| valid_fit | 0.149248 | 0.093252 | 279 / 8 | 102 |
| valid_select | 0.130841 | 0.082337 | 91 / 4 | 34 |

S6-6 aggressive residual `alpha1_l20.001_clip0.5_cfq0`:

| Split | HR@20 | NDCG@20 | CF preserved/lost | Direct recovered |
|---|---:|---:|---:|---:|
| valid_fit | 0.155263 | 0.095161 | 279 / 8 | 112 |
| valid_select | 0.137072 | 0.084272 | 90 / 5 | 37 |

S6-6 preservation residual `alpha0.1_l20.001_clip0.25_cfq18`:

| Split | HR@20 | NDCG@20 | CF preserved/lost | Direct recovered |
|---|---:|---:|---:|---:|
| valid_fit | 0.133459 | 0.085736 | 282 / 5 | 66 |
| valid_select | 0.116303 | 0.076331 | 93 / 2 | 19 |

S6-6R confirms the same mechanism-level difficulty: high direct promotion recovers auxiliary targets but loses too many CF hits; conservative promotion protects CF but eliminates too much direct gain.

## Runtime

| Field | Value |
|---|---:|
| total wall seconds | 18.052960 |
| feature generation wall seconds | 10.617129 |
| training wall seconds | 0.029918 |
| inference wall seconds | 6.037206 |
| samples/sec, fit+select | 600.112 |
| model size bytes | 0 |

No selected model was frozen, so model size is zero.

## Artifacts

- `configs/s6_cost_aware_aux/direct_promotion_protocol.json`
- `scripts/s6_direct_promotion_gate_validation.py`
- `tests/test_s6_direct_promotion_gate_validation.py`
- `results/s6_cost_aware_aux/Industrial_and_Scientific/s6_6r_direct_promotion_gate_report.json`
- `results/s6_cost_aware_aux/Industrial_and_Scientific/direct_promotion/promotion_v1/predeclared_policy_grid.json`
- `results/s6_cost_aware_aux/Industrial_and_Scientific/direct_promotion/promotion_v1/train_manifest.json`
- `results/s6_cost_aware_aux/Industrial_and_Scientific/direct_promotion/promotion_v1/exit_code.txt`

## Closeout Recommendation

Proceed to stage closeout with `REVISE_STAGE_CLOSEOUT`. Do not expand the lightweight ranker search. Do not open test. Keep S6-5 frozen projected ranker as the strongest validated S6 ranking result.
