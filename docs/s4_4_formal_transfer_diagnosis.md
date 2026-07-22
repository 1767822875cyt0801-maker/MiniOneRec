# S4-4 Formal Transfer Diagnosis and Next-Stage Gate

## Scope

This audit uses the latest returned formal full-valid bundle:

`incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036`

No GPU job, training, evaluation, generation, SID reconstruction, lambda tuning, p3 expansion, Text-SID fusion, or test split evaluation was run. The analysis reads returned formal valid artifacts plus local train/valid SID metadata only.

## One-Line Conclusion

Formal artifacts are internally valid and isolated. SASRec-SID remains weaker as a single stream than CF-SID, but it contributes nonzero unique hits; the gate is **AUXILIARY_STREAM_GO**: use it only as an auxiliary candidate stream, not as a standalone winner.

## Formal Artifact and Parity Closeout

| stream | sid_version | complete | rows | split | test_read | invalid_sid | cand_count | hashes_ok | reusable_derived |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | cf_k512_dedup | True | 4532 | valid | False | 0 | 50/50.0/50 | True | True |
| treatment | sasrec_v3_k512_dedup | True | 4532 | valid | False | 0 | 50/50.0/50 | True | True |

Cross-stream isolation checks: `{"baseline_paths_contain_cf_only": true, "candidate_sha256_distinct": true, "checkpoint_sha256_distinct": true, "eval_csv_sha256_distinct": true, "item2sid_sha256_distinct": true, "prediction_sha256_distinct": true, "sid2items_sha256_distinct": true, "treatment_paths_contain_sasrec_only": true}`.

The artifact schema has no explicit `reusable` field; `reusable_derived=true` means complete formal full-valid artifacts, matching row counts, valid split, `test_read=false`, zero invalid SIDs, fixed 50-candidate count, and file SHA256 values matching the manifest.

## Formal Metric Comparison

| metric | CF | SASRec | abs_delta | rel_delta | ratio | CF_smoke_to_formal | SAS_smoke_to_formal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hr@1 | 0.046337 | 0.037732 | -0.008605 | -0.185714 | 0.814286 | 0.019859 | 0.032436 |
| hr@5 | 0.067961 | 0.060018 | -0.007944 | -0.116883 | 0.883117 | 0.023389 | 0.051192 |
| hr@10 | 0.080980 | 0.065093 | -0.015887 | -0.196185 | 0.803815 | 0.023389 | 0.055163 |
| hr@20 | 0.102383 | 0.073036 | -0.029347 | -0.286638 | 0.713362 | 0.028464 | 0.060680 |
| ndcg@1 | 0.046337 | 0.037732 | -0.008605 | -0.185714 | 0.814286 | n/a | n/a |
| ndcg@5 | 0.057889 | 0.050469 | -0.007420 | -0.128176 | 0.871824 | n/a | n/a |
| ndcg@10 | 0.062060 | 0.052092 | -0.009968 | -0.160617 | 0.839383 | 0.021611 | 0.044634 |
| ndcg@20 | 0.067449 | 0.054094 | -0.013355 | -0.198005 | 0.801995 | 0.022918 | 0.046055 |
| mrr | 0.058812 | 0.048849 | -0.009963 | -0.169408 | 0.830592 | 0.021641 | 0.041843 |
| target_in_pool_rate | 0.138791 | 0.087599 | -0.051192 | -0.368839 | 0.631161 | 0.036628 | 0.069064 |
| target_in_pool_count | 629 | 397 | -232 | -0.368839 | 0.631161 | n/a | n/a |
| avg_rank_if_hit_0_based | 11.996820 | 7.848866 | -4.147954 | -0.345754 | 0.654246 | n/a | n/a |
| hr20_given_target_in_pool | 0.737679 | 0.833753 | 0.096074 | 0.130239 | 1.130239 | n/a | n/a |

Interpretation:

- CF-SID has higher standalone HR/NDCG/MRR and higher candidate pool recall.
- SASRec-SID target-in-pool rate is `0.087599` versus CF `0.138791`.
- Conditional HR@20 given target-in-pool is `0.833753` for SASRec-SID versus `0.737679` for CF-SID, supporting the hypothesis that the dominant SASRec weakness is candidate coverage rather than final ordering after the target is already in the pool.

## Per-Sample Complementarity

| K | CF-only | SASRec-only | both | neither | union_HR | abs_lift_vs_CF | rel_lift_vs_CF | avg_overlap | avg_jaccard | avg_RBO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 72 | 33 | 138 | 4289 | 0.053619 | 0.007282 | 0.157143 | 0.201015 | 0.201015 | 0.201015 |
| 5 | 99 | 63 | 209 | 4161 | 0.081862 | 0.013901 | 0.204545 | 0.756178 | 0.090359 | 0.181714 |
| 10 | 147 | 75 | 220 | 4090 | 0.097529 | 0.016549 | 0.204360 | 1.234113 | 0.069289 | 0.163599 |
| 20 | 219 | 86 | 245 | 3982 | 0.121359 | 0.018976 | 0.185345 | 2.048985 | 0.055496 | 0.150179 |
| 50 | 349 | 117 | 280 | 3786 | 0.164607 | 0.025816 | 0.186010 | 4.237423 | 0.044771 | 0.143582 |

Full top-50 candidate list identical rate: `0.000000`. Full predicted SID list identical rate: `0.000000`.

Rank-difference distribution with missing top-50 imputed separately: `{"both_missing_top50": 3786, "cf_better_both_hit": 49, "cf_better_sasrec_missing": 349, "sasrec_better_both_hit": 84, "sasrec_better_cf_missing": 117, "tie_both_hit": 147}`.

Detailed per-sample rows are written to `results/s4_sasrec_sid_valid/Industrial_and_Scientific/s4_4_per_sample_complementarity.csv`.

## Final-Beam Prefix Coverage Diagnosis

| stream | d1_rate | d2_rate | d3_rate | full_rate | d1->d2 | d2->d3 | d3->full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.735216 | 0.477935 | 0.270962 | 0.138791 | 0.650060 | 0.566944 | 0.512215 |
| treatment | 0.397617 | 0.115843 | 0.087599 | 0.087599 | 0.291343 | 0.756190 | 1.000000 |

This is final-beam prefix coverage only. The returned predictions contain final beams, not decoding-time beam states, so these rates must not be interpreted as actual beam survival during constrained decoding.

Detailed prefix rows are written to `results/s4_sasrec_sid_valid/Industrial_and_Scientific/s4_4_prefix_coverage.csv`.

## SID Structural Diagnosis

Selected structural facts:

- CF SID length distribution: `{3: 1504, 4: 2182}`.
- SASRec SID length distribution: `{3: 3487, 4: 199}`.
- SASRec declared cold fallback item count: `39`; valid target rows among those items: `125`.
- SASRec max prefix bucket size: `39`; CF max prefix bucket size: `88`.

Selected slice metrics:

| family | slice | stream | n | HR@20 | NDCG@20 | pool_rate |
| --- | --- | --- | --- | --- | --- | --- |
| dedup_vs_non_dedup_item | dedup_len4 | baseline | 2797 | 0.148373 | 0.100149 | 0.200572 |
| dedup_vs_non_dedup_item | dedup_len4 | treatment | 361 | 0.157895 | 0.152420 | 0.193906 |
| dedup_vs_non_dedup_item | non_dedup_len3 | baseline | 1735 | 0.028242 | 0.014734 | 0.039193 |
| dedup_vs_non_dedup_item | non_dedup_len3 | treatment | 4171 | 0.065692 | 0.045584 | 0.078398 |
| prefix_bucket_size | 1 | baseline | 1735 | 0.028242 | 0.014734 | 0.039193 |
| prefix_bucket_size | 1 | treatment | 4171 | 0.065692 | 0.045584 | 0.078398 |
| prefix_bucket_size | 2-5 | baseline | 521 | 0.055662 | 0.031233 | 0.063340 |
| prefix_bucket_size | 2-5 | treatment | 236 | 0.241525 | 0.233151 | 0.296610 |
| prefix_bucket_size | 21+ | baseline | 1404 | 0.216524 | 0.148718 | 0.298433 |
| prefix_bucket_size | 21+ | treatment | 125 | 0.000000 | 0.000000 | 0.000000 |
| min_target_sid_token_frequency | 1-20 | baseline | 1225 | 0.000000 | 0.000000 | 0.000000 |
| min_target_sid_token_frequency | 1-20 | treatment | 1071 | 0.006536 | 0.004159 | 0.006536 |
| min_target_sid_token_frequency | 21-100 | baseline | 831 | 0.049338 | 0.024884 | 0.077016 |
| min_target_sid_token_frequency | 21-100 | treatment | 561 | 0.001783 | 0.001783 | 0.001783 |
| min_target_sid_token_frequency | 101+ | baseline | 2471 | 0.171186 | 0.115338 | 0.228652 |
| min_target_sid_token_frequency | 101+ | treatment | 2758 | 0.117114 | 0.086910 | 0.141044 |
| target_popularity_train_interactions | 0 | baseline | 125 | 0.000000 | 0.000000 | 0.000000 |
| target_popularity_train_interactions | 0 | treatment | 125 | 0.000000 | 0.000000 | 0.000000 |
| target_popularity_train_interactions | 1 | baseline | 87 | 0.000000 | 0.000000 | 0.000000 |
| target_popularity_train_interactions | 1 | treatment | 87 | 0.000000 | 0.000000 | 0.000000 |
| target_popularity_train_interactions | 2-5 | baseline | 222 | 0.018018 | 0.013004 | 0.027027 |
| target_popularity_train_interactions | 2-5 | treatment | 222 | 0.018018 | 0.015766 | 0.036036 |
| target_popularity_train_interactions | 6-20 | baseline | 1064 | 0.017857 | 0.008808 | 0.028195 |
| target_popularity_train_interactions | 6-20 | treatment | 1064 | 0.008459 | 0.004511 | 0.014098 |
| target_popularity_train_interactions | 21+ | baseline | 3034 | 0.145353 | 0.096711 | 0.195452 |
| target_popularity_train_interactions | 21+ | treatment | 3034 | 0.104812 | 0.078066 | 0.123270 |
| user_history_length | 1 | baseline | 393 | 0.170483 | 0.116086 | 0.206107 |
| user_history_length | 1 | treatment | 393 | 0.111959 | 0.082119 | 0.114504 |
| user_history_length | 2-3 | baseline | 1294 | 0.137558 | 0.097699 | 0.177743 |
| user_history_length | 2-3 | treatment | 1294 | 0.103555 | 0.081695 | 0.122102 |
| user_history_length | 4-5 | baseline | 1500 | 0.094000 | 0.060486 | 0.128000 |
| user_history_length | 4-5 | treatment | 1500 | 0.068000 | 0.048981 | 0.085333 |
| user_history_length | 6-10 | baseline | 1345 | 0.057993 | 0.031900 | 0.093680 |
| user_history_length | 6-10 | treatment | 1345 | 0.037918 | 0.025053 | 0.049071 |
| sasrec_39_cold_fallback_items | sasrec_cold_fallback_target | baseline | 125 | 0.000000 | 0.000000 | 0.000000 |
| sasrec_39_cold_fallback_items | not_sasrec_cold_fallback_target | baseline | 4407 | 0.105287 | 0.069362 | 0.142727 |
| sasrec_39_cold_fallback_items | sasrec_cold_fallback_target | treatment | 125 | 0.000000 | 0.000000 | 0.000000 |
| sasrec_39_cold_fallback_items | not_sasrec_cold_fallback_target | treatment | 4407 | 0.075108 | 0.055628 | 0.090084 |

Full slice metrics are written to `results/s4_sasrec_sid_valid/Industrial_and_Scientific/s4_4_slice_metrics.csv`.

## Bottleneck Classification

Primary bottlenecks classified from the formal valid evidence:

`candidate coverage weakness, hierarchical prefix weakness, SID frequency imbalance, representation/tokenizer geometry mismatch, cold-item fallback weakness`

Evidence summary:

- SASRec-SID pool recall is lower than CF-SID by `-0.051192` absolute.
- SASRec-SID HR@20 / target-in-pool is higher than CF-SID: `0.833753` vs `0.737679`.
- SASRec-SID has `86` SASRec-only HR@20 hits and `117` SASRec-only HR@50 hits.
- Union oracle improves over CF by `0.018976` at HR@20 and `0.025816` at HR@50.

No engineering failure was found in the formal artifact chain. No scheduler/test/generalization claim is made.

## Gate

Decision: **AUXILIARY_STREAM_GO**

Reason: SASRec-SID is weaker as a single stream but contributes nontrivial SASRec-only valid hits and improves union oracle over CF without artifact failures.

Frozen boundary:

- Do not claim SASRec-SID single-stream superiority over CF-SID.
- Do not compare direct SASRec item-ranking metrics with Qwen SID-generation metrics as equivalent.
- Do not use test split evidence.
- Next engineering mainline: auxiliary candidate stream/fusion design using frozen valid-only artifacts.
