# S5 Final Test Closeout Audit

Status: **GO_CLOSEOUT**

This document records the read-only closeout audit for the S5 frozen final test. The audit used only the synchronized final evidence bundle:

- Bundle: `incoming/s5_final_test_closeout/s5_final_test_closeout_bundle.tar.gz`
- Bundle SHA256 file: `incoming/s5_final_test_closeout/s5_final_test_closeout_bundle.tar.gz.sha256`

No training, generation, candidate evaluation, fusion, reranking, finalize step, or GPU workload was rerun during this closeout documentation task.

## Evidence Integrity

The bundle exists, is non-empty, and was verified with `sha256sum -c` from `incoming/s5_final_test_closeout/`.

The bundle was unpacked into an independent temporary audit directory during the read-only audit. The file list in:

`results/s5_auxiliary_fusion/Industrial_and_Scientific/s5_final_test_closeout/final_frozen_test_files.sha256`

was verified successfully for all listed files.

Required artifacts were present and non-empty:

- `final_test_metrics.json`
- `final_test_complete_manifest.json`
- `partial_resume_closeout_manifest.json`
- `partial_resume_manifest.json`
- CF predictions, candidates, and candidate report
- SASRec predictions, candidates, and candidate report
- `configs/s5_auxiliary_fusion/frozen_release_config.json`
- `results/s5_auxiliary_fusion/Industrial_and_Scientific/s5_1_release_manifest.json`
- execution log
- exit-code file

Execution exit code: `0`.

Prediction row counts:

- CF predictions: `4533`
- SASRec predictions: `4533`

The input inventory in `partial_resume_closeout_manifest.json` recorded:

- test CSV rows: `4533`
- `item_id_aligned=true`
- `history_item_id_aligned=true`
- CF test CSV SHA256: `50843bfd5574a34b4d1cc95dc0fa7fd911bb9a516fd167b98538ba95212d7300`
- rebuilt SASRec test CSV SHA256: `cfed313aa1acca6a3ed17ffc33b86e90ebfa5275f878ddfacb05838c8ed017fc`

The concrete `partial_resume_manifest.json` contains concrete 64-character SHA256 values for:

- CF predictions
- CF candidates
- CF candidate report
- rebuilt SASRec test CSV

No evidence was found for `allow-overwrite`, overwrite bypass, CF generation rerun, or CF candidate evaluation rerun.

## Partial Resume Protocol

The final test initially completed the CF stream and then stopped before SASRec generation because:

`data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/test.csv`

was missing.

The final closeout evidence shows that the approved partial-resume path started at:

`sasrec_test_generation`

The resume protocol verified the existing CF artifacts, rebuilt the SASRec test CSV deterministically, generated SASRec predictions, evaluated SASRec candidates, and then completed the frozen final metrics computation.

The log audit confirmed:

- `cf_test_generation` count: `0`
- `cf_test_candidate_eval` count: `0`
- `--allow-overwrite` count: `0`
- final completion manifest: `complete=true`

## Frozen Parameter Inventory

The final test used the frozen S5 release configuration:

- selected candidate policy: `source_aware_rrf_lam0.75_bonus0.01`
- `lambda_sasrec=0.75`
- `source_bonus=0.01`
- projection: `source_independent_projection`
- candidate mode: `exact_full_sid_only`
- CF SID: `cf_k512_dedup`
- SASRec SID: `sasrec_v3_k512_dedup`
- `num_beams=50`
- `max_new_tokens=6`
- seed: `42`
- frozen ranker: P2-4 history-aware ranker
- frozen ranker SHA256: `6a3bca7cc714d295572ce4a3953d60719a1c72757895835554a83d747dca65ce`
- projection contract hash: `5bfd5ee02aa46eb5cf9069556fc5d4aa96ff424933d7a88a6be81b49d81989c8`

The final metrics file records:

`test_result_must_not_change_parameters=true`

S5 final-test results are frozen. They must not be used to tune lambda, source bonus, projection, ranker, beam count, token budget, SID construction, checkpoint selection, or any other test-time parameter.

## Warning Audit

The execution log contained:

`No valid tokens found ... at step 5`

Count: `1846`

Candidate SID validity remained clean:

- CF invalid SID count: `0`
- CF invalid SID rate: `0.0`
- SASRec invalid SID count: `0`
- SASRec invalid SID rate: `0.0`

Interpretation: the LogitProcessor warnings did not prevent final artifact completion, did not create invalid candidate SIDs, and did not block final metric computation. No decoding logic was modified or rerun during closeout.

## Final Metrics

| Policy | HR@1 | HR@5 | HR@10 | HR@20 | HR@50 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 | MRR | hits@20 | target-in-pool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CF-only | 0.041694 | 0.059784 | 0.072138 | 0.088904 | 0.127068 | 0.041694 | 0.050739 | 0.054709 | 0.058905 | 0.066428 | 0.051725 | 403 | 576 / 4533 |
| SASRec-only | 0.035517 | 0.054269 | 0.057798 | 0.064858 | 0.078535 | 0.035517 | 0.045957 | 0.047088 | 0.048844 | 0.051517 | 0.044489 | 294 | 356 / 4533 |
| fixed RRF | 0.038826 | 0.058681 | 0.074344 | 0.091110 | 0.124421 | 0.038826 | 0.049563 | 0.054615 | 0.058891 | 0.065477 | 0.051228 | 413 | 681 / 4533 |
| projected ranker | 0.038826 | 0.068167 | 0.080521 | 0.102581 | 0.135010 | 0.038826 | 0.053716 | 0.057668 | 0.063181 | 0.069684 | 0.053377 | 465 | 681 / 4533 |

Additional audited facts:

- union target-in-pool count: `681`
- CF target-in-pool count: `576`
- added targets from union pool: `105`
- union oracle hits@20: `477`
- projected ranker CF hits preserved: `394`
- projected ranker CF hits lost: `9`
- projected ranker SASRec-only recoveries: `26`
- CF-only candidate noise ratio: `0.997459`
- SASRec-only candidate noise ratio: `0.998429`
- projected ranker candidate noise ratio: `0.998433`
- average union size for union policies: `95.8833002426649`

## Improvements

Projected ranker versus CF-only:

- HR@20 absolute uplift: `+0.01367747628502096`
- HR@20 relative uplift: `+15.38%`
- NDCG@20 absolute uplift: `+0.004275448652909255`
- NDCG@20 relative uplift: `+7.26%`

Fixed RRF versus CF-only:

- HR@20 absolute change: `+0.0022060445621001484`
- HR@20 relative change: `+2.48%`
- NDCG@20 absolute change: `-0.000014310331065976734`
- NDCG@20 relative change: `-0.024%`

Projected ranker versus fixed RRF:

- HR@20 absolute change: `+0.01147143172292081`
- HR@20 relative change: `+12.59%`
- NDCG@20 absolute change: `+0.004289758983975225`
- NDCG@20 relative change: `+7.28%`

SASRec-only versus CF-only:

- HR@20 absolute change: `-0.02404588572689169`
- HR@20 relative change: `-27.05%`
- NDCG@20 absolute change: `-0.010060718948488721`
- NDCG@20 relative change: `-17.08%`

Union candidate pool versus CF pool:

- target-in-pool count increase: `+105`
- target-in-pool rate increase: `+0.0231634679020516`

## Final Research Conclusion

S5 final-test evidence supports the frozen auxiliary conversion claim. SASRec-only underperformed CF-only, but SASRec introduced complementary candidates that expanded the reachable target pool from `576` to `681` targets. Fixed RRF exposed only a small HR@20 gain and did not improve NDCG@20. The frozen history-aware projected ranker converted the auxiliary candidate pool into a clear final-test improvement over CF-only: `+15.38%` relative HR@20 and `+7.26%` relative NDCG@20.

S5 is formally closed. The test results are final evidence only and cannot be used for further parameter tuning.
