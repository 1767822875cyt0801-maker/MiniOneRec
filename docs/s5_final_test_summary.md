# S5 Final Test Summary

## Research Motivation

MiniOneRec uses semantic IDs as the output space for generative recommendation. The baseline CF-SID stream is strong because it captures collaborative structure directly from user-item interactions, but it is limited by the representation geometry of the CF embedding space. S5 asked whether a behavior-derived auxiliary stream could add complementary recall without replacing the CF baseline.

The central research question was:

Can SASRec-derived item representations recover targets that CF-SID misses, and can a frozen ranker convert those auxiliary candidates into measurable final-test gains without using test results for tuning?

## Why SASRec-SID Was Introduced

SASRec-SID was introduced as a behavior-aware alternative to CF-SID. Instead of using only collaborative item embeddings, SASRec embeddings encode sequential behavior. The hypothesis was not that SASRec-only would dominate CF-only, but that SASRec might retrieve different items, especially items that are hard for the CF stream.

This made SASRec-SID useful as an auxiliary candidate source.

## Validation Findings

The validation-stage S5 gate selected:

- candidate policy: `source_aware_rrf_lam0.75_bonus0.01`
- `lambda_sasrec=0.75`
- `source_bonus=0.01`
- compatibility mode: `source_independent_projection`
- ranker: frozen P2-4 history-aware ranker

The validation gate concluded:

`FROZEN_RANKER_CONVERTS`

The key validation lesson was that fixed candidate fusion alone was not enough. The auxiliary stream added recall, but ranking was required to convert the added pool into top-20 accuracy.

## Final-Test Findings

Final-test metrics:

| Policy | HR@20 | NDCG@20 | hits@20 | target-in-pool |
|---|---:|---:|---:|---:|
| CF-only | 0.08890359585263623 | 0.058905061869372645 | 403 | 576 / 4533 |
| SASRec-only | 0.06485771012574454 | 0.048844342920883924 | 294 | 356 / 4533 |
| fixed RRF | 0.09110964041473638 | 0.05889075153830667 | 413 | 681 / 4533 |
| projected ranker | 0.10258107213765719 | 0.0631805105222819 | 465 | 681 / 4533 |

Projected ranker versus CF-only:

- HR@20 absolute uplift: `+0.01367747628502096`
- HR@20 relative uplift: `+15.38%`
- NDCG@20 absolute uplift: `+0.004275448652909255`
- NDCG@20 relative uplift: `+7.26%`

The final result is positive: the frozen projected ranker improves both HR@20 and NDCG@20 over CF-only on the held-out final test.

## Why SASRec-Only Underperformed CF-Only

SASRec-only underperformed CF-only:

- SASRec-only HR@20: `0.06485771012574454`
- CF-only HR@20: `0.08890359585263623`
- SASRec-only NDCG@20: `0.048844342920883924`
- CF-only NDCG@20: `0.058905061869372645`

This suggests that the SASRec-derived SID space is weaker as a standalone retrieval system for this dataset and model setup. Sequential behavior embeddings may be noisier, less aligned with exact item retrieval, or less suited to the SID generation objective than the CF-SID baseline.

## Why SASRec Candidates Remain Complementary

Even though SASRec-only was weaker, it expanded the candidate target pool:

- CF target-in-pool count: `576`
- union target-in-pool count: `681`
- added targets: `105`
- union oracle hits@20: `477`

The projected ranker preserved most CF hits while recovering some SASRec-only targets:

- CF hits preserved: `394`
- CF hits lost: `9`
- SASRec-only recoveries: `26`

This is the core S5 finding: auxiliary behavior candidates are not a replacement for CF candidates, but they are useful when a learned frozen ranker can identify when to trust them.

## Why Fixed RRF Was Insufficient

Fixed RRF improved HR@20 slightly:

- CF-only HR@20: `0.08890359585263623`
- fixed RRF HR@20: `0.09110964041473638`

But fixed RRF did not improve NDCG@20:

- CF-only NDCG@20: `0.058905061869372645`
- fixed RRF NDCG@20: `0.05889075153830667`

The union candidate pool added many items and high candidate noise. Fixed RRF could expose more candidates, but it did not have enough context to decide which auxiliary candidates deserved promotion.

## How the Projected Ranker Converted Auxiliary Recall

The projected ranker used the frozen source-independent projection. Source-specific identity features were zeroed after normalization, so the ranker did not treat SASRec as if it were the original text stream. It retained source-independent ranking, score, popularity, history similarity, bucket, and expansion features.

This allowed the frozen P2 history-aware ranker to convert the larger union pool into better ranking:

- projected ranker HR@20: `0.10258107213765719`
- projected ranker NDCG@20: `0.0631805105222819`
- projected ranker hits@20: `465`

The ranker was therefore the effective conversion mechanism. The auxiliary stream created opportunity; the ranker selected usable opportunities.

## Engineering Problems Encountered

The final test encountered one operational failure after the CF stream completed:

`data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/test.csv`

was missing on AutoDL.

The failure happened before SASRec generation started. CF generation and CF candidate evaluation had already completed. The final process was recovered through a strict partial-resume path, not by deleting or rerunning CF artifacts.

The LogitProcessor emitted `No valid tokens found ... at step 5` warnings during generation. The closeout audit counted `1846` such warnings. Candidate reports still showed:

- CF invalid SID count: `0`
- SASRec invalid SID count: `0`

The warnings did not compromise final artifact completeness.

## Deterministic Recovery of SASRec Test CSV

The missing SASRec test CSV was reconstructed deterministically from frozen inputs:

- CF test CSV row order and `history_item_id` / `item_id`
- SASRec frozen `item2sid.json`
- SASRec valid CSV SID serialization schema

The rebuilt SASRec test CSV SHA256 was recorded as:

`cfed313aa1acca6a3ed17ffc33b86e90ebfa5275f878ddfacb05838c8ed017fc`

The final input inventory confirmed:

- rows: `4533`
- `item_id_aligned=true`
- `history_item_id_aligned=true`

## Strict Partial-Resume Mechanism

The partial resume accepted only one state:

- CF predictions exist and match expected hash
- CF candidates exist and match expected hash
- CF candidate report exists and matches expected hash
- SASRec predictions/candidates/report do not exist yet
- final metrics do not exist yet

The resume started from:

`sasrec_test_generation`

The audit confirmed:

- CF generation was not rerun
- CF candidate evaluation was not rerun
- `allow-overwrite` was not used
- final exit code was `0`

## Final Conclusion

S5 demonstrates a useful pattern for behavior-augmented generative recommendation: a behavior-derived SID stream may be weaker as a standalone retrieval system, but it can still provide complementary candidate coverage. Simple fixed fusion is too blunt for this noisy union pool. A frozen, history-aware ranker with a carefully defined source-independent projection can convert the additional recall into final-test ranking gains.

For a paper, technical report, resume, or interview, the concise result is:

S5 added a SASRec-derived auxiliary SID stream to a CF-SID generative recommender. SASRec-only underperformed CF-only, but it expanded the target candidate pool by 105 targets. A frozen history-aware projected ranker converted this auxiliary recall into a final-test improvement of `+15.38%` relative HR@20 and `+7.26%` relative NDCG@20 over CF-only, under a strict no-test-tuning protocol.

S5 is complete and closed.
