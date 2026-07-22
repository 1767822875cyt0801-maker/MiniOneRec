# S4-3 Smoke Full-Valid Closeout

## Scope

This audit is read-only over the returned AutoDL smoke full-valid bundle. It does not run GPU, start SFT, run generation, read/evaluate `test`, rebuild SID artifacts, modify fusion, or compare SASRec direct item-ranking against Qwen SID-generation metrics.

Bundle root:

`incoming/s4_smoke_full_valid/s4_smoke_full_valid_closeout_20260714_144743`

The run is `smoke/valid/full_valid`: smoke checkpoints trained with the bounded smoke budget and evaluated over all 4532 valid rows. It is not a 30k formal result.

## Artifact Inventory

| stream | sid_version | scope | complete | prediction_rows | candidate_rows | valid_rows | checkpoint_tree_sha256 | pred_sha256 | candidates_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | cf_k512_dedup | full_valid | True | 4532 | 4532 | 4532 | 62f12b0a49b3f97e064dbd653a132afa7b98b28c33954041c83fd361b9402f68 | bfb015865fa2 | 052eef9e3b54 |
| treatment | sasrec_v3_k512_dedup | full_valid | True | 4532 | 4532 | 4532 | 36048caf9b6011e3ccd56fe8a3fd9cabc97708ea1dc83f3bcba5c89055e9f588 | f35f2d50bbda | d7c89f9148b8 |

Required files were present for both streams: `artifact_manifest.json`, `generation/predictions.json`, `candidates/candidates.jsonl`, `candidates/candidate_report.json`, and `summary/metrics_summary.json`.

Manifest checks passed for stream, SID version, full-valid scope, `complete=true`, expected/actual prediction rows, and artifact SHA256. Both candidate reports use `split=valid`; parity manifest records `test_read=false`.

## Fairness Matrix

| field | value / status | evidence |
| --- | --- | --- |
| base_model | /root/autodl-tmp/models/Qwen2.5-0.5B | proved equal by parity manifest / generated command plan |
| candidate_mode | exact_full_sid_only | proved equal by parity manifest / generated command plan |
| gradient_accumulation_steps | 8 | proved equal by parity manifest / generated command plan |
| learning_rate | 2e-5 | proved equal by parity manifest / generated command plan |
| max_new_tokens | 6 | proved equal by parity manifest / generated command plan |
| num_beams | 50 | proved equal by parity manifest / generated command plan |
| num_train_epochs | 1 | proved equal by parity manifest / generated command plan |
| per_device_train_batch_size | 4 | proved equal by parity manifest / generated command plan |
| sample | 2000 | proved equal by parity manifest / generated command plan |
| seed | 42 | proved equal by parity manifest / generated command plan |
| valid_rows | 4532 | proved equal by parity manifest / generated command plan |
| checkpoint path/hash | different by stream | expected path difference; each artifact manifest fingerprints its own final_checkpoint |
| train/valid CSV | different by SID version | expected path/hash difference; no cross-stream mix observed |
| item2sid/sid2items/valid_sid_set | different by SID version | expected path/hash difference; candidate_report inputs and artifact manifest match stream |
| SID token set | False | Different SID mappings naturally introduce different SID special token sets; this is not a routing failure but should be monitored as representational/tokenizer-surface difference. |

Fields proven equal by evidence include shared `BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B`, `sample=2000`, `num_train_epochs=1`, `learning_rate=2e-5`, per-device batch `4`, gradient accumulation `8`, seed `42`, `num_beams=50`, `max_new_tokens=6`, exact candidate mode, and `valid_rows=4532`.

Evidence-insufficient fields are limited to trainer-computed effective step count, generation beam scores, and per-sample latency because those are not written by the current artifacts. No fairness-invalidating mismatch was found. Expected path/hash differences exist for SID-specific train/valid CSVs, mappings, output directories, and checkpoints.

## Metric Comparison

All metrics below are item-level before any reranker. HR/NDCG/MRR were recomputed from `candidates.jsonl`; `exact_candidate_recall` comes from exact candidate pool target inclusion and matches HR@50 because each row has 50 candidates.

| metric | CF-SID baseline | SASRec-SID treatment | absolute delta | relative delta |
| --- | --- | --- | --- | --- |
| hr@1 | 2.6478% | 0.5296% | -2.1183% | -80.00% |
| hr@3 | 3.9276% | 0.7502% | -3.1774% | -80.90% |
| hr@5 | 4.4572% | 0.8826% | -3.5746% | -80.20% |
| hr@10 | 5.7590% | 0.9929% | -4.7661% | -82.76% |
| hr@20 | 7.3919% | 1.2357% | -6.1562% | -83.28% |
| hr@50 | 10.2162% | 1.8535% | -8.3628% | -81.86% |
| ndcg@10 | 4.0449% | 0.7458% | -3.2991% | -81.56% |
| ndcg@20 | 4.4531% | 0.8039% | -3.6492% | -81.95% |
| ndcg@50 | 5.0076% | 0.9245% | -4.0832% | -81.54% |
| mrr | 3.7172% | 0.7006% | -3.0165% | -81.15% |
| valid_sid_rate | 100.0000% | 100.0000% | 0.0000% | 0.00% |
| invalid_prediction_count | 0 | 0 | 0 | n/a |
| empty_candidate_count | 0 | 0 | 0 | n/a |
| average_candidate_count | 50 | 50 | 0 | 0.00% |
| exact_candidate_recall | 10.2162% | 1.8535% | -8.3628% | -81.86% |
| prediction_row_count | 4532 | 4532 | 0 | 0.00% |

The treatment is substantially weaker in this 2k smoke run. This is a research signal and a formal-risk warning, not by itself an engineering gate failure.

## Per-Sample Complementarity

Union oracle means target present in the union of CF and SASRec candidate sets at K. It is a candidate upper bound only; it is not a fusion/reranking result.

| K | CF hit / SASRec miss | SASRec hit / CF miss | both hit | neither hit | union oracle HR | avg overlap count | avg Jaccard | CF dup total | SASRec dup total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| @10 | 253 | 37 | 8 | 4234 | 6.5755% | 0.2246 | 0.011936 | 0 | 0 |
| @20 | 325 | 46 | 10 | 4151 | 8.4069% | 0.3111 | 0.008035 | 0 | 0 |
| @50 | 451 | 72 | 12 | 3997 | 11.8049% | 0.7509 | 0.007640 | 0 | 0 |

The streams have low candidate overlap. SASRec contributes some unique hits even though its standalone smoke metrics are weak.

## Anomaly Diagnosis

| stream | check | value | classification |
| --- | --- | --- | --- |
| baseline | invalid SID count | 0 | no engineering failure |
| baseline | empty candidate count | 0 | no engineering failure |
| baseline | candidate count min/mean/max | 50/50.00/50 | normal fixed 50 beams/candidates |
| baseline | all samples same first SID | False | no collapse |
| baseline | unique first predicted SID count | 624 | diverse enough for smoke audit |
| baseline | predicted SID token length distribution | {'3': 60636, '4': 165964} | contains 3/4-token SIDs; no invalid SID evidence |
| baseline | predicted SID with <d_ token count | 165964 | fourth-layer dedup token generated in some beams |
| treatment | invalid SID count | 0 | no engineering failure |
| treatment | empty candidate count | 0 | no engineering failure |
| treatment | candidate count min/mean/max | 50/50.00/50 | normal fixed 50 beams/candidates |
| treatment | all samples same first SID | False | no collapse |
| treatment | unique first predicted SID count | 649 | diverse enough for smoke audit |
| treatment | predicted SID token length distribution | {'3': 204136, '4': 22464} | contains 3/4-token SIDs; no invalid SID evidence |
| treatment | predicted SID with <d_ token count | 22464 | fourth-layer dedup token generated in some beams |

Diagnosis:

- Engineering failure: none found in the full-valid artifacts.
- Expected smoke variance: possible, because only 2000 training rows were used.
- Representation weakness: plausible for SASRec-SID in this smoke checkpoint, given much lower target inclusion.
- Generative decoding weakness: possible, but no invalid SID, empty candidate, row-count, or EOS/max-token truncation failure is visible from these artifacts.
- Evidence insufficiency: beam scores and latency are unavailable; formal training dynamics cannot be inferred from smoke.

## Formal Gate

Decision: `CONDITIONAL_GO_FORMAL`

Rationale: Full valid smoke pipeline is complete and valid; treatment underperforms CF strongly in 2k smoke but has no invalid SID, empty candidate, row-count, or routing failure. Proceed to formal only as controlled engineering gate, not as performance claim.

Formal gate conditions satisfied:

- both artifacts are complete;
- both have 4532 predictions and 4532 candidate rows;
- `test_read=false`;
- routing and checkpoint provenance are stream-correct;
- exact candidate pipeline produced non-empty fixed-size candidates;
- invalid SID count is zero for both streams;
- no evidence of row loss, artifact corruption, candidate collapse, or cross-stream path mixing.

Condition: proceed only as a controlled 30k formal experiment. Do not claim SASRec-SID improves recommendation quality from this smoke run.

## Next AutoDL Commands

These commands use the same base model, sample budget by `CONFIG_MODE=formal` (`30000`), epoch, learning rate, batch, gradient accumulation, seed, beam, max-new-token, and exact candidate mode. They remain valid-only and must not read `test`.

```bash
# 1. Environment preflight
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=env-preflight \
CONFIG_MODE=formal \
bash scripts/run_s4_sasrec_sid_valid.sh

# 2. Formal parity / command manifest
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=parity \
DRY_RUN=0 \
CONFIG_MODE=formal \
bash scripts/run_s4_sasrec_sid_valid.sh

# 3. CF-SID baseline formal train, sample=30000 by formal default
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=train \
TRAIN_STREAM=baseline \
DRY_RUN=0 \
CONFIG_MODE=formal \
bash scripts/run_s4_sasrec_sid_valid.sh

# 4. SASRec-SID treatment formal train, same budget and controls
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=train \
TRAIN_STREAM=treatment \
DRY_RUN=0 \
CONFIG_MODE=formal \
bash scripts/run_s4_sasrec_sid_valid.sh

# 5. Bounded candidate preflight, baseline then treatment
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=candidates \
TRAIN_STREAM=baseline \
CANDIDATE_ROW_LIMIT=8 \
DRY_RUN=0 \
CONFIG_MODE=formal \
bash scripts/run_s4_sasrec_sid_valid.sh

BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=candidates \
TRAIN_STREAM=treatment \
CANDIDATE_ROW_LIMIT=8 \
DRY_RUN=0 \
CONFIG_MODE=formal \
bash scripts/run_s4_sasrec_sid_valid.sh

# 6. Full-valid candidates, baseline then treatment
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=candidates \
TRAIN_STREAM=baseline \
CANDIDATE_ROW_LIMIT=0 \
DRY_RUN=0 \
CONFIG_MODE=formal \
bash scripts/run_s4_sasrec_sid_valid.sh

BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=candidates \
TRAIN_STREAM=treatment \
CANDIDATE_ROW_LIMIT=0 \
DRY_RUN=0 \
CONFIG_MODE=formal \
bash scripts/run_s4_sasrec_sid_valid.sh

# 7. Final artifact audits, both streams and scopes
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B STAGE=audit-existing TRAIN_STREAM=baseline CANDIDATE_ROW_LIMIT=8 CONFIG_MODE=formal bash scripts/run_s4_sasrec_sid_valid.sh
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B STAGE=audit-existing TRAIN_STREAM=treatment CANDIDATE_ROW_LIMIT=8 CONFIG_MODE=formal bash scripts/run_s4_sasrec_sid_valid.sh
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B STAGE=audit-existing TRAIN_STREAM=baseline CANDIDATE_ROW_LIMIT=0 CONFIG_MODE=formal bash scripts/run_s4_sasrec_sid_valid.sh
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B STAGE=audit-existing TRAIN_STREAM=treatment CANDIDATE_ROW_LIMIT=0 CONFIG_MODE=formal bash scripts/run_s4_sasrec_sid_valid.sh
```

## Cannot Claim Yet

- No formal 30k result exists yet.
- No scheduler/fusion/reranking conclusion is supported.
- No Text-SID or lambda tuning conclusion is supported.
- Smoke treatment underperformance is not proof that SASRec-SID is worse under the formal budget.
- Union oracle is only a candidate-set upper bound, not an implemented fusion result.
