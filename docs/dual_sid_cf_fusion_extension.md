# Dual-SID CF Fusion Extension

Date: 2026-06-29

This note records the Text-SID + CF-SID dual-stream candidate fusion extension
after the Stage 6 CS-SID item-level evaluation. It is based on the AutoDL result
sync in `docs/autodl_dual_sid_results_sync_2026-06-29.md` and the recovered
project handoff in `docs/recovered_codex_handoff_2026-06-29.md`.

## 1. Consistency Check

The AutoDL sync supports treating the dual-SID results as comparable to the
existing Stage 6 item-level results.

| Check | Industrial_and_Scientific | Office_Products | Status |
|---|---:|---:|---|
| Text/CF test.csv exists | yes / yes | yes / yes | PASS |
| test.csv rows | 4533 / 4533 | 4866 / 4866 | PASS |
| `user_id` sequence | equal | equal | PASS |
| `history_item_id` sequence | equal | equal | PASS |
| `item_id` ground truth | equal | equal | PASS |
| exact candidate rows | 4533 / 4533 | 4866 / 4866 | PASS |
| p3 candidate rows | 4533 / 4533 | 4866 / 4866 | PASS |
| exact fusion/rerank/schema reports | present | present | PASS |
| p3 fusion/rerank/schema reports | present | present | PASS |

No conclusion-changing inconsistency was found. Two caveats remain:

- Industrial candidate paths use mixed suffixes such as `v2` for text and `v1`
  for CF, while Office uses `vdual`. The row counts, target identity checks, and
  report contents are aligned, so this is a naming/archival issue rather than a
  metric issue.
- The local VSCode workspace does not contain the AutoDL result JSONs. The
  evidence here depends on the manually synced AutoDL inspection log. For final
  archival, keep the synced log or copy the relevant JSON summaries.

## 2. Motivation

The original Text-SID stream and the behavior/CF-SID stream make different
errors. Text-SID is strong because product text provides stable semantic
structure, while CF-SID can surface behaviorally substitutable items that text
does not rank in the top 20.

The dual-SID CF fusion experiment tests this hypothesis at item level:

1. Generate item candidates from Text-SID predictions.
2. Generate item candidates from CF-SID predictions.
3. Align the two streams by test row and target item.
4. Fuse candidates with min-rank and RRF.
5. Apply the same lightweight non-leaky reranker used in Stage 6.

This is CPU post-processing over existing prediction files. It does not retrain
SFT, does not start a GPU job, and does not change the Stage 6 training setup.

## 3. Relation To The CS-SID Mainline

Dual-SID CF fusion is not a replacement for the CS-SID mainline.

The original mainline result remains:

```text
Text-SID baseline -> CS embedding -> CS-SID -> sid-only SFT
-> candidate expansion -> lightweight rerank
```

That pipeline proves that a single improved SID version can beat the Text-SID
baseline. The dual-SID experiment is a Stage 6 extension:

```text
Text-SID candidates + CF-SID candidates -> item-level fusion -> rerank
```

It shows that Text-SID and CF-SID candidate pools are complementary even when
the standalone CF stream is not stronger than Text.

Recommended naming:

```text
Stage 6.5: Dual-SID Text+CF Candidate Fusion Extension
```

## 4. Dual-SID Results

### Industrial_and_Scientific

| Mode | Text HR@20 | CF HR@20 | Union HR@20 | Behavior-only hits@20 | Minrank HR@20 | RRF HR@20 | RRF NDCG@20 | Rerank HR@20 | Rerank NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 0.128612 | 0.120450 | 0.163468 | 158 | 0.131921 | 0.130157 | 0.075960 | 0.145158 | 0.080448 |
| p3 | 0.112508 | 0.073902 | 0.138760 | 119 | 0.101919 | 0.113832 | 0.063378 | 0.141407 | 0.079782 |

### Office_Products

| Mode | Text HR@20 | CF HR@20 | Union HR@20 | Behavior-only hits@20 | Minrank HR@20 | RRF HR@20 | RRF NDCG@20 | Rerank HR@20 | Rerank NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 0.144472 | 0.142622 | 0.189684 | 220 | 0.161529 | 0.162145 | 0.092189 | 0.172215 | 0.101124 |
| p3 | 0.141800 | 0.113440 | 0.177148 | 172 | 0.146732 | 0.148788 | 0.086779 | 0.168722 | 0.099451 |

## 5. Recommended Main Metric

Use `exact + rerank` as the main dual-SID extension metric.

| Category | exact + rerank HR@20 | exact + rerank NDCG@20 | p3 + rerank HR@20 | p3 + rerank NDCG@20 | Winner |
|---|---:|---:|---:|---:|---|
| Industrial_and_Scientific | 0.145158 | 0.080448 | 0.141407 | 0.079782 | exact |
| Office_Products | 0.172215 | 0.101124 | 0.168722 | 0.099451 | exact |

Exact is consistently better on both categories:

- Industrial: exact beats p3 by +0.003751 HR@20 and +0.000666 NDCG@20.
- Office: exact beats p3 by +0.003494 HR@20 and +0.001673 NDCG@20.

## 6. Comparison With Stage 6 CS-SID

The old Stage 6 best results were:

| Category | Stage 6 best setting | Stage 6 HR@20 | Stage 6 NDCG@20 |
|---|---|---:|---:|
| Industrial_and_Scientific | `cs_alpha0.2_k512_dedup + prefix@3 rerank` | 0.1366 | 0.0784 |
| Office_Products | `cs_alpha0.7_k512_dedup + prefix@3 rerank` | 0.1492 | 0.0884 |

Dual-SID exact + rerank improves over both:

| Category | Stage 6 best HR@20 | Dual exact + rerank HR@20 | HR gain | Stage 6 best NDCG@20 | Dual exact + rerank NDCG@20 | NDCG gain |
|---|---:|---:|---:|---:|---:|---:|
| Industrial_and_Scientific | 0.136600 | 0.145158 | +0.008558 | 0.078400 | 0.080448 | +0.002048 |
| Office_Products | 0.149200 | 0.172215 | +0.023015 | 0.088400 | 0.101124 | +0.012724 |

This makes dual-SID exact + rerank the strongest current item-level result in
the project, while CS-SID remains the strongest single-SID mainline.

## 7. What The Fusion Diagnostics Mean

### Union Recall

`union_HR@20` asks whether the target appears in either stream's top-20
candidate list. It is an upper-bound-style diagnostic for two-stream
complementarity, not the final ranked recommendation metric.

| Category | Mode | Union HR@20 | Rerank HR@20 | Union minus rerank |
|---|---|---:|---:|---:|
| Industrial_and_Scientific | exact | 0.163468 | 0.145158 | +0.018310 |
| Industrial_and_Scientific | p3 | 0.138760 | 0.141407 | -0.002647 |
| Office_Products | exact | 0.189684 | 0.172215 | +0.017468 |
| Office_Products | p3 | 0.177148 | 0.168722 | +0.008426 |

For exact mode, both categories still lose about 1.7-1.8 HR points between
union top-20 recall and final reranked top-20. This is the remaining sorting
loss: useful candidates are present, but not always ranked high enough.

For Industrial p3, rerank HR@20 is higher than union HR@20 because p3 creates a
larger fused candidate pool. Rerank can pull targets from outside the original
two-stream top-20 union into the final top-20. This means `union_HR@20` is not a
complete upper bound for p3; candidate pool recall at the full budget is the
better upper-bound diagnostic.

### Behavior-only Hits

`behavior_only_hit20` counts test rows where CF top-20 hits the target but Text
top-20 does not.

| Category | exact behavior-only hits@20 | p3 behavior-only hits@20 |
|---|---:|---:|
| Industrial_and_Scientific | 158 | 119 |
| Office_Products | 220 | 172 |

These counts are the clearest evidence that CF-SID contributes complementary
targets rather than merely duplicating Text-SID candidates.

### Rerank Gain

The lightweight reranker improves over raw RRF in every mode:

| Category | Mode | RRF HR@20 | Rerank HR@20 | HR gain | RRF NDCG@20 | Rerank NDCG@20 | NDCG gain |
|---|---|---:|---:|---:|---:|---:|---:|
| Industrial_and_Scientific | exact | 0.130157 | 0.145158 | +0.015001 | 0.075960 | 0.080448 | +0.004488 |
| Industrial_and_Scientific | p3 | 0.113832 | 0.141407 | +0.027576 | 0.063378 | 0.079782 | +0.016404 |
| Office_Products | exact | 0.162145 | 0.172215 | +0.010070 | 0.092189 | 0.101124 | +0.008936 |
| Office_Products | p3 | 0.148788 | 0.168722 | +0.019934 | 0.086779 | 0.099451 | +0.012672 |

The reranker is especially important when candidate expansion is noisy.

## 8. Why p3 Is Worse Than Exact

Prefix@3 expansion was useful in the single CS-SID Stage 6 setting, but it is
less stable in dual-SID fusion.

Observed behavior:

- Industrial exact Text HR@20 is 0.128612, but p3 Text HR@20 drops to 0.112508.
- Industrial exact CF HR@20 is 0.120450, but p3 CF HR@20 drops to 0.073902.
- Office exact CF HR@20 is 0.142622, but p3 CF HR@20 drops to 0.113440.
- p3 candidate previews show much larger first-row candidate lists, for example
  181 for Industrial CF p3 and 171 for Office CF p3, compared with 20 in exact.

Interpretation: p3 increases candidate breadth but introduces more early-rank
noise. The current fusion and rerank heuristics can recover part of this noise,
but exact remains better in final top-20 HR/NDCG on both categories.

## 9. Lambda Sweep Observation

The exact-mode lambda sweep shows strong sensitivity in RRF weighting.

| Category | lambda_text | RRF HR@20 | RRF NDCG@20 |
|---|---:|---:|---:|
| Industrial_and_Scientific | 0.5 | 0.130157 | 0.075960 |
| Industrial_and_Scientific | 0.6 | 0.128612 | 0.075323 |
| Industrial_and_Scientific | 0.7 | 0.128612 | 0.075006 |
| Industrial_and_Scientific | 0.8 | 0.128612 | 0.074784 |
| Office_Products | 0.5 | 0.162145 | 0.092189 |
| Office_Products | 0.6 | 0.144472 | 0.087249 |
| Office_Products | 0.7 | 0.144472 | 0.086950 |
| Office_Products | 0.8 | 0.144472 | 0.086704 |

At lambda 0.6-0.8, HR@20 collapses back to the Text stream HR@20. This likely
means the RRF top-20 becomes too text-dominated and behavior-only hits are
pushed out. It does not by itself prove an implementation bug, because
`union_HR@20` and `behavior_only_hit20` remain positive, but it is a real
weight-sensitivity risk.

Recommended default for this extension:

```text
lambda_text = 0.5
k_rrf = 60
source_weight = 4.0 for rerank
main mode = exact
```

## 10. Recommended Final Report Wording

Short version:

```text
After the single-SID CS-SID pipeline, we added a Stage 6.5 dual-SID candidate
fusion extension that combines Text-SID and train-only CF-SID item candidates.
The two streams are complementary: CF contributes behavior-only top-20 hits that
Text misses. With exact candidate fusion and the same non-leaky lightweight
reranker, dual-SID reaches 0.1452/0.0804 HR/NDCG@20 on Industrial and
0.1722/0.1011 on Office, exceeding the previous Stage 6 CS-SID best results of
0.1366/0.0784 and 0.1492/0.0884 respectively. We report this as the strongest
two-stream extension, while CS-SID remains the main single-SID result.
```

Table for final report:

| Category | Stage 6 CS-SID best | Dual-SID exact + rerank | Interpretation |
|---|---:|---:|---|
| Industrial_and_Scientific | 0.1366 / 0.0784 | 0.1452 / 0.0804 | dual-stream extension improves HR and NDCG |
| Office_Products | 0.1492 / 0.0884 | 0.1722 / 0.1011 | larger dual-stream gain, strong CF complementarity |

Suggested phrasing discipline:

- Say "strongest two-stream extension result".
- Do not say "CS-SID is replaced".
- Keep the original Stage 6 CS-SID table as the main single-SID result.
- Add the dual-SID table as an extension/ablation that shows candidate-pool
  complementarity.

## 11. Remaining Risks

1. Manual result sync.
   The raw AutoDL JSON reports are not present in the local workspace. The synced
   inspection log is enough for analysis, but final reproducibility would be
   stronger if the key reports were archived or regenerated.

2. File naming inconsistency.
   Industrial uses mixed candidate suffixes such as `v2` and `v1`; Office uses
   `vdual`. The metrics align, but a future summary script should normalize this.

3. RRF weight sensitivity.
   Lambda 0.6-0.8 collapses HR@20 back to the Text stream. This should be
   treated as a tuning and robustness issue, not hidden.

4. Sorting loss remains.
   Exact union recall is about 0.017-0.018 HR higher than final rerank HR on both
   categories. Better learned or calibrated reranking could still unlock gains.

5. p3 noise.
   Prefix expansion increases candidate breadth but reduces final exactness in
   dual-stream fusion. It should not be the main reported mode for this extension.

6. Test-set tuning risk.
   Rerank and fusion weights are fixed heuristics, but any future tuning should
   use validation data rather than repeatedly optimizing on test.

## 12. Future Work

- Add a full AutoDL archival command that copies only lightweight reports:
  `summary.csv`, `dual_fusion_report.json`, `schema_report.json`, and
  `rerank_report.json`.
- Extend `summarize_final_results.py` later with an optional
  `--include-dual-sid` flag that writes a separate dual-SID extension summary
  instead of mixing it into the CS-SID main table.
- Add full JSONL row-level identity checks for candidate files, not just row
  count and first-row preview, when producing the final archive.
- Try a validation-tuned reranker or learned ranker to reduce the remaining
  union-to-rerank sorting loss.
- Explore adaptive fusion weights by category, because Office benefits more from
  dual-stream fusion than Industrial.
