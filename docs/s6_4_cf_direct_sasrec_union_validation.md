# S6-4 CF + Direct-SASRec Union Validation

Verdict: **GO_FROZEN_RANKER_VALIDATION**

This stage evaluates candidate-level complementarity between frozen CF-SID candidates and formal direct-SASRec item retrieval. It does not run CF Qwen, SASRec-SID Qwen, direct SASRec inference, ranker training, frozen-ranker scoring, or any test-set workload.

Machine-readable report:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/s6_4_union_validation_report.json
```

Run id:

```text
union_v1
```

## Evidence Integrity

Frozen inputs:

| Input | Evidence |
|---|---|
| S6 split manifest | `results/s6_cost_aware_aux/Industrial_and_Scientific/s6_validation_split_manifest.json` |
| frozen CF full-valid candidates | `incoming/s4_formal_full_valid/.../cf_k512_dedup/.../full_valid/candidates/candidates.jsonl` |
| frozen CF SHA256 | `eaef029b5679e525c06e51fabbac36bc6f029c9da9165d641634cbf945cd5281` |
| direct-SASRec formal bundle | `incoming/s6_formal_direct_sasrec/s6_formal_direct_sasrec_formal_v1_bundle.tar.gz` |
| direct-SASRec checkpoint SHA256 | `577f51a9a539b9cd6307eea16c04db303691e349c0f32e245e44b8710cee94bd` |
| dual-Qwen comparison candidates | frozen S4 formal CF and SASRec-SID full-valid candidate artifacts |

Checks passed:

- CF full-valid candidate SHA256 matches the frozen value.
- Direct-SASRec candidate hashes match their formal-v1 reports.
- Direct reports have `test_read=false`.
- S4 candidate reports do not record `test_read=true`; S4 summaries record validation split.
- No test path was accepted by the S6-4 runner.
- No model inference or ranker execution was performed.

## CF Split-View Audit

Deterministic CF split views were created by filtering frozen full-valid CF candidates by exact S6 split row ids, preserving S6 split source-row order.

| Split | Rows | Output |
|---|---:|---|
| valid_fit | 2660 | `results/s6_cost_aware_aux/Industrial_and_Scientific/cf_split_views/union_v1/valid_fit/candidates.jsonl` |
| valid_select | 963 | `results/s6_cost_aware_aux/Industrial_and_Scientific/cf_split_views/union_v1/valid_select/candidates.jsonl` |
| valid_gate | 909 | `results/s6_cost_aware_aux/Industrial_and_Scientific/cf_split_views/union_v1/valid_gate/candidates.jsonl` |

For every split, alignment by `row_index`, `target_item_id`, and normalized `history_item_id` passed. Candidate order and candidate contents were not changed.

## Union Construction

For each split and K in `{20, 50, 100}`, the union view merges:

```text
frozen CF split candidates + formal_v1 direct-SASRec candidates
```

Deterministic ordering policy:

1. Keep all CF candidates in original CF order.
2. Append direct-SASRec-only candidates in direct-SASRec rank order.

Each union candidate preserves:

- `item_id`;
- `from_cf`;
- `from_sasrec_direct`;
- `cf_rank_0_based`;
- `cf_reciprocal_rank`;
- `sasrec_direct_rank_0_based`;
- `sasrec_direct_reciprocal_rank`;
- `overlap`;
- copied CF detail;
- copied direct-SASRec score/detail sidecar;
- `union_rank_0_based`.

Union order is not treated as a trained final recommender ranking. The primary metric here is target-in-pool recall and complementarity.

## Split-By-K Union Table

| Split | K | CF targets | Direct targets | Union targets | CF-only | Direct-only | Both | Uplift vs CF | Retention vs dual-Qwen | Avg union size | Overlap candidates | Added candidates | Candidate noise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valid_fit | 20 | 386 | 358 | 526 | 168 | 140 | 218 | 140 | 2.029 | 67.243 | 7334 | 45866 | 0.9969 |
| valid_fit | 50 | 386 | 437 | 586 | 149 | 200 | 237 | 200 | 2.899 | 94.970 | 13381 | 119619 | 0.9983 |
| valid_fit | 100 | 386 | 529 | 656 | 127 | 270 | 259 | 270 | 3.913 | 142.264 | 20578 | 245422 | 0.9989 |
| valid_select | 20 | 128 | 123 | 182 | 59 | 54 | 69 | 54 | 2.160 | 67.388 | 2515 | 16745 | 0.9968 |
| valid_select | 50 | 128 | 146 | 199 | 53 | 71 | 75 | 71 | 2.840 | 95.332 | 4495 | 43655 | 0.9984 |
| valid_select | 100 | 128 | 180 | 230 | 50 | 102 | 78 | 102 | 4.080 | 142.699 | 7031 | 89269 | 0.9989 |
| valid_gate | 20 | 115 | 104 | 157 | 53 | 42 | 62 | 42 | 1.826 | 67.507 | 2266 | 15914 | 0.9974 |
| valid_gate | 50 | 115 | 134 | 179 | 45 | 64 | 70 | 64 | 2.783 | 95.523 | 4070 | 41380 | 0.9985 |
| valid_gate | 100 | 115 | 163 | 204 | 41 | 89 | 74 | 89 | 3.870 | 142.955 | 6404 | 84496 | 0.9989 |

Safety checks across all union views:

| Check | Result |
|---|---:|
| CF target preservation | 100% |
| duplicate candidates after union | 0 |
| candidate provenance failures | 0 |
| row/target/history alignment failures | 0 |
| invalid candidate count | 0 |

## Dual-Qwen Uplift-Retention Comparison

The dual-Qwen validation upper bound was reconstructed from frozen S4 formal full-valid CF and SASRec-SID candidate artifacts. It is split-compatible because both files share validation `row_index`, target, and normalized history.

Full-valid sanity comparison:

| Metric | Count | Rate |
|---|---:|---:|
| CF target-in-pool | 629 | 0.1388 |
| SASRec-SID target-in-pool | 397 | 0.0876 |
| CF + SASRec-SID union target-in-pool | 746 | 0.1646 |
| dual-Qwen auxiliary uplift | 117 | 0.0258 |

For S6 K selection, the split-level comparison is used:

```text
direct auxiliary uplift = direct-union target count - CF target count
dual-Qwen auxiliary uplift = dual-Qwen-union target count - CF target count
retention = direct auxiliary uplift / dual-Qwen auxiliary uplift
```

The candidate gate is retention >= 90%.

## K Selection

Predeclared policy:

1. Use only `valid_fit` and `valid_select`.
2. Choose the smallest K that satisfies retention >= 90% on both splits.
3. Inspect `valid_gate` only after selection as a direction confirmation.

Selection evidence:

| K | valid_fit retention | valid_select retention | Passes |
|---:|---:|---:|---|
| 20 | 2.029 | 2.160 | yes |
| 50 | 2.899 | 2.840 | yes |
| 100 | 3.913 | 4.080 | yes |

Selected K:

```text
K = 20
```

Rationale: K20 is the smallest budget satisfying the candidate gate on both selection splits. K50 and K100 add more targets but also add much larger candidate pools and higher candidate noise. Since S6-4 is only a candidate-level gate and not a ranker-quality gate, the predeclared smallest-passing-K rule freezes K20 for the next frozen-ranker validation stage.

`valid_gate` confirmation after selection:

| Selected K | valid_gate retention | Gate pass |
|---:|---:|---|
| 20 | 1.826 | yes |

`valid_gate` was not used to revise the selected K.

## Candidate-Noise Analysis

Direct retrieval is complementary but noisy:

- K20 adds 140 direct-only targets on `valid_fit` and 54 on `valid_select`.
- K20 average union size is about 67 candidates, versus 50 CF candidates.
- K50/K100 increase target recall but also expand average union size to about 95 and 142 candidates.
- Candidate-noise ratios are high because most added candidates are non-target candidates; this is expected for candidate recall expansion and must be handled by the next ranker gate.

This supports K20 as the smallest sufficient candidate expansion before frozen-ranker validation.

## Cost-Gate Status

Frozen direct-SASRec validation runtime:

| Quantity | Value |
|---|---:|
| model inference seconds | 24.967018 |
| formal wall seconds | 34.233973 |
| peak CUDA allocated bytes | 13118464 |

Comparable validation-side SASRec-SID Qwen runtime evidence was not found in the available frozen validation artifacts. The formal cost ratio remains pending.

Future measurement protocol:

1. Use validation-side frozen S4/SASRec-SID generation logs only.
2. Do not use final-test runtime for validation-side K selection.
3. Compare full validation sample count, wall time, model/inference time where available, GPU type, and decoding budget.
4. Keep cost evidence separate from candidate-quality evidence.

## Output Artifacts

Created S6-4 artifacts:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/cf_split_views/union_v1/<split>/candidates.jsonl
results/s6_cost_aware_aux/Industrial_and_Scientific/cf_split_views/union_v1/<split>/candidate_report.json
results/s6_cost_aware_aux/Industrial_and_Scientific/union/union_v1/<split>/k<K>/candidates.jsonl
results/s6_cost_aware_aux/Industrial_and_Scientific/union/union_v1/<split>/k<K>/candidate_report.json
results/s6_cost_aware_aux/Industrial_and_Scientific/s6_4_union_validation_report.json
```

No S4, S5, direct formal-v1, CF full-valid, SASRec-SID full-valid, or test artifact was modified.

## Commands

Dry-run:

```bash
cd /root/autodl-tmp/projects/MiniOneRec
python3 scripts/s6_cf_direct_union_validation.py --dry-run
```

Formal validation-only union command:

```bash
cd /root/autodl-tmp/projects/MiniOneRec
python3 scripts/s6_cf_direct_union_validation.py
```

Focused tests:

```bash
python3 -m unittest tests.test_s6_cf_direct_union_validation
```

## Rollback

S6-4 created only derived S6 artifacts. To remove them before a clean rerun:

```bash
rm -rf results/s6_cost_aware_aux/Industrial_and_Scientific/cf_split_views/union_v1
rm -rf results/s6_cost_aware_aux/Industrial_and_Scientific/union/union_v1
rm -f results/s6_cost_aware_aux/Industrial_and_Scientific/s6_4_union_validation_report.json
```

Do not remove or modify the frozen S4/S5/S6 input artifacts.

## Next Gate

Proceed to frozen-ranker validation using:

```text
K = 20
```

The next stage may evaluate whether the frozen ranker can convert K20 candidate-pool recall into ranking quality without using test data for tuning.
