# S6-2 Smoke Audit and Efficient Validation Plan

Verdicts:

- smoke verdict: `GO_SMOKE_AUDIT`
- deterministic rerun design: ready
- formal validation verdict: `GO_FORMAL_VALIDATION`

No test data was read. No S5 artifact was modified. No CF Qwen or SASRec-SID Qwen generation was run.

## Evidence Integrity

Smoke bundle:

```text
incoming/s6_direct_sasrec_smoke/s6_direct_sasrec_valid_fit_k20_smoke_bundle.tar.gz
```

Bundle SHA256:

```text
5146a341db30f77a1b1c0ba76adb1683b3e99880598530a8e2982402adae4fe1
```

The bundle SHA256 file verified successfully. The bundle was extracted to an isolated temporary audit directory, and `smoke_output_files.sha256` verified all smoke output hashes:

| file | SHA256 |
|---|---|
| `candidates.jsonl` | `6d757d9fe825e2c330bed7b5a3fd8403566356625dfc6e6bde81f1a57f85e644` |
| `candidate_features.jsonl` | `1f321b86036541f04341ad1a56e5cdfae67e27599cea0f3afe042ab873877669` |
| `candidate_report.json` | `233316ea7e871f280085087baa33be0faa015547b63ec0fd51f62b65ed36f932` |

Required artifacts were present and non-empty:

- `candidates.jsonl`
- `candidate_features.jsonl`
- `candidate_report.json`
- `execution.log`
- `exit_code.txt`
- S6 config
- checkpoint compatibility report
- validation split manifest

Exit code: `0`.

Formal-v3 checkpoint SHA256 in the smoke report:

```text
577f51a9a539b9cd6307eea16c04db303691e349c0f32e245e44b8710cee94bd
```

`test_read=false` was recorded, and the audited smoke evidence did not contain a test CSV path.

## Smoke Data Audit

Independent audit script:

```text
scripts/audit_s6_direct_sasrec_smoke.py
```

Audit report:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/s6_2_smoke_audit_report.json
```

Verified:

- `candidates.jsonl` rows: `2660`
- `candidate_features.jsonl` rows: `2660`
- exact alignment with the `valid_fit` subset
- `user_id`, `target_item_id`, `history_item_id`, and `row_index` align with source valid rows
- every row has exactly 20 candidates
- every row has unique candidates
- every candidate belongs to the frozen item universe
- candidate ranks are exactly `1..20`
- score features are finite
- candidate scores are monotonically non-increasing by rank
- reported hashes match recomputed hashes
- reported metrics match recomputed metrics

Recomputed metrics:

| metric | value |
|---|---:|
| HR@1 | `0.08421052631578947` |
| HR@5 | `0.10676691729323308` |
| HR@10 | `0.12030075187969924` |
| HR@20 | `0.13458646616541353` |
| NDCG@1 | `0.08421052631578947` |
| NDCG@5 | `0.09606603138913469` |
| NDCG@10 | `0.10043363598437421` |
| NDCG@20 | `0.1040675961090237` |
| MRR | `0.09533325501939471` |
| target-in-pool | `358 / 2660` |
| target-in-pool rate | `0.13458646616541353` |

Safety counters:

- duplicate candidate rows: `0`
- invalid item count: `0`
- non-finite feature count: `0`
- rank violations: `0`
- monotonic score violations: `0`

## Determinism Design

The first smoke bundle is immutable and must not be overwritten.

The exporter now supports isolated output paths via:

- `--output-root`
- `--run-id`
- existing exact `--output-dir`

The exporter rejects existing non-empty output artifacts by default. No `allow-overwrite` path is introduced.

Deterministic comparison rules:

- `candidate_item_ids` ordering must be byte-identical, or semantically identical under canonical JSONL hashing.
- `candidate_features.jsonl` must match after excluding no fields; the current direct features contain no runtime-only values.
- `candidate_report.json` semantic fields must match.
- Runtime and memory fields are excluded from semantic determinism hashes:
  - wall-clock time
  - samples per second
  - CUDA peak allocated
  - CUDA peak reserved
  - model inference time
  - K-view derivation time

Isolated deterministic rerun command for AutoDL:

```bash
python3 export_sasrec_direct_candidates.py \
  --config configs/s6_cost_aware_aux/dev_config.json \
  --checkpoint-report results/s6_cost_aware_aux/Industrial_and_Scientific/s6_checkpoint_compatibility_report.json \
  --split valid_fit \
  --k 20 \
  --run-id deterministic_rerun_v1 \
  --device cuda
```

## Efficient K100 Design

Separate K20, K50, and K100 runs would repeat sequence encoding and full-item scoring. This is redundant.

The exporter now supports one-pass max-K inference:

```text
--k 100 --derive-prefix-views 20 50 100
```

Contract:

- run the model once per split with `max_k=100`;
- score the full item universe once per sample;
- write K100 candidates;
- derive K20 and K50 as exact prefixes of K100;
- write separate `candidate_report.json` files for K20, K50, and K100;
- record model inference time once and K-view derivation time separately.

Formal model-pass budget:

- `valid_fit`: one pass
- `valid_select`: one pass
- `valid_gate`: one pass

Not nine passes.

Efficient formal validation commands for AutoDL:

```bash
python3 export_sasrec_direct_candidates.py --split valid_fit --k 100 --derive-prefix-views 20 50 100 --run-id formal_v1 --device cuda
python3 export_sasrec_direct_candidates.py --split valid_select --k 100 --derive-prefix-views 20 50 100 --run-id formal_v1 --device cuda
python3 export_sasrec_direct_candidates.py --split valid_gate --k 100 --derive-prefix-views 20 50 100 --run-id formal_v1 --device cuda
```

Expected output inventory per split:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/direct_sasrec/<split>/formal_v1/k20/candidates.jsonl
results/s6_cost_aware_aux/Industrial_and_Scientific/direct_sasrec/<split>/formal_v1/k20/candidate_features.jsonl
results/s6_cost_aware_aux/Industrial_and_Scientific/direct_sasrec/<split>/formal_v1/k20/candidate_report.json
results/s6_cost_aware_aux/Industrial_and_Scientific/direct_sasrec/<split>/formal_v1/k50/...
results/s6_cost_aware_aux/Industrial_and_Scientific/direct_sasrec/<split>/formal_v1/k100/...
```

## Union Dry-Run Compatibility

Union dry-run entry point:

```text
merge_direct_sasrec_candidates.py --dry-run
```

The union contract preserves:

- `from_cf`
- `from_sasrec_direct`
- `cf_rank`
- `sasrec_direct_rank`
- reciprocal ranks through direct score sidecar/details
- overlap provenance
- direct SASRec score details

The merge rejects row misalignment, target mismatch, and existing output artifacts.

Formal union must wait until direct formal outputs exist. The dry-run command shape is:

```bash
python3 merge_direct_sasrec_candidates.py \
  --cf-candidates incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036/results/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/formal/valid/full_valid/candidates/candidates.jsonl \
  --direct-candidates results/s6_cost_aware_aux/Industrial_and_Scientific/direct_sasrec/valid_fit/formal_v1/k20/candidates.jsonl \
  --output-jsonl results/s6_cost_aware_aux/Industrial_and_Scientific/union/cf_direct_sasrec/formal_v1/valid_fit/k20/candidates.jsonl \
  --report results/s6_cost_aware_aux/Industrial_and_Scientific/union/cf_direct_sasrec/formal_v1/valid_fit/k20/candidate_report.json \
  --dry-run
```

## Frozen Ranker Projection Compatibility

The S5 frozen projected ranker may only receive features that are semantically supported by the frozen schema.

Allowed through the source-independent projection:

- rank/order-derived features;
- source-independent fusion rank features;
- source-independent popularity/history similarity features computed by the existing frozen feature code;
- bucket/prefix fields only when they have the same semantics as the frozen candidate details.

Not allowed for the frozen S5 ranker:

- direct SASRec raw score;
- z-score;
- score margin;
- top-1 margin;
- score-minus-top-K-mean.

Verdict: partial compatibility. CF + direct SASRec can be represented as item-level candidates and union provenance, but direct score features require a future S6 lightweight-ranker schema if they are to be used by a learned ranker. They must not be forced into the S5 frozen ranker.

## Cost Evidence

Smoke timing:

- wall clock: `14.893858574330807` seconds
- samples per second: `178.59710341176756`
- CUDA peak allocated: `13118464` bytes
- CUDA peak reserved: `25165824` bytes

Comparable validation-side SASRec-SID Qwen timing source:

- not yet found as a complete audited timing artifact.

The cost-ratio gate remains pending. Future measurement protocol:

1. use validation-only split;
2. measure SASRec-SID Qwen generation wall clock for the same rows and hardware;
3. record batch size, beam count, max tokens, checkpoint, GPU, and output hashes;
4. compute:

```text
cost_ratio = direct_SASRec_candidate_time / SASRec_SID_Qwen_generation_time
```

Do not use final-test timing for parameter selection.

## Final Gates

Retained gates:

- Candidate gate: retain at least 90% of dual-Qwen auxiliary target-in-pool uplift.
- Ranking gate: retain at least 80% of dual-Qwen HR@20 uplift and NDCG@20 uplift.
- Cost gate: direct SASRec generation time <= 20% of SASRec-SID Qwen generation time.
- Safety gate: invalid candidates = 0, deterministic semantic hashes, row alignment = 100%, CF top-20 hit preservation >= 97%.

Final S6-2 verdict:

```text
GO_FORMAL_VALIDATION
```
