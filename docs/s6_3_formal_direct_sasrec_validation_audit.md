# S6-3 Formal Direct-SASRec Validation Audit

Verdict: **GO_UNION_VALIDATION**

This is a read-only audit of the formal direct-SASRec validation bundle:

```text
incoming/s6_formal_direct_sasrec/s6_formal_direct_sasrec_formal_v1_bundle.tar.gz
incoming/s6_formal_direct_sasrec/s6_formal_direct_sasrec_formal_v1_bundle.tar.gz.sha256
```

No test data was read. No direct inference, union generation, frozen-ranker run, training, commit, merge, tag, or artifact overwrite was performed.

The machine-readable audit report is:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/s6_3_formal_direct_sasrec_audit_report.json
```

## Evidence Integrity

Outer bundle SHA256 verification passed.

Internal `formal_v1_files.sha256` verification passed for all bundled files, including:

- formal direct-SASRec K20/K50/K100 candidates, features, and reports for `valid_fit`, `valid_select`, and `valid_gate`;
- split manifest;
- checkpoint compatibility report;
- execution logs;
- formal launcher;
- exporter and merge scripts.

Exit-code audit:

| Stage | Exit code |
|---|---:|
| overall | 0 |
| valid_fit | 0 |
| valid_select | 0 |
| valid_gate | 0 |

The formal run used validation-only inputs and each candidate report records `test_read=false`.

## Row And Candidate Safety

All formal views passed row and candidate safety checks.

| Check | Result |
|---|---:|
| duplicate row ids | 0 |
| duplicate candidates within a row | 0 |
| invalid candidate item ids | 0 |
| non-finite direct-SASRec feature values | 0 |
| rank/order violations | 0 |
| score monotonicity violations | 0 |
| row alignment failures | 0 |
| candidate-feature alignment failures | 0 |
| metric mismatches after recomputation | 0 |
| artifact hash mismatches | 0 |

The audit normalized `history_item_id` representation before comparison because the source validation CSV stores histories as a bracketed string while candidate artifacts store histories as JSON lists.

## Recomputed Metrics

Metrics below were independently recomputed from the formal `candidates.jsonl` artifacts.

| Split | K | Samples | HR@1 | HR@5 | HR@10 | HR@20 | HR@50 | HR@100 | NDCG@20 | NDCG@50 | NDCG@100 | MRR | Target in pool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valid_fit | 20 | 2660 | 0.084211 | 0.106767 | 0.120301 | 0.134586 | n/a | n/a | 0.104068 | n/a | n/a | 0.095333 | 358 |
| valid_fit | 50 | 2660 | 0.084211 | 0.106767 | 0.120301 | 0.134586 | 0.164286 | n/a | 0.104068 | 0.109810 | n/a | 0.096194 | 437 |
| valid_fit | 100 | 2660 | 0.084211 | 0.106767 | 0.120301 | 0.134586 | 0.164286 | 0.198872 | 0.104068 | 0.109810 | 0.115358 | 0.096664 | 529 |
| valid_select | 20 | 963 | 0.074766 | 0.095535 | 0.110073 | 0.127726 | n/a | n/a | 0.094707 | n/a | n/a | 0.085376 | 123 |
| valid_select | 50 | 963 | 0.074766 | 0.095535 | 0.110073 | 0.127726 | 0.151610 | n/a | 0.094707 | 0.099453 | n/a | 0.086144 | 146 |
| valid_select | 100 | 963 | 0.074766 | 0.095535 | 0.110073 | 0.127726 | 0.151610 | 0.186916 | 0.094707 | 0.099453 | 0.105165 | 0.086643 | 180 |
| valid_gate | 20 | 909 | 0.068207 | 0.096810 | 0.104510 | 0.114411 | n/a | n/a | 0.089078 | n/a | n/a | 0.081472 | 104 |
| valid_gate | 50 | 909 | 0.068207 | 0.096810 | 0.104510 | 0.114411 | 0.147415 | n/a | 0.089078 | 0.095541 | n/a | 0.082474 | 134 |
| valid_gate | 100 | 909 | 0.068207 | 0.096810 | 0.104510 | 0.114411 | 0.147415 | 0.179318 | 0.089078 | 0.095541 | 0.100725 | 0.082933 | 163 |

Aggregate validation-side direct-SASRec metrics:

| K | Samples | HR@20 | NDCG@20 | HR@50 | NDCG@50 | HR@100 | NDCG@100 | MRR | Target in pool |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 4532 | 0.129082 | 0.099072 | n/a | n/a | n/a | n/a | 0.090437 | 585 |
| 50 | 4532 | 0.129082 | 0.099072 | 0.158208 | 0.104747 | n/a | n/a | 0.091306 | 717 |
| 100 | 4532 | 0.129082 | 0.099072 | 0.158208 | 0.104747 | 0.192410 | 0.110257 | 0.091781 | 872 |

## Prefix Consistency

K20 and K50 views are exact prefixes of K100 for every audited split:

- candidate item ids match the K100 prefix;
- candidate details match the K100 prefix;
- candidate feature rows match the K100 prefix.

This confirms the efficient formal protocol: one K100 model pass per split, with K20/K50/K100 views derived from that pass.

## Split Stability

For HR@20 and NDCG@20, all K values share the same top-20 prefix, so split ranges are identical across K:

| K | HR@20 min | HR@20 max | HR@20 range | NDCG@20 min | NDCG@20 max | NDCG@20 range |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.114411 | 0.134586 | 0.020175 | 0.089078 | 0.104068 | 0.014990 |
| 50 | 0.114411 | 0.134586 | 0.020175 | 0.089078 | 0.104068 | 0.014990 |
| 100 | 0.114411 | 0.134586 | 0.020175 | 0.089078 | 0.104068 | 0.014990 |

Target-in-pool rates increase with K on every split:

| K | min | max | range |
|---:|---:|---:|---:|
| 20 | 0.114411 | 0.134586 | 0.020175 |
| 50 | 0.147415 | 0.164286 | 0.016871 |
| 100 | 0.179318 | 0.198872 | 0.019554 |

## Candidate-Budget Recommendation

Recommendation: carry **K20, K50, and K100** into CF + direct-SASRec union validation.

Rationale:

- K20 already fixes the direct top-20 ranking view.
- K50 adds 132 validation targets over K20.
- K100 adds another 155 validation targets over K50.
- K50/K100 are derived from the same K100 model pass in the formal protocol, so they expose recall headroom without additional model inference.
- The downstream union/ranking gate should decide whether the extra recall is worth candidate noise.

No final test budget should be selected from test results.

## Runtime Aggregation

Runtime aggregation counts each split's K100 report once and does not double-count derived K20/K50 views.

| Quantity | Value |
|---|---:|
| samples | 4532 |
| model inference seconds | 24.967018 |
| formal wall seconds | 34.233973 |
| derivation/write seconds | 9.266954 |
| samples/model-second | 181.519472 |
| samples/wall-second | 132.383116 |
| max CUDA allocated bytes | 13118464 |
| max CUDA reserved bytes | 25165824 |

Validation-side direct-SASRec runtime evidence is available. Comparable frozen SASRec-SID Qwen validation runtime evidence was not present in this formal direct bundle, so the final cost gate remains pending until the union/cost closeout can compare against an audited Qwen-side runtime source.

## Frozen CF Validation Alignment

Frozen CF formal full-valid candidates were found at:

```text
incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036/results/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/formal/valid/full_valid/candidates/candidates.jsonl
```

SHA256:

```text
eaef029b5679e525c06e51fabbac36bc6f029c9da9165d641634cbf945cd5281
```

The file contains 4532 unique validation rows. For every S6 split, frozen CF candidates align by `row_index`, `target_item_id`, and normalized `history_item_id`.

Union validation should derive per-split CF views from this frozen full-valid file by filtering row ids. It must not rerun CF generation.

## Union Input Inventory

Direct-SASRec inputs:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/direct_sasrec/<split>/formal_v1/k<K>/candidates.jsonl
```

Splits:

| Split | Rows |
|---|---:|
| valid_fit | 2660 |
| valid_select | 963 |
| valid_gate | 909 |

Budgets:

```text
K=20, 50, 100
```

Frozen CF input:

```text
incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036/results/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/formal/valid/full_valid/candidates/candidates.jsonl
```

## Future Union Commands

The next phase should materialize filtered CF split views, then call the existing merge tool. Example command template:

```bash
cd /root/autodl-tmp/projects/MiniOneRec

CF_FULL="incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036/results/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/formal/valid/full_valid/candidates/candidates.jsonl"

for SPLIT in valid_fit valid_select valid_gate; do
  for K in 20 50 100; do
    DIRECT="results/s6_cost_aware_aux/Industrial_and_Scientific/direct_sasrec/${SPLIT}/formal_v1/k${K}/candidates.jsonl"
    CF_SPLIT="results/s6_cost_aware_aux/Industrial_and_Scientific/cf_filtered/${SPLIT}/formal_v1/candidates.jsonl"
    OUT="results/s6_cost_aware_aux/Industrial_and_Scientific/union_cf_direct_sasrec/${SPLIT}/formal_v1/k${K}/candidates.jsonl"
    REPORT="results/s6_cost_aware_aux/Industrial_and_Scientific/union_cf_direct_sasrec/${SPLIT}/formal_v1/k${K}/union_report.json"

    python3 - "${CF_FULL}" "${DIRECT}" "${CF_SPLIT}" <<'PY'
import json
import sys
from pathlib import Path

cf_full = Path(sys.argv[1])
direct = Path(sys.argv[2])
out = Path(sys.argv[3])
direct_ids = set()
with open(direct, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            direct_ids.add(str(row.get("row_index", row.get("sample_id", ""))))
out.parent.mkdir(parents=True, exist_ok=True)
with open(cf_full, "r", encoding="utf-8") as src, open(out, "w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        row = json.loads(line)
        rid = str(row.get("row_index", row.get("sample_id", "")))
        if rid in direct_ids:
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
PY

    python3 merge_direct_sasrec_candidates.py \
      --cf-candidates "${CF_SPLIT}" \
      --direct-candidates "${DIRECT}" \
      --output-jsonl "${OUT}" \
      --report "${REPORT}"
  done
done
```

These commands are for the next gate only. They were not run in this audit.

## Recommended Next Gate

Proceed to CF + direct-SASRec union validation with:

1. frozen CF full-valid candidates filtered by split row ids;
2. formal direct-SASRec K20/K50/K100 candidates;
3. no test access;
4. no ranker training from test;
5. validation-side union metrics and cost comparison audited before any future final-test policy is discussed.

Final S6-3 verdict: **GO_UNION_VALIDATION**.
