# S6 Direct SASRec Validation Protocol

Status: validation protocol frozen; checkpoint compatibility passed; direct retrieval implementation is ready for validation-only smoke.

## Scope

S6 is a validation-only research stage for replacing the SASRec-SID Qwen auxiliary stream with direct SASRec item retrieval. It must not read or use final-test data, and it must not change any frozen S5 parameter.

## Development Split

The S6 development split is generated from:

```text
data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv
```

Manifest:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/s6_validation_split_manifest.json
```

Split contract:

- split key: `user_id`
- seed: `42`
- assignment: `sha256('s6_validation_split.v1:{seed}:{user_id}') mod 100`
- `valid_fit`: bucket `< 60`
- `valid_select`: `60 <= bucket < 80`
- `valid_gate`: `80 <= bucket < 100`
- same user never appears in multiple subsets
- original row order is preserved within each subset
- no original CSV is modified
- `test_read=false`

Observed split counts:

| split | rows | users | SHA256 |
|---|---:|---:|---|
| valid_fit | 2660 | 1188 | `59bce8bd06c23fb4c10648970bcec6400cf949ff07a1302f654a57822074524d` |
| valid_select | 963 | 413 | `54a2fb4fd7983b59845af1eccaee5128995409f338e45abc9be57f11d81b9520` |
| valid_gate | 909 | 410 | `78249b22f11698f3f1887108f0a701732f766a243be0b3c00d8dd061a801f8ab` |

All user-overlap checks are zero.

## Required Direct SASRec Computation

Direct retrieval must use the full SASRec sequence-model checkpoint. The exported item embedding matrix alone is not sufficient.

Required computation after checkpoint restoration:

```text
history_ids = parse(history_item_id)
internal_ids = [row_index[item_id] + 1 for item_id in history_ids]
input_ids = right_pad_or_truncate(internal_ids, max_seq_len=10, padding_id=0)
last_position = length(input_ids != 0) - 1
hidden = MiniSASRec.encode(input_ids)
h_u = hidden[last_position]
scores = h_u dot item_embedding.weight[1:].T
topK = deterministic top-K over canonical item_order
```

The output must be deterministic and finite. It must preserve row alignment with the source validation subset.

## Candidate Budgets

The only S6 development candidate budgets are:

```text
K = 20, 50, 100
```

No larger search space is allowed without a new protocol update.

## Candidate Output Contract

Direct SASRec candidate output should preserve the existing candidate JSONL contract:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/direct_sasrec/<split>/candidates.jsonl
results/s6_cost_aware_aux/Industrial_and_Scientific/direct_sasrec/<split>/candidate_features.jsonl
results/s6_cost_aware_aux/Industrial_and_Scientific/direct_sasrec/<split>/candidate_report.json
```

Required row fields:

- `row_index`
- `user_id`
- `target_item_id`
- `history_item_id`
- `candidate_item_ids`
- `candidate_details`

Required direct-SASRec detail or sidecar fields:

- `item_id`
- `sasrec_direct_rank`
- `sasrec_direct_score`
- `sasrec_direct_reciprocal_rank`
- `sasrec_direct_zscore`
- `sasrec_direct_top1_margin`
- `sasrec_direct_score_minus_topk_mean`

Direct score features should go into `candidate_features.jsonl` if they cannot fit the existing `candidate_details` contract without ambiguity.

## Union Contract

CF + direct SASRec union output should preserve duplicate provenance:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/union/<config_id>/<split>/candidates.jsonl
results/s6_cost_aware_aux/Industrial_and_Scientific/union/<config_id>/<split>/candidate_report.json
```

Required per-candidate provenance:

- `sample_id`
- `user_id`
- `target_item_id`
- `candidate_item_id`
- `from_cf`
- `from_sasrec_direct`
- `cf_rank`
- `sasrec_direct_rank`
- `reciprocal_rank` features
- direct SASRec score features when available
- overlap indicator

## Evaluation Policies

E0: CF-only baseline from existing validation artifacts.

E1: frozen S5 dual-Qwen validation upper bound, reused from existing artifacts only. Do not rerun.

E2: CF + direct SASRec with rank-only RRF.

E3: CF + direct SASRec with the existing frozen projected ranker only if schema compatibility is confirmed. If direct score features are needed, train a future lightweight S6 ranker on validation splits only; do not modify the S5 frozen ranker.

## Cost Instrumentation

Future direct retrieval reports must include:

- total wall-clock time;
- samples per second;
- candidate generation time;
- fusion time;
- ranker time;
- CUDA peak allocated memory;
- CUDA peak reserved memory;
- CPU RSS when practical;
- batch latency p50/p95 when practical;
- device and software environment.

Derived quantities:

```text
cost_ratio =
  direct_SASRec_candidate_time / SASRec_SID_Qwen_generation_time

uplift_retention_hr =
  (HR_direct - HR_cf) / (HR_dual_qwen - HR_cf)

uplift_retention_ndcg =
  (NDCG_direct - NDCG_cf) / (NDCG_dual_qwen - NDCG_cf)
```

Unavailable timing values must be `null` with an explicit missing reason. Do not fabricate timing numbers.

## Acceptance Gates

Candidate recall gate:

- retain at least 90% of the auxiliary target-in-pool uplift of the dual-Qwen validation upper bound.

Ranking gate:

- retain at least 80% of dual-Qwen HR@20 uplift;
- retain at least 80% of dual-Qwen NDCG@20 uplift.

Cost gate:

- direct SASRec candidate generation time no more than 20% of SASRec-SID Qwen generation time;
- preferred target: no more than 10%.

Safety gate:

- invalid candidate item count = 0;
- row alignment = 100%;
- deterministic rerun hashes match;
- CF top-20 hit preservation at least 97%.

Allowed verdicts:

- `GO_DIRECT_SASREC`
- `GO_DIRECT_SASREC_WITH_NEW_RANKER`
- `REVISE_DIRECT_SASREC`
- `KEEP_DUAL_QWEN_UPPER_BOUND_ONLY`
- `BLOCKED_MISSING_FULL_SASREC_CHECKPOINT`

## Command Plan

Current local safe command:

```bash
python3 scripts/build_s6_validation_split.py
```

Checkpoint compatibility report:

```text
results/s6_cost_aware_aux/Industrial_and_Scientific/s6_checkpoint_compatibility_report.json
```

Dry-run command:

```bash
python3 export_sasrec_direct_candidates.py \
  --config configs/s6_cost_aware_aux/dev_config.json \
  --checkpoint-report results/s6_cost_aware_aux/Industrial_and_Scientific/s6_checkpoint_compatibility_report.json \
  --split valid_fit \
  --k 20 \
  --dry-run
```

CPU synthetic/local command:

```bash
python3 -m unittest tests.test_s6_checkpoint_audit tests.test_s6_validation_split tests.test_s6_direct_sasrec_candidates -v
```

GPU smoke command to run on AutoDL:

```bash
python3 export_sasrec_direct_candidates.py \
  --config configs/s6_cost_aware_aux/dev_config.json \
  --checkpoint-report results/s6_cost_aware_aux/Industrial_and_Scientific/s6_checkpoint_compatibility_report.json \
  --split valid_fit \
  --k 20 \
  --device cuda
```

Efficient formal validation commands:

```bash
python3 export_sasrec_direct_candidates.py --split valid_fit --k 100 --derive-prefix-views 20 50 100 --run-id formal_v1 --device cuda
python3 export_sasrec_direct_candidates.py --split valid_select --k 100 --derive-prefix-views 20 50 100 --run-id formal_v1 --device cuda
python3 export_sasrec_direct_candidates.py --split valid_gate --k 100 --derive-prefix-views 20 50 100 --run-id formal_v1 --device cuda
```

These commands are validation-only and must not read final-test data.
