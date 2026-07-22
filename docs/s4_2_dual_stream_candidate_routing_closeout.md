# S4-2 Dual-Stream Candidate Routing Closeout

## Scope

This closeout fixes only the S4 valid candidate pipeline wiring for the two frozen streams:

- baseline: `cf_k512_dedup`
- treatment: `sasrec_v3_k512_dedup`

It does not run GPU training, does not start formal evaluation, does not read `test`, does not rebuild SID artifacts, and does not change core SFT semantics.

## Root Cause

The previous `STAGE=candidates` runner generated a full command plan and selected the treatment candidate commands even when `TRAIN_STREAM=baseline` was requested. As a result, the AutoDL "baseline candidate" attempt actually used the `sasrec_v3_k512_dedup` checkpoint/data/result paths.

The same failed AutoDL attempt reached the end of valid generation but `evaluate.py` could not open:

`results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/smoke/valid/generation/predictions.json`

because the runner had not created the parent `generation/` directory before GPU inference. That run produced no `predictions.json`, did not execute candidate evaluation, and must not be counted as either baseline or treatment evidence.

## Fixed Contract

`STAGE=candidates` now strictly respects `TRAIN_STREAM`:

| Stream | Checkpoint | Input CSV/mapping | Result root |
|---|---|---|---|
| baseline | `outputs/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/smoke/valid/sft/final_checkpoint` | `data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/` | `results/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/smoke/valid/` |
| treatment | `outputs/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/smoke/valid/sft/final_checkpoint` | `data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/` | `results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/smoke/valid/` |

Each stream owns:

- `generation/predictions.json`
- `candidates/candidates.jsonl`
- `candidates/candidate_report.json`
- `summary/metrics_summary.json`
- `summary/candidate_prepare_manifest.json`
- `artifact_manifest.json`

S4-2B additionally separates candidate scopes:

| Scope | `CANDIDATE_ROW_LIMIT` | Directory |
|---|---:|---|
| bounded preflight | `8` | `preflight_rows8/` |
| full valid | `0` | `full_valid/` |

Bounded and full-valid artifacts never share paths.

Before GPU inference, the runner creates only parent directories, verifies each is writable through an independent `.write_probe`, and refuses unsafe artifact states. The write probe does not create `predictions.json`, `candidates.jsonl`, or `candidate_report.json`.

Candidate evaluation starts only after prediction row alignment is validated against the selected valid CSV.

Each run writes `artifact_manifest.json` with `complete=false` before GPU inference and updates it to `complete=true` only after generation, prediction row validation, exact candidate evaluation, and summary succeed.

The manifest schema is `s4_candidate_artifact_manifest.v1` and includes:

- `stream`
- `sid_version`
- `config_mode`
- `scope`
- `candidate_row_limit`
- `expected_prediction_rows`
- `actual_prediction_rows`
- `eval_csv_sha256`
- checkpoint fingerprint
- `item2sid_sha256`
- `sid2items_sha256`
- `num_beams`
- `max_new_tokens`
- `candidate_mode`
- `seed`
- `created_at`
- `complete`
- output artifact fingerprints

## Artifact Audit

`STAGE=audit-existing` classifies the selected stream/scope as:

- `missing`
- `empty`
- `malformed`
- `partial`
- `bounded`
- `full_valid`
- `wrong_stream`
- `wrong_sid_version`
- `wrong_scope`
- `legacy_unknown`

Unsafe states print quarantine/move commands and are not overwritten automatically.

`REUSE_COMPLETE=1` explicitly reuses complete matching artifacts. `OVERWRITE=1` is limited to an intentional rerun of the same stream and same scope. It must not be used across streams or to turn bounded output into full-valid output.

## Generation Termination Contract

The S4 generator uses:

`MAX_NEW_TOKENS=6`

This is derived from observed SID inputs:

- maximum SID token count: `4`
- newline token budget: `1`
- EOS token budget: `1`

The same value is used for baseline and treatment. The constrained logits processor also includes a minimal terminal defense: once EOS is already present in the generated suffix, it allows EOS continuation instead of querying the SID trie again. This preserves legal-prefix constraints before EOS and avoids warnings caused by generation continuing after a complete SID/newline/EOS sequence.

## Local Validation

Validated locally without GPU:

- S4 focused unit tests cover stream-specific candidate routing, isolated paths, output precreation, overwrite refusal, prediction row alignment, test-path rejection, token budget, and terminal EOS defense.
- `STAGE=parity DRY_RUN=0 CONFIG_MODE=smoke` refreshes the parity manifest with stream-specific commands.
- `STAGE=candidates TRAIN_STREAM=baseline DRY_RUN=1` emits only baseline generation/evaluation paths.
- `STAGE=candidates TRAIN_STREAM=treatment DRY_RUN=1` emits only treatment generation/evaluation paths.

## AutoDL Order

1. Environment preflight.
2. Parity.
3. Baseline bounded candidate preflight:
   `STAGE=candidates TRAIN_STREAM=baseline CANDIDATE_ROW_LIMIT=8 DRY_RUN=0`.
4. Treatment bounded candidate preflight:
   `STAGE=candidates TRAIN_STREAM=treatment CANDIDATE_ROW_LIMIT=8 DRY_RUN=0`.
5. Baseline full valid candidates:
   `STAGE=candidates TRAIN_STREAM=baseline CANDIDATE_ROW_LIMIT=0 DRY_RUN=0`.
6. Treatment full valid candidates:
   `STAGE=candidates TRAIN_STREAM=treatment CANDIDATE_ROW_LIMIT=0 DRY_RUN=0`.
7. Summarize each stream.

Use `STAGE=audit-existing` first if any artifact exists. Quarantine `legacy_unknown`, `wrong_stream`, `malformed`, or unsafe partial directories before rerun.

## STOP/GO

GO for AutoDL bounded candidate preflight after syncing the changed files. STOP for formal until both stream candidate preflights pass and full valid candidate artifacts are generated under their isolated roots.
