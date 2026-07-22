# S4-0 CF Valid Restore and Parity Closeout

## Scope

This closeout restores and validates only the S4 fair-experiment inputs for `Industrial_and_Scientific`.

Not performed:

- no GPU SFT;
- no test read or evaluation;
- no `sft.py` or `evaluate.py` changes;
- no SASRec retraining;
- no SASRec-SID regeneration;
- no Text + Strong Behavior fusion.

## Frozen Inputs

SASRec behavior embedding:

`data/Amazon/behavior_embeddings/sasrec/Industrial_and_Scientific/formal_v3_finite_maskfix_seed42/`

SASRec SID treatment:

`data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/`

CF SID baseline:

`data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/`

Explicit S4 base model for future fair SFT:

`/root/autodl-tmp/models/Qwen2.5-0.5B`

Local WSL cannot access that `/root/...` AutoDL path, so local parity records the explicit path but cannot hash it. AutoDL must verify the path exists before GPU smoke.

## Repository and Evidence Audit

Existing dirty worktree was preserved. The pre-existing tracked dirty file `scripts/run_stage6_candidate_rerank.sh` was not modified by this closeout.

SASRec v3 evidence audited:

- `artifact_manifest.json`: finite embedding, `num_items=3686`, `dimension=128`, `test_read=false`.
- `metrics_summary.json`: full valid candidate count `3686`, no unranked/nonfinite final valid samples.
- `s3_manifest.json`: `test_read=false`.
- `s3_static_quality_report.json`: `num_items=3686`, `full_unique=3686`, `full_collision_buckets=0`.

Treatment hashes:

- train CSV: `25ea0521c5bd34fe960b2dc5b8e488dfe80eefef9afb052a7b364af9319c9a53`
- valid CSV: `c707f2859b499acdebb2d5e25c0a129b350b8e0d997cfc291aa1986b54f79234`
- item2sid: `74d5d2cd21eef2d01260cdc6c3d790430045a97753064ad5a5c826756f425b35`

CF baseline evidence audited:

- `generation_report.json`: post-dedup `num_unique_sid=3686`, `collision_rate=0.0`.
- `rewrite_train_report.json`: train rows `36259`, SID parse success `1.0`.
- `rewrite_valid_report.json`: valid rows `4532`, target missing `0`, history missing `0`, SID parse success `1.0`.

CF hashes:

- train CSV: `f6b72dfdeac3cb55b53e643b61cb606eeb8b1ca758de1cdf67c2f839ed8b207e`
- valid CSV: `5234a104809d20585aa9ba91dfdf98d348d9fe6429a131a737a89d77c5780e9e`
- item2sid: `3f37cee168a0c08b9f6629ef0739e01a4c7fa90f7ad252dcd6a14a8a5e164e07`

## CF Valid Restore

`cf_k512_dedup/Industrial_and_Scientific/valid.csv` exists locally.

It was not overwritten. A temporary restore was generated under `/tmp` from:

- official valid CSV: `data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- frozen CF mapping: `data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/item2sid.json`

Result:

- existing CF valid SHA256: `5234a104809d20585aa9ba91dfdf98d348d9fe6429a131a737a89d77c5780e9e`
- temporary regenerated SHA256: `5234a104809d20585aa9ba91dfdf98d348d9fe6429a131a737a89d77c5780e9e`
- byte-identical: yes
- rows: `4532`
- target item mismatches: `0`
- history item mismatches: `0`
- missing mappings: `0`
- malformed SIDs: `0`

Therefore the CF valid restore is complete without reclustering and without test access.

## Historical CF Checkpoint Status

Historical documents record the shared intended base model as:

`/root/autodl-tmp/models/Qwen2.5-0.5B`

They also record prior SFT settings such as `epoch=1`, `candidate_mode=exact`, and `sid-only` task. However, the current local repository does not contain enough formal checkpoint metadata to prove that an existing historical CF checkpoint and the future SASRec-SID run share the same exact base model, sample budget, seed, LR, and training entry.

Classification:

`historical_nonfair_baseline`

Required fair path:

Train both CF-SID and SASRec-SID streams from the same explicit base model in isolated output directories before making any S4 comparison.

## Parity Matrix

| Field | CF baseline | SASRec treatment | Status |
|---|---:|---:|---|
| SID version | `cf_k512_dedup` | `sasrec_v3_k512_dedup` | intentionally different |
| train rows | 36259 | 36259 | match |
| valid rows | 4532 | 4532 | match |
| item count | 3686 | 3686 | match |
| full unique SID | 3686 | 3686 | match |
| full collision | 0 | 0 | match |
| test read | false | false | match |
| candidate mode | exact | exact | match |
| explicit base model | `/root/autodl-tmp/models/Qwen2.5-0.5B` | `/root/autodl-tmp/models/Qwen2.5-0.5B` | configured |
| local base model hash | unavailable | unavailable | must verify on AutoDL |
| SID token string set | differs | differs | train both from same base |
| SFT output namespace | `.../cf_k512_dedup/...` | `.../sasrec_v3_k512_dedup/...` | isolated |

## Parity Manifest

Machine-readable parity manifest:

`results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/smoke/valid/s4_parity_manifest.json`

Closeout report:

`results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/smoke/valid/s4_0_cf_valid_restore_and_parity_report.json`

## Runner Hardening

The S4 runner was minimally hardened during closeout:

- inaccessible local base model paths are recorded instead of crashing fingerprint generation;
- baseline and treatment training commands now use separate output directories.
- S4-1 hardening adds a runtime environment preflight before non-dry-run `torchrun`.
- smoke now uses bounded `SAMPLE=2000` instead of `sample=-1`.
- formal uses predefined `SAMPLE=30000`.
- valid generation commands use explicit positive `MAX_NEW_TOKENS`; S4-2 freezes the value to `6` for 4-level SID plus newline plus EOS.

This does not change SFT semantics, evaluation semantics, SID construction, or candidate generation logic.

## Local Validation

Commands run:

```bash
python3 -m unittest tests.test_s4_sasrec_sid_valid_pipeline tests.test_sasrec_v3_behavior_sid
python3 -m compileall -q scripts tests
find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n
git diff --check
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B ALLOW_MISSING_BASE_MODEL=1 STAGE=parity DRY_RUN=0 CONFIG_MODE=smoke bash scripts/run_s4_sasrec_sid_valid.sh
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B ALLOW_MISSING_BASE_MODEL=1 STAGE=train DRY_RUN=1 CONFIG_MODE=smoke bash scripts/run_s4_sasrec_sid_valid.sh
```

Expected local limitation:

`/root/autodl-tmp/models/Qwen2.5-0.5B` is an AutoDL path and is not locally hashable from WSL.

## Local-to-AutoDL Sync Manifest

Sync the minimum changed files:

- `scripts/s4_sasrec_sid_valid_pipeline.py`
- `scripts/run_s4_sasrec_sid_valid.sh`
- `tests/test_s4_sasrec_sid_valid_pipeline.py`
- `docs/s4_sasrec_sid_valid_experiment.md`
- `docs/s4_0_cf_valid_restore_and_parity_closeout.md`
- `results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/smoke/valid/s4_parity_manifest.json`
- `results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/smoke/valid/s4_0_cf_valid_restore_and_parity_report.json`
- `data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/valid.csv` if AutoDL lacks it.

Do not sync broad `results/`, `outputs/`, or `data/` directories wholesale.

## Next AutoDL Commands

1. CPU environment check:

```bash
python3 --version
test -d /root/autodl-tmp/models/Qwen2.5-0.5B
```

Success flag: base model path exists.

2. CPU parity audit:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=parity \
DRY_RUN=0 \
CONFIG_MODE=smoke \
bash scripts/run_s4_sasrec_sid_valid.sh
```

Success flag: parity manifest written and `test_read=false`.

3. CPU unit tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/minionerec_pycache \
python3 -m unittest tests.test_s4_sasrec_sid_valid_pipeline
```

Success flag: tests pass.

4. CPU runtime preflight:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=env-preflight \
DRY_RUN=1 \
CONFIG_MODE=smoke \
bash scripts/run_s4_sasrec_sid_valid.sh
```

Success flag: Python, `torchrun`, CUDA, Transformers, Accelerate, PEFT, `AutoModelForCausalLM`, `AutoTokenizer`, and `GenerationMixin` all import cleanly.

5. CPU train DRY_RUN:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=train \
DRY_RUN=1 \
CONFIG_MODE=smoke \
bash scripts/run_s4_sasrec_sid_valid.sh
```

Success flag: prints isolated CF and SASRec train commands; no checkpoint/candidate written.

6. GPU smoke, baseline CF:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=train \
TRAIN_STREAM=baseline \
DRY_RUN=0 \
CONFIG_MODE=smoke \
bash scripts/run_s4_sasrec_sid_valid.sh
```

Success flag: baseline checkpoint saved under the isolated `cf_k512_dedup` output namespace; no test read.

7. GPU smoke, SASRec treatment:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=train \
TRAIN_STREAM=treatment \
DRY_RUN=0 \
CONFIG_MODE=smoke \
bash scripts/run_s4_sasrec_sid_valid.sh
```

Success flag: checkpoint saved; no test read.

8. GPU formal:

Repeat both streams with `CONFIG_MODE=formal` only after smoke passes.

9. Valid exact candidate generation:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=candidates \
DRY_RUN=0 \
CONFIG_MODE=formal \
bash scripts/run_s4_sasrec_sid_valid.sh
```

Success flag: 4532 valid predictions and exact candidate report.

## Stop/Go Decision

GO to AutoDL S4-1 smoke preparation after AutoDL verifies the explicit base model path exists.

STOP for any fair comparison claim until both CF-SID and SASRec-SID are trained from the same explicit base model under the S4 isolated output layout.
