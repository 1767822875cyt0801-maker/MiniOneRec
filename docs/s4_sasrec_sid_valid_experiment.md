# S4 SASRec Strong Behavior-SID Valid-Only Experiment Flow

## Scope

S4 compares generative recommendation behavior for:

- Baseline SID: `cf_k512_dedup`
- Treatment SID: `sasrec_v3_k512_dedup`
- Category: `Industrial_and_Scientific`
- Split: `valid`

The comparison is valid only when both streams are trained from the same explicit base model or the same frozen pre-SFT checkpoint. Continuing treatment training from a CF-SID finetuned checkpoint is forbidden.

This stage does not read, generate, or evaluate `test`.

## Frozen Treatment Input

Treatment input is:

`data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/`

Required files:

- `train.csv`
- `valid.csv`
- `item2sid.json`
- `sid2items.json`
- `valid_sid_set.json`
- `index.json`
- `info.txt`
- `reports/s3_manifest.json`
- `reports/s3_static_quality_report.json`

Accepted S3 properties:

- train rows: `36259`
- valid rows: `4532`
- item count: `3686`
- full SID uniqueness: `3686`
- SID parse rate: `1.0`
- `test_read=false`
- no treatment `test.csv`

## Runner

Use:

```bash
BASE_MODEL=/path/to/shared/base/model \
STAGE=parity \
CONFIG_MODE=smoke \
DRY_RUN=1 \
bash scripts/run_s4_sasrec_sid_valid.sh
```

Stages:

- `env-preflight`: validates Python, `torchrun`, CUDA, and required imports before any `torchrun`.
- `parity`: validates baseline/treatment SID inputs and writes `s4_parity_manifest.json` when `DRY_RUN=0`.
- `train`: prints or runs the isolated SFT command selected by `TRAIN_STREAM=baseline|treatment`.
- `candidates`: prints or runs stream-specific valid generation and exact candidate evaluation selected by `TRAIN_STREAM=baseline|treatment`.
- `summarize`: computes the stream-specific S4 candidate summary, including MRR, from the exact candidate report.

Defaults:

- `split=valid`
- exact full SID candidate expansion only
- resume disabled
- overwrite disabled
- output namespace contains the selected SID version, `seed42`, `smoke/formal`, and `valid`
- candidate artifact scope is explicit:
  - bounded preflight: `preflight_rows<N>/`
  - full valid: `full_valid/`
- smoke uses bounded `SAMPLE=2000`
- formal uses predefined `SAMPLE=30000`
- `MAX_NEW_TOKENS=6`

## Runtime Preflight

Before non-dry-run training, the S4 runner executes:

```bash
python3 scripts/s4_sasrec_sid_valid_pipeline.py --action env-preflight ...
```

The preflight records:

- `sys.executable`
- Python version
- `which torchrun`
- numpy version
- torch version
- transformers version
- accelerate version
- peft version
- `torch.cuda.is_available()`
- GPU count/name when available
- import checks for `AutoModelForCausalLM`, `AutoTokenizer`, and `GenerationMixin`

If required imports fail, S4 stops before `torchrun` and reports that the intended AutoDL training environment must be activated. It never installs or upgrades packages.

## Parameter Matrix

| Parameter | Smoke | Formal |
|---|---:|---:|
| sample | 2000 | 30000 |
| epochs | 1 | 1 |
| learning rate | env/default `2e-5` | env/default `2e-5` |
| per-device train batch | env/default `4` | env/default `4` |
| gradient accumulation | env/default `8` | env/default `8` |
| seed | 42 | 42 |
| candidate mode | exact | exact |
| max new tokens | 6 | 6 |

All baseline/treatment values must match except SID paths and isolated output paths.

## Stop Conditions

Stop before claiming a fair comparison if:

- baseline/treatment do not use the same explicit `BASE_MODEL`;
- runtime preflight fails;
- the baseline `valid.csv` is missing;
- any runner path points to `test`;
- output paths overlap historical CF/Text/Stage7 artifacts;
- tokenizer SID token sets are not representable as complete tokens;
- `<d_i>` tokens are absent or truncated;
- local tests fail;
- formal results are not produced by a real AutoDL run.

## Local Evidence Note

The current WSL repository contains both S4 input streams:

- `data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/valid.csv`
- `data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/valid.csv`

The stale local note that CF `valid.csv` was missing has been superseded by parity evidence. Historical CF formal checkpoint parity still must not be inferred unless the shared `BASE_MODEL` is explicit.

## Output Contracts

Parity manifest:

`results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/<mode>/valid/s4_parity_manifest.json`

Baseline SFT output:

`outputs/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/<mode>/valid/sft/`

Treatment SFT output:

`outputs/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/<mode>/valid/sft/`

Candidate output scopes:

- `CANDIDATE_ROW_LIMIT=8` writes under `preflight_rows8/`.
- `CANDIDATE_ROW_LIMIT=0` writes under `full_valid/`.
- bounded artifacts never share paths with full-valid artifacts.

Baseline valid predictions:

`results/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/<mode>/valid/<scope>/generation/predictions.json`

Treatment valid predictions:

`results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/<mode>/valid/<scope>/generation/predictions.json`

Baseline exact candidates:

`results/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/<mode>/valid/<scope>/candidates/candidates.jsonl`

Treatment exact candidates:

`results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/<mode>/valid/<scope>/candidates/candidates.jsonl`

Baseline candidate report:

`results/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/<mode>/valid/<scope>/candidates/candidate_report.json`

Treatment candidate report:

`results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/<mode>/valid/<scope>/candidates/candidate_report.json`

Each scope also writes:

`results/s4_sasrec_sid_valid/Industrial_and_Scientific/<sid_version>/seed42/<mode>/valid/<scope>/artifact_manifest.json`

The artifact manifest records stream, SID version, config mode, candidate row limit, expected/actual prediction rows, input hashes, checkpoint fingerprint, generation parameters, creation time, and `complete`.

## Candidate Recovery Policy

Use `STAGE=audit-existing TRAIN_STREAM=<baseline|treatment> CANDIDATE_ROW_LIMIT=<N>` before rerunning if artifacts already exist.

Classifications:

- `missing`: safe to run.
- `bounded` / `full_valid`: complete matching artifacts; reuse only with `REUSE_COMPLETE=1`, rerun only with `OVERWRITE=1`.
- `partial`: default stop; same stream/scope rerun may use `OVERWRITE=1`.
- `empty`, `malformed`, `legacy_unknown`, `wrong_stream`, `wrong_sid_version`, `wrong_scope`: default stop and manually quarantine using the printed `mv` command.

`OVERWRITE=1` must not be used to convert bounded artifacts into full-valid artifacts or to cross streams.
