# S4-1 Smoke Runner Environment and Contract Hardening

## Root Cause

AutoDL S4 parity passed, but baseline smoke failed before model loading because the runner reached `torchrun` without proving that the selected Python environment could import the required training stack.

Observed AutoDL symptoms:

- Python came from `/root/miniconda3/bin/python`;
- Python 3.12 base environment was used;
- numpy dtype repr hit `RecursionError`;
- Transformers `GenerationMixin` import failed;
- `AutoModelForCausalLM` import failed;
- `torchrun` finally reported `ChildFailedError`.

This was an environment preflight gap, not an SFT logic failure.

Two command-contract gaps were also fixed:

- `CONFIG_MODE=smoke` previously produced `--sample -1`, meaning full train data.
- valid generation previously emitted `--max_new_tokens 0`.

## Runner Changes

Changed files:

- `scripts/s4_sasrec_sid_valid_pipeline.py`
- `scripts/run_s4_sasrec_sid_valid.sh`
- `tests/test_s4_sasrec_sid_valid_pipeline.py`
- `docs/s4_sasrec_sid_valid_experiment.md`
- `docs/s4_0_cf_valid_restore_and_parity_closeout.md`

Added report:

`results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/smoke/valid/s4_1_smoke_runner_env_contract_hardening_report.json`

## Runtime Preflight

New stage:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=env-preflight \
DRY_RUN=1 \
CONFIG_MODE=smoke \
bash scripts/run_s4_sasrec_sid_valid.sh
```

The preflight records and verifies:

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
- `AutoModelForCausalLM`
- `AutoTokenizer`
- `GenerationMixin`

When any required import fails, the runner exits before `torchrun` with an explicit message to activate the intended AutoDL training environment. It does not install or upgrade packages.

## Local Preflight Result

Local WSL preflight fails as expected because this local Python environment does not contain the GPU training stack:

- `torchrun` not found on PATH;
- `torch` missing;
- `transformers` missing;
- `accelerate` missing;
- `peft` missing;
- `AutoModelForCausalLM`, `AutoTokenizer`, and `GenerationMixin` imports fail.

This validates the fail-fast path locally. It is not an AutoDL result.

## Smoke/Formal Parameter Contract

| Parameter | Smoke | Formal |
|---|---:|---:|
| sample | 2000 | 30000 |
| epochs | 1 | 1 |
| learning rate | default/override `2e-5` | default/override `2e-5` |
| per-device train batch | default/override `4` | default/override `4` |
| gradient accumulation | default/override `8` | default/override `8` |
| seed | 42 | 42 |
| candidate mode | exact | exact |
| max new tokens | 6 | 6 |

Overrides are allowed through explicit environment variables and are recorded into the parity manifest. Baseline and treatment commands are generated from the same effective values.

## Max New Tokens Basis

`evaluate.py` can infer `max_new_tokens` when the argument is `0`, but S4 command generation now avoids implicit auto mode.

S4 uses:

`MAX_NEW_TOKENS=6`

Reason:

- SIDs have three semantic tokens;
- append-dedup SIDs may have a fourth `<d_i>` token;
- generation needs one token for newline and one token for EOS after the full SID;
- `6` is the explicit 4-level SID + newline + EOS budget used by both baseline and treatment.

## Test Evidence

Local checks:

```bash
PYTHONPYCACHEPREFIX=/tmp/minionerec_pycache python3 -m unittest tests.test_s4_sasrec_sid_valid_pipeline
PYTHONPYCACHEPREFIX=/tmp/minionerec_pycache python3 -m unittest tests.test_s4_sasrec_sid_valid_pipeline tests.test_sasrec_v3_behavior_sid
PYTHONPYCACHEPREFIX=/tmp/minionerec_pycache python3 -m compileall -q scripts tests
find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n
git diff --check
```

S4 dry runs:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
ALLOW_MISSING_BASE_MODEL=1 \
STAGE=parity \
DRY_RUN=0 \
CONFIG_MODE=smoke \
bash scripts/run_s4_sasrec_sid_valid.sh

BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
ALLOW_MISSING_BASE_MODEL=1 \
STAGE=train \
TRAIN_STREAM=baseline \
DRY_RUN=1 \
CONFIG_MODE=smoke \
bash scripts/run_s4_sasrec_sid_valid.sh

BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
ALLOW_MISSING_BASE_MODEL=1 \
STAGE=train \
TRAIN_STREAM=treatment \
DRY_RUN=1 \
CONFIG_MODE=smoke \
bash scripts/run_s4_sasrec_sid_valid.sh
```

The generated smoke train commands now contain:

- `--sample 2000`
- `--max_new_tokens 6` in generation command
- isolated CF/SASRec output directories
- `test_read=false`

## AutoDL Command Order

1. Environment preflight:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=env-preflight \
DRY_RUN=1 \
CONFIG_MODE=smoke \
bash scripts/run_s4_sasrec_sid_valid.sh
```

2. Parity:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=parity \
DRY_RUN=0 \
CONFIG_MODE=smoke \
bash scripts/run_s4_sasrec_sid_valid.sh
```

3. Baseline smoke:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=train \
TRAIN_STREAM=baseline \
DRY_RUN=0 \
CONFIG_MODE=smoke \
bash scripts/run_s4_sasrec_sid_valid.sh
```

4. Treatment smoke:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
STAGE=train \
TRAIN_STREAM=treatment \
DRY_RUN=0 \
CONFIG_MODE=smoke \
bash scripts/run_s4_sasrec_sid_valid.sh
```

## Stop/Go

GO to AutoDL smoke only after `STAGE=env-preflight` passes in the intended training environment.

STOP if preflight reports import failures, missing `torchrun`, unavailable CUDA when GPU training is expected, or a base Python environment is selected.
