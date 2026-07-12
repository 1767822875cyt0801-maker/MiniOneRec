#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the train-only SASRec embedding builder.

Local default is DRY_RUN=1. Smoke and formal configs are separated by
CONFIG_MODE. This runner never accepts a test CSV.

Environment:
  DRY_RUN       Default: 1
  CATEGORY      Default: Industrial_and_Scientific
  CONFIG_MODE   smoke or formal. Default: smoke
  SEED          Default: 42
  PYTHON        Default: python
  OUTPUT_DIR    Optional override

Examples:
  DRY_RUN=1 CATEGORY=Industrial_and_Scientific CONFIG_MODE=smoke bash scripts/run_sasrec_embedding_builder.sh
  DRY_RUN=0 CATEGORY=Industrial_and_Scientific CONFIG_MODE=smoke bash scripts/run_sasrec_embedding_builder.sh
  DRY_RUN=0 CATEGORY=Industrial_and_Scientific CONFIG_MODE=formal bash scripts/run_sasrec_embedding_builder.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage
  exit 2
fi

CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
CONFIG_MODE="${CONFIG_MODE:-smoke}"
DRY_RUN="${DRY_RUN:-1}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"

case "$CONFIG_MODE" in
  smoke)
    MAX_SEQ_LEN="${MAX_SEQ_LEN:-5}"
    EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
    NUM_LAYERS="${NUM_LAYERS:-1}"
    NUM_HEADS="${NUM_HEADS:-2}"
    DROPOUT="${DROPOUT:-0.1}"
    LEARNING_RATE="${LEARNING_RATE:-0.001}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    EPOCHS="${EPOCHS:-1}"
    MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-256}"
    MAX_VALID_ROWS="${MAX_VALID_ROWS:-128}"
    ;;
  formal)
    MAX_SEQ_LEN="${MAX_SEQ_LEN:-10}"
    EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
    NUM_LAYERS="${NUM_LAYERS:-2}"
    NUM_HEADS="${NUM_HEADS:-2}"
    DROPOUT="${DROPOUT:-0.2}"
    LEARNING_RATE="${LEARNING_RATE:-0.001}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    EPOCHS="${EPOCHS:-20}"
    MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"
    MAX_VALID_ROWS="${MAX_VALID_ROWS:-0}"
    ;;
  *)
    echo "CONFIG_MODE must be smoke or formal, got: $CONFIG_MODE" >&2
    exit 2
    ;;
esac

TRAIN_CSV="${TRAIN_CSV:-data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv}"
VALID_CSV="${VALID_CSV:-data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv}"
ITEM_ORDER="${ITEM_ORDER:-data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.item_order.json}"
ROW_INDEX="${ROW_INDEX:-data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.row_index.json}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  if [[ "$CONFIG_MODE" == "formal" ]]; then
    OUTPUT_DIR="data/Amazon/behavior_embeddings/sasrec/${CATEGORY}/formal_v3_finite_maskfix_seed${SEED}"
  else
    OUTPUT_DIR="data/Amazon/behavior_embeddings/sasrec/${CATEGORY}/${CONFIG_MODE}_train_only_seed${SEED}"
  fi
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

echo "===== SASRec Embedding Builder Environment ====="
echo "category=${CATEGORY}"
echo "config_mode=${CONFIG_MODE}"
echo "dry_run=${DRY_RUN}"
echo "output_dir=${OUTPUT_DIR}"
"$PYTHON" - <<'PY'
import os
import platform
import subprocess
import sys

print("python=" + sys.version.replace("\n", " "))
print("platform=" + platform.platform())
try:
    import torch
    print("torch=" + torch.__version__)
    print("torch_cuda_available=" + str(torch.cuda.is_available()))
    print("torch_cuda_version=" + str(getattr(torch.version, "cuda", None)))
except Exception as exc:
    print("torch=not_available: " + repr(exc))
try:
    result = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
    print("gpu=" + (result.stdout.strip() or result.stderr.strip() or "not_available"))
except Exception as exc:
    print("gpu=not_available: " + repr(exc))
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    print("git_commit=" + commit)
except Exception as exc:
    print("git_commit=not_available: " + repr(exc))
try:
    status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    print("git_status_short=" + (status if status else "clean"))
except Exception as exc:
    print("git_status_short=not_available: " + repr(exc))
PY

cmd=(
  "$PYTHON" build_sasrec_embeddings.py
  --category "$CATEGORY"
  --train-csv "$TRAIN_CSV"
  --valid-csv "$VALID_CSV"
  --item-order "$ITEM_ORDER"
  --row-index "$ROW_INDEX"
  --output-dir "$OUTPUT_DIR"
  --max-seq-len "$MAX_SEQ_LEN"
  --embedding-dim "$EMBEDDING_DIM"
  --num-layers "$NUM_LAYERS"
  --num-heads "$NUM_HEADS"
  --dropout "$DROPOUT"
  --learning-rate "$LEARNING_RATE"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --seed "$SEED"
  --max-train-rows "$MAX_TRAIN_ROWS"
  --max-valid-rows "$MAX_VALID_ROWS"
)

if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
