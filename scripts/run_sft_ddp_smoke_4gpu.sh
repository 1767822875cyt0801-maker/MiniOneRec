#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
NUM_GPUS="${NUM_GPUS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_ddp_smoke_${CATEGORY}_Qwen25_15B_Instruct_4gpu}"

TRAIN_CSV="${TRAIN_CSV:-$(ls ./data/Amazon/train/${CATEGORY}*.csv | head -1)}"
VALID_CSV="${VALID_CSV:-$(ls ./data/Amazon/valid/${CATEGORY}*.csv | head -1)}"
SID_INDEX_PATH="${SID_INDEX_PATH:-./data/Amazon/index/${CATEGORY}.index.json}"
ITEM_META_PATH="${ITEM_META_PATH:-./data/Amazon/index/${CATEGORY}.item.json}"

bash scripts/run_sft_ddp.sh \
  --category "$CATEGORY" \
  --base-model "$BASE_MODEL" \
  --train-csv "$TRAIN_CSV" \
  --valid-csv "$VALID_CSV" \
  --sid-index-path "$SID_INDEX_PATH" \
  --item-meta-path "$ITEM_META_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --num-gpus "$NUM_GPUS" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-4}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
  --learning-rate "${LEARNING_RATE:-2e-5}" \
  --cutoff-len "${CUTOFF_LEN:-512}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --bf16 "${BF16:-True}" \
  --sample "${SAMPLE:-1024}" \
  --gradient-checkpointing "${GRADIENT_CHECKPOINTING:-False}"
