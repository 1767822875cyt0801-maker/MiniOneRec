#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_sft_ddp.sh \
    --category Industrial_and_Scientific \
    --base-model /path/to/Qwen2.5-1.5B-Instruct \
    --train-csv ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --valid-csv ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --sid-index-path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item-meta-path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --output-dir outputs/sft_ddp_Industrial_and_Scientific_Qwen25_15B_Instruct_4a100 \
    --num-gpus 4 \
    --per-device-train-batch-size 8 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-5 \
    --cutoff-len 512 \
    --num-train-epochs 10 \
    --bf16 True \
    --sample -1 \
    --gradient-checkpointing False

Set ALLOW_EXISTING_OUTPUT_DIR=1 only when intentionally resuming into an existing output dir.
EOF
}

CATEGORY=""
BASE_MODEL=""
TRAIN_CSV=""
VALID_CSV=""
SID_INDEX_PATH=""
ITEM_META_PATH=""
OUTPUT_DIR=""
NUM_GPUS=4
PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=""
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=2e-5
CUTOFF_LEN=512
NUM_TRAIN_EPOCHS=10
BF16=True
SAMPLE=-1
GRADIENT_CHECKPOINTING=False
RESUME_FROM_CHECKPOINT=""
SAVE_TOTAL_LIMIT=1
EARLY_STOPPING_PATIENCE=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --category) CATEGORY="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --train-csv) TRAIN_CSV="$2"; shift 2 ;;
    --valid-csv) VALID_CSV="$2"; shift 2 ;;
    --sid-index-path) SID_INDEX_PATH="$2"; shift 2 ;;
    --item-meta-path) ITEM_META_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --per-device-train-batch-size) PER_DEVICE_TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --per-device-eval-batch-size) PER_DEVICE_EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --gradient-accumulation-steps) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --cutoff-len) CUTOFF_LEN="$2"; shift 2 ;;
    --num-train-epochs) NUM_TRAIN_EPOCHS="$2"; shift 2 ;;
    --bf16) BF16="$2"; shift 2 ;;
    --sample) SAMPLE="$2"; shift 2 ;;
    --gradient-checkpointing) GRADIENT_CHECKPOINTING="$2"; shift 2 ;;
    --resume-from-checkpoint) RESUME_FROM_CHECKPOINT="$2"; shift 2 ;;
    --save-total-limit) SAVE_TOTAL_LIMIT="$2"; shift 2 ;;
    --early-stopping-patience) EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$CATEGORY" || -z "$BASE_MODEL" || -z "$TRAIN_CSV" || -z "$VALID_CSV" || -z "$SID_INDEX_PATH" || -z "$ITEM_META_PATH" || -z "$OUTPUT_DIR" ]]; then
  echo "Missing required argument." >&2
  usage
  exit 2
fi

if [[ -e "$OUTPUT_DIR" && "${ALLOW_EXISTING_OUTPUT_DIR:-0}" != "1" ]]; then
  echo "Output dir already exists: $OUTPUT_DIR" >&2
  echo "Refusing to overwrite. Set ALLOW_EXISTING_OUTPUT_DIR=1 only for an intentional resume." >&2
  exit 1
fi

if [[ -z "$PER_DEVICE_EVAL_BATCH_SIZE" ]]; then
  PER_DEVICE_EVAL_BATCH_SIZE="$PER_DEVICE_TRAIN_BATCH_SIZE"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))

echo "Starting DDP SFT"
echo "  category=$CATEGORY"
echo "  base_model=$BASE_MODEL"
echo "  output_dir=$OUTPUT_DIR"
echo "  num_gpus=$NUM_GPUS"
echo "  per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
echo "  effective_batch_size=$EFFECTIVE_BATCH_SIZE"
echo "  learning_rate=$LEARNING_RATE"
echo "  cutoff_len=$CUTOFF_LEN"
echo "  num_train_epochs=$NUM_TRAIN_EPOCHS"
echo "  sample=$SAMPLE"
echo "  gradient_checkpointing=$GRADIENT_CHECKPOINTING"

ARGS=(
  sft.py
  --base_model "$BASE_MODEL"
  --train_file "$TRAIN_CSV"
  --eval_file "$VALID_CSV"
  --output_dir "$OUTPUT_DIR"
  --category "$CATEGORY"
  --train_from_scratch False
  --seed 42
  --sid_index_path "$SID_INDEX_PATH"
  --item_meta_path "$ITEM_META_PATH"
  --freeze_LLM False
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --cutoff_len "$CUTOFF_LEN"
  --num_epochs "$NUM_TRAIN_EPOCHS"
  --bf16 "$BF16"
  --sample "$SAMPLE"
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
  --ddp_find_unused_parameters False
  --save_on_each_node False
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --load_best_model_at_end True
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE"
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

torchrun --standalone --nproc_per_node="$NUM_GPUS" "${ARGS[@]}"
