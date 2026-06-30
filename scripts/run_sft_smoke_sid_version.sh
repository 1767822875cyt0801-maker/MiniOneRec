#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  CATEGORY=Industrial_and_Scientific \
  SID_VERSION=cs_alpha0.7_k512_dedup \
  BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B \
  bash scripts/run_sft_smoke_sid_version.sh

Environment:
  CATEGORY              Required category name.
  SID_VERSION           Required SID version in experiment_manifest.json.
  BASE_MODEL            Required base model or checkpoint path.
  MANIFEST              Default: data/Amazon/sid_maps/experiment_manifest.json
  OUTPUT_DIR            Default: outputs/sft_smoke_${CATEGORY}_${SID_VERSION}
  SAMPLE_SIZE           Default: 2000
  NUM_EPOCHS            Default: 1
  CUTOFF_LEN            Default: 512
  BATCH_SIZE            Default: 64
  MICRO_BATCH_SIZE      Default: 4
  LEARNING_RATE         Default: 3e-4
  SFT_TASK_MODE         Default: sid_only. Use all/mixed to include auxiliary tasks.
  BF16                  Default: True
  SAVE_DURING_TRAINING  Default: True. Set False for disk-light smoke runs.
  EARLY_STOPPING_PATIENCE Default: 3. Set 0 to disable early stopping callback.
  LOAD_BEST_MODEL_AT_END  Default: True. Set False for fixed-step smoke comparisons.
  ITEM_META_PATH        Default: data/Amazon/index/${CATEGORY}.item.json
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

CATEGORY="${CATEGORY:?CATEGORY is required}"
SID_VERSION="${SID_VERSION:?SID_VERSION is required}"
BASE_MODEL="${BASE_MODEL:?BASE_MODEL is required}"
MANIFEST="${MANIFEST:-data/Amazon/sid_maps/experiment_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_smoke_${CATEGORY}_${SID_VERSION}}"
SAMPLE_SIZE="${SAMPLE_SIZE:-2000}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
CUTOFF_LEN="${CUTOFF_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
SFT_TASK_MODE="${SFT_TASK_MODE:-sid_only}"
BF16="${BF16:-True}"
SAVE_DURING_TRAINING="${SAVE_DURING_TRAINING:-True}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-True}"
ITEM_META_PATH="${ITEM_META_PATH:-data/Amazon/index/${CATEGORY}.item.json}"
PYTHON="${PYTHON:-python}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR already exists, refusing to overwrite: $OUTPUT_DIR" >&2
  exit 1
fi

eval "$("$PYTHON" - "$MANIFEST" "$SID_VERSION" "$CATEGORY" <<'PY'
import json
import shlex
import sys

manifest_path, sid_version, category = sys.argv[1:4]
manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
entry = manifest.get("sid_versions", {}).get(sid_version, {}).get(category)
if not entry:
    raise SystemExit(f"Manifest entry not found: sid_version={sid_version}, category={category}")
required = ["train_csv", "valid_csv", "index"]
missing = [key for key in required if key not in entry]
if missing:
    raise SystemExit(f"Manifest entry missing required keys: {missing}")
print(f"TRAIN_CSV={shlex.quote(entry['train_csv'])}")
print(f"VALID_CSV={shlex.quote(entry['valid_csv'])}")
print(f"SID_INDEX_PATH={shlex.quote(entry['index'])}")
PY
)"

for required_path in "$TRAIN_CSV" "$VALID_CSV" "$SID_INDEX_PATH" "$ITEM_META_PATH"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Required file not found: $required_path" >&2
    exit 1
  fi
done

echo "SFT smoke"
echo "  CATEGORY=$CATEGORY"
echo "  SID_VERSION=$SID_VERSION"
echo "  TRAIN_CSV=$TRAIN_CSV"
echo "  VALID_CSV=$VALID_CSV"
echo "  SID_INDEX_PATH=$SID_INDEX_PATH"
echo "  ITEM_META_PATH=$ITEM_META_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  SAMPLE_SIZE=$SAMPLE_SIZE"
echo "  SFT_TASK_MODE=$SFT_TASK_MODE"
echo "  SAVE_DURING_TRAINING=$SAVE_DURING_TRAINING"
echo "  EARLY_STOPPING_PATIENCE=$EARLY_STOPPING_PATIENCE"
echo "  LOAD_BEST_MODEL_AT_END=$LOAD_BEST_MODEL_AT_END"

"$PYTHON" sft.py \
  --base_model "$BASE_MODEL" \
  --train_file "$TRAIN_CSV" \
  --eval_file "$VALID_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --sample "$SAMPLE_SIZE" \
  --seed 42 \
  --batch_size "$BATCH_SIZE" \
  --micro_batch_size "$MICRO_BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --cutoff_len "$CUTOFF_LEN" \
  --category "$CATEGORY" \
  --sid_index_path "$SID_INDEX_PATH" \
  --item_meta_path "$ITEM_META_PATH" \
  --bf16 "$BF16" \
  --save_during_training "$SAVE_DURING_TRAINING" \
  --sft_task_mode "$SFT_TASK_MODE" \
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
  --load_best_model_at_end "$LOAD_BEST_MODEL_AT_END"
