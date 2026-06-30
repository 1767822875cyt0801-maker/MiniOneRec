#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  CATEGORY=Industrial_and_Scientific \
  SID_VERSION=cs_alpha0.7_k512_dedup \
  OUTPUT_DIR=outputs/sft_smoke_Industrial_and_Scientific_cs_alpha0.7_k512_dedup \
  bash scripts/eval_smoke_sid_version.sh

Environment:
  CATEGORY              Required category name.
  SID_VERSION           Required SID version in experiment_manifest.json.
  MANIFEST              Default: data/Amazon/sid_maps/experiment_manifest.json
  OUTPUT_DIR            SFT output dir. CHECKPOINT defaults to ${OUTPUT_DIR}/final_checkpoint.
  CHECKPOINT            Optional explicit model checkpoint.
  PRED_DIR              Default: results/eval_smoke_${CATEGORY}_${SID_VERSION}_beam${NUM_BEAMS}
  CALC_DIR              Default: results/calc_plus_smoke_${CATEGORY}_${SID_VERSION}_beam${NUM_BEAMS}
  NUM_BEAMS             Default: 20
  BATCH_SIZE            Default: 4
  MAX_NEW_TOKENS        Default: 0, which lets evaluate.py infer automatically.
  RUN_CALC_PLUS         Default: 1
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
MANIFEST="${MANIFEST:-data/Amazon/sid_maps/experiment_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_smoke_${CATEGORY}_${SID_VERSION}}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/final_checkpoint}"
NUM_BEAMS="${NUM_BEAMS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
PRED_DIR="${PRED_DIR:-results/eval_smoke_${CATEGORY}_${SID_VERSION}_beam${NUM_BEAMS}}"
CALC_DIR="${CALC_DIR:-results/calc_plus_smoke_${CATEGORY}_${SID_VERSION}_beam${NUM_BEAMS}}"
RUN_CALC_PLUS="${RUN_CALC_PLUS:-1}"
PYTHON="${PYTHON:-python}"

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "Checkpoint directory not found: $CHECKPOINT" >&2
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
required = ["train_csv", "test_csv", "info", "item2sid", "sid2items", "valid_sid_set"]
missing = [key for key in required if key not in entry]
if missing:
    raise SystemExit(f"Manifest entry missing required keys: {missing}")
for key in required:
    print(f"{key.upper()}={shlex.quote(entry[key])}")
PY
)"

for required_path in "$TRAIN_CSV" "$TEST_CSV" "$INFO" "$ITEM2SID" "$SID2ITEMS" "$VALID_SID_SET"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Required file not found: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$PRED_DIR"

echo "Evaluate smoke"
echo "  CATEGORY=$CATEGORY"
echo "  SID_VERSION=$SID_VERSION"
echo "  CHECKPOINT=$CHECKPOINT"
echo "  INFO=$INFO"
echo "  TEST_CSV=$TEST_CSV"
echo "  NUM_BEAMS=$NUM_BEAMS"
echo "  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" evaluate.py \
  --base_model "$CHECKPOINT" \
  --train_file "$TRAIN_CSV" \
  --info_file "$INFO" \
  --category "$CATEGORY" \
  --test_data_path "$TEST_CSV" \
  --result_json_data "$PRED_DIR/predictions.json" \
  --batch_size "$BATCH_SIZE" \
  --num_beams "$NUM_BEAMS" \
  --max_new_tokens "$MAX_NEW_TOKENS"

if [[ "$RUN_CALC_PLUS" == "1" ]]; then
  "$PYTHON" calc_plus.py \
    --prediction-file "$PRED_DIR/predictions.json" \
    --test-csv "$TEST_CSV" \
    --train-csv "$TRAIN_CSV" \
    --item2sid "$ITEM2SID" \
    --sid2items "$SID2ITEMS" \
    --valid-sid-set "$VALID_SID_SET" \
    --output-dir "$CALC_DIR" \
    --sid-version "$SID_VERSION"
fi
