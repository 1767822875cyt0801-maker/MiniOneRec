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
  EVAL_SPLIT            valid or test. Default: test.
  PRED_DIR              Default: results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}/${SID_VERSION}/predictions
  CALC_DIR              Default: results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}/${SID_VERSION}/calc
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
EVAL_SPLIT="${EVAL_SPLIT:-test}"
NUM_BEAMS="${NUM_BEAMS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
PRED_DIR="${PRED_DIR:-results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}/${SID_VERSION}/predictions}"
CALC_DIR="${CALC_DIR:-results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}/${SID_VERSION}/calc}"
RUN_CALC_PLUS="${RUN_CALC_PLUS:-1}"
PYTHON="${PYTHON:-python}"

if [[ "$EVAL_SPLIT" != "valid" && "$EVAL_SPLIT" != "test" ]]; then
  echo "EVAL_SPLIT must be valid or test, got: $EVAL_SPLIT" >&2
  exit 2
fi

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "Checkpoint directory not found: $CHECKPOINT" >&2
  exit 1
fi

eval "$("$PYTHON" - "$MANIFEST" "$SID_VERSION" "$CATEGORY" "$EVAL_SPLIT" <<'PY'
import json
import shlex
import sys
from pathlib import Path

manifest_path, sid_version, category, eval_split = sys.argv[1:5]
manifest = {}
if Path(manifest_path).exists():
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
entry = manifest.get("sid_versions", {}).get(sid_version, {}).get(category, {})
version_dir = Path("data/Amazon/sid_versions") / sid_version / category
fallbacks = {
    "train_csv": version_dir / "train.csv",
    "valid_csv": version_dir / "valid.csv",
    "test_csv": version_dir / "test.csv",
    "info": version_dir / "info.txt",
    "item2sid": version_dir / "item2sid.json",
    "sid2items": version_dir / "sid2items.json",
    "valid_sid_set": version_dir / "valid_sid_set.json",
}
required = ["train_csv", f"{eval_split}_csv", "info", "item2sid", "sid2items", "valid_sid_set"]
resolved = {}
for key in required:
    value = entry.get(key)
    if not value:
        value = fallbacks[key].as_posix()
    resolved[key] = value
for key, value in resolved.items():
    env_key = "EVAL_CSV" if key == f"{eval_split}_csv" else key.upper()
    print(f"{env_key}={shlex.quote(value)}")
PY
)"

for required_path in "$TRAIN_CSV" "$EVAL_CSV" "$INFO" "$ITEM2SID" "$SID2ITEMS" "$VALID_SID_SET"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Required file not found: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$PRED_DIR"

echo "Evaluate smoke"
echo "  CATEGORY=$CATEGORY"
echo "  SID_VERSION=$SID_VERSION"
echo "  EVAL_SPLIT=$EVAL_SPLIT"
echo "  CHECKPOINT=$CHECKPOINT"
echo "  INFO=$INFO"
echo "  EVAL_CSV=$EVAL_CSV"
echo "  NUM_BEAMS=$NUM_BEAMS"
echo "  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" evaluate.py \
  --base_model "$CHECKPOINT" \
  --train_file "$TRAIN_CSV" \
  --info_file "$INFO" \
  --category "$CATEGORY" \
  --test_data_path "$EVAL_CSV" \
  --result_json_data "$PRED_DIR/predictions.json" \
  --batch_size "$BATCH_SIZE" \
  --num_beams "$NUM_BEAMS" \
  --max_new_tokens "$MAX_NEW_TOKENS"

if [[ "$RUN_CALC_PLUS" == "1" ]]; then
  "$PYTHON" calc_plus.py \
    --prediction-file "$PRED_DIR/predictions.json" \
    --eval-csv "$EVAL_CSV" \
    --eval-split "$EVAL_SPLIT" \
    --train-csv "$TRAIN_CSV" \
    --item2sid "$ITEM2SID" \
    --sid2items "$SID2ITEMS" \
    --valid-sid-set "$VALID_SID_SET" \
    --output-dir "$CALC_DIR" \
    --sid-version "$SID_VERSION"
fi
