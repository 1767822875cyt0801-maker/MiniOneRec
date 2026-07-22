#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build SASRec v3 Strong Behavior-SID artifacts.

Defaults are split-safe: train/valid only, no test CSV, no overwrite.

Environment:
  CATEGORY      Default: Industrial_and_Scientific
  CONFIG_MODE   smoke or formal. Default: formal
  DRY_RUN       Default: 1
  SID_VERSION   Default: sasrec_v3_k512_dedup
  SASREC_DIR    Default: data/Amazon/behavior_embeddings/sasrec/$CATEGORY/formal_v3_finite_maskfix_seed42
  OUTPUT_ROOT   Default: data/Amazon/sid_versions
  TRAIN_CSV     Optional override
  VALID_CSV     Optional override
  ITEM_JSON     Optional override
  EXPECTED_NUM_ITEMS Default: 3686
  EXPECTED_DIM       Default: 128
  PYTHON        Default: python3
  OVERWRITE     Default: 0

Examples:
  DRY_RUN=1 CONFIG_MODE=formal bash scripts/run_sasrec_v3_behavior_sid.sh
  DRY_RUN=0 CONFIG_MODE=smoke SID_VERSION=sasrec_v3_smoke_k32_dedup bash scripts/run_sasrec_v3_behavior_sid.sh
  DRY_RUN=0 CONFIG_MODE=formal bash scripts/run_sasrec_v3_behavior_sid.sh
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
CONFIG_MODE="${CONFIG_MODE:-formal}"
DRY_RUN="${DRY_RUN:-1}"
SID_VERSION="${SID_VERSION:-sasrec_v3_k512_dedup}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/Amazon/sid_versions}"
SASREC_DIR="${SASREC_DIR:-data/Amazon/behavior_embeddings/sasrec/${CATEGORY}/formal_v3_finite_maskfix_seed42}"
PYTHON="${PYTHON:-python3}"
OVERWRITE="${OVERWRITE:-0}"
SEED="${SEED:-42}"
EXPECTED_NUM_ITEMS="${EXPECTED_NUM_ITEMS:-3686}"
EXPECTED_DIM="${EXPECTED_DIM:-128}"

case "$CONFIG_MODE" in
  smoke)
    CODEBOOK_SIZE="${CODEBOOK_SIZE:-32}"
    MAX_ITER="${MAX_ITER:-5}"
    ;;
  formal)
    CODEBOOK_SIZE="${CODEBOOK_SIZE:-512}"
    MAX_ITER="${MAX_ITER:-100}"
    ;;
  *)
    echo "CONFIG_MODE must be smoke or formal, got: $CONFIG_MODE" >&2
    exit 2
    ;;
esac

cmd=(
  "$PYTHON" scripts/build_sasrec_v3_behavior_sid.py
  --category "$CATEGORY"
  --config-mode "$CONFIG_MODE"
  --sasrec-dir "$SASREC_DIR"
  --sid-version "$SID_VERSION"
  --output-root "$OUTPUT_ROOT"
  --codebook-size "$CODEBOOK_SIZE"
  --max-iter "$MAX_ITER"
  --seed "$SEED"
  --expected-num-items "$EXPECTED_NUM_ITEMS"
  --expected-dim "$EXPECTED_DIM"
)

if [[ -n "${TRAIN_CSV:-}" ]]; then
  cmd+=(--train-csv "$TRAIN_CSV")
fi
if [[ -n "${VALID_CSV:-}" ]]; then
  cmd+=(--valid-csv "$VALID_CSV")
fi
if [[ -n "${ITEM_JSON:-}" ]]; then
  cmd+=(--item-json "$ITEM_JSON")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "===== SASRec v3 Behavior-SID Runner ====="
echo "category=${CATEGORY}"
echo "config_mode=${CONFIG_MODE}"
echo "dry_run=${DRY_RUN}"
echo "sid_version=${SID_VERSION}"
echo "sasrec_dir=${SASREC_DIR}"
echo "output_root=${OUTPUT_ROOT}"
echo "codebook_size=${CODEBOOK_SIZE}"
echo "max_iter=${MAX_ITER}"
echo "overwrite=${OVERWRITE}"
echo "expected_num_items=${EXPECTED_NUM_ITEMS}"
echo "expected_dim=${EXPECTED_DIM}"

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
