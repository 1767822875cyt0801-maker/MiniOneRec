#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  CATEGORY=Industrial_and_Scientific \
  SID_VERSION_LIST="cf cs_alpha0.7 cs_alpha0.5 cs_alpha0.2" \
  OUTPUT_ROOT=data/Amazon/sid_versions \
  CS_EMB_ROOT=data/Amazon/cs_embeddings \
  MANIFEST=data/Amazon/sid_maps/experiment_manifest.json \
  bash scripts/generate_sid_versions_and_rewrite_csv.sh [--overwrite]

Environment:
  CATEGORY              Required category name.
  SID_VERSION_LIST      Space-separated versions. Default: cf cs_alpha0.7 cs_alpha0.5 cs_alpha0.2
  OUTPUT_ROOT           Default: data/Amazon/sid_versions
  CS_EMB_ROOT           Default: data/Amazon/cs_embeddings
  TEXT_EMB_ROOT         Default: data/Amazon/index
  MANIFEST              Default: data/Amazon/sid_maps/experiment_manifest.json
  CODEBOOK_SIZE         Default: 256
  NUM_LEVELS            Default: 3
  SEED                  Default: 42
  DEDUP_MODE            Default: none. Version names ending in _dedup force append.
  OVERWRITE             Default: 0. Set 1 or pass --overwrite to replace existing version outputs.
  SKIP_MISSING          Default: 0. Set 1 to skip SID versions whose embedding file is missing.
EOF
}

OVERWRITE="${OVERWRITE:-0}"
if [[ "${1:-}" == "--overwrite" ]]; then
  OVERWRITE=1
  shift
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage
  exit 2
fi

CATEGORY="${CATEGORY:?CATEGORY is required, e.g. Industrial_and_Scientific}"
SID_VERSION_LIST="${SID_VERSION_LIST:-cf cs_alpha0.7 cs_alpha0.5 cs_alpha0.2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/Amazon/sid_versions}"
CS_EMB_ROOT="${CS_EMB_ROOT:-data/Amazon/cs_embeddings}"
TEXT_EMB_ROOT="${TEXT_EMB_ROOT:-data/Amazon/index}"
MANIFEST="${MANIFEST:-data/Amazon/sid_maps/experiment_manifest.json}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
NUM_LEVELS="${NUM_LEVELS:-3}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"
SKIP_MISSING="${SKIP_MISSING:-0}"
DEDUP_MODE="${DEDUP_MODE:-none}"

TRAIN_CSV="${TRAIN_CSV:-$(ls "data/Amazon/train/${CATEGORY}"*.csv | head -1)}"
VALID_CSV="${VALID_CSV:-$(ls "data/Amazon/valid/${CATEGORY}"*.csv | head -1)}"
TEST_CSV="${TEST_CSV:-$(ls "data/Amazon/test/${CATEGORY}"*.csv | head -1)}"
ITEM_JSON="${ITEM_JSON:-data/Amazon/index/${CATEGORY}.item.json}"
ITEM_ORDER="${ITEM_ORDER:-${CS_EMB_ROOT}/${CATEGORY}/${CATEGORY}.item_order.json}"
ROW_INDEX="${ROW_INDEX:-${CS_EMB_ROOT}/${CATEGORY}/${CATEGORY}.row_index.json}"

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args=(--overwrite)
fi

embedding_path_for_version() {
  local version="$1"
  case "$version" in
    text_mbk*)
      echo "${TEXT_EMB_ROOT}/${CATEGORY}.emb-qwen-td.npy"
      ;;
    cf*)
      echo "${CS_EMB_ROOT}/${CATEGORY}/${CATEGORY}.cf_emb.npy"
      ;;
    cs_alpha*)
      local rest="${version#cs_alpha}"
      local alpha="${rest%%_*}"
      echo "${CS_EMB_ROOT}/${CATEGORY}/${CATEGORY}.cs_emb_alpha${alpha}.npy"
      ;;
    *)
      echo "Unsupported SID version: ${version}" >&2
      return 1
      ;;
  esac
}

codebook_size_for_version() {
  local version="$1"
  if [[ "$version" == *"_k512"* ]]; then
    echo "512"
  elif [[ "$version" == *"_k256"* ]]; then
    echo "256"
  else
    echo "$CODEBOOK_SIZE"
  fi
}

dedup_mode_for_version() {
  local version="$1"
  if [[ "$version" == *"_dedup" ]]; then
    echo "append"
  else
    echo "$DEDUP_MODE"
  fi
}

echo "Batch SID generation and CSV rewrite"
echo "  CATEGORY=${CATEGORY}"
echo "  SID_VERSION_LIST=${SID_VERSION_LIST}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  CS_EMB_ROOT=${CS_EMB_ROOT}"
echo "  TEXT_EMB_ROOT=${TEXT_EMB_ROOT}"
echo "  MANIFEST=${MANIFEST}"
echo "  CODEBOOK_SIZE=${CODEBOOK_SIZE}"
echo "  DEDUP_MODE=${DEDUP_MODE}"
echo "  OVERWRITE=${OVERWRITE}"
echo "  SKIP_MISSING=${SKIP_MISSING}"

required_paths=(
  "$TRAIN_CSV"
  "$VALID_CSV"
  "$TEST_CSV"
  "$ITEM_JSON"
  "$ITEM_ORDER"
  "$ROW_INDEX"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Required input file not found: $required_path" >&2
    exit 1
  fi
done

VALID_SID_VERSION_LIST=()
for SID_VERSION in $SID_VERSION_LIST; do
  EMB_PATH="$(embedding_path_for_version "$SID_VERSION")"
  if [[ ! -f "$EMB_PATH" ]]; then
    if [[ "$SKIP_MISSING" == "1" ]]; then
      echo "Skipping ${SID_VERSION}: embedding file not found: ${EMB_PATH}" >&2
      continue
    fi
    echo "Embedding file not found for SID_VERSION=${SID_VERSION}: ${EMB_PATH}" >&2
    echo "Either build this embedding first, remove ${SID_VERSION} from SID_VERSION_LIST, or set SKIP_MISSING=1." >&2
    exit 1
  fi
  VALID_SID_VERSION_LIST+=("$SID_VERSION")
done

if [[ "${#VALID_SID_VERSION_LIST[@]}" -eq 0 ]]; then
  echo "No SID versions to process after input validation." >&2
  exit 1
fi

for SID_VERSION in "${VALID_SID_VERSION_LIST[@]}"; do
  echo
  echo "===== ${CATEGORY} / ${SID_VERSION} ====="
  EMB_PATH="$(embedding_path_for_version "$SID_VERSION")"
  VERSION_CODEBOOK_SIZE="$(codebook_size_for_version "$SID_VERSION")"
  VERSION_DEDUP_MODE="$(dedup_mode_for_version "$SID_VERSION")"
  VERSION_DIR="${OUTPUT_ROOT}/${SID_VERSION}/${CATEGORY}"
  REPORT_DIR="${VERSION_DIR}/reports"

  "$PYTHON" run_rqkmeans_with_emb.py \
    --category "$CATEGORY" \
    --emb-path "$EMB_PATH" \
    --item-order "$ITEM_ORDER" \
    --row-index "$ROW_INDEX" \
    --sid-version "$SID_VERSION" \
    --output-root "$OUTPUT_ROOT" \
    --item-json "$ITEM_JSON" \
    --num-levels "$NUM_LEVELS" \
    --codebook-size "$VERSION_CODEBOOK_SIZE" \
    --seed "$SEED" \
    --dedup-mode "$VERSION_DEDUP_MODE" \
    "${overwrite_args[@]}"

  "$PYTHON" rewrite_sid_csv.py \
    --input-csv "$TRAIN_CSV" \
    --output-csv "${VERSION_DIR}/train.csv" \
    --item2sid "${VERSION_DIR}/item2sid.json" \
    --report-path "${REPORT_DIR}/rewrite_train_report.json" \
    "${overwrite_args[@]}"

  "$PYTHON" rewrite_sid_csv.py \
    --input-csv "$VALID_CSV" \
    --output-csv "${VERSION_DIR}/valid.csv" \
    --item2sid "${VERSION_DIR}/item2sid.json" \
    --report-path "${REPORT_DIR}/rewrite_valid_report.json" \
    "${overwrite_args[@]}"

  "$PYTHON" rewrite_sid_csv.py \
    --input-csv "$TEST_CSV" \
    --output-csv "${VERSION_DIR}/test.csv" \
    --item2sid "${VERSION_DIR}/item2sid.json" \
    --report-path "${REPORT_DIR}/rewrite_test_report.json" \
    "${overwrite_args[@]}"

  "$PYTHON" update_sid_manifest.py \
    --manifest "$MANIFEST" \
    --category "$CATEGORY" \
    --sid-version "$SID_VERSION" \
    --index "${VERSION_DIR}/index.json" \
    --info "${VERSION_DIR}/info.txt" \
    --item2sid "${VERSION_DIR}/item2sid.json" \
    --sid2items "${VERSION_DIR}/sid2items.json" \
    --valid-sid-set "${VERSION_DIR}/valid_sid_set.json" \
    --item-mapping "${VERSION_DIR}/item_mapping.json" \
    --train-csv "${VERSION_DIR}/train.csv" \
    --valid-csv "${VERSION_DIR}/valid.csv" \
    --test-csv "${VERSION_DIR}/test.csv"

  "$PYTHON" analyze_sid.py \
    --manifest "$MANIFEST" \
    --sid-version "$SID_VERSION" \
    --categories "$CATEGORY" \
    --output-dir "${REPORT_DIR}/analysis"

  "$PYTHON" check_sid_stage0.py \
    --manifest "$MANIFEST" \
    --category "$CATEGORY" \
    --sid-version "$SID_VERSION" \
    --report-path "${REPORT_DIR}/stage0_check_report.json"
done

echo
echo "All requested SID versions completed for ${CATEGORY}."
