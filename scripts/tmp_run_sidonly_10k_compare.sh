#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Temporary one-click runner for Industrial SID-only 10k comparison.

Default workflow:
  1. Train text_mbk_k256_dedup, sample=10000, 1 epoch, sid_only.
  2. Evaluate beam20 and run calc_plus.
  3. Train cs_alpha0.7_k512_dedup, sample=10000, 1 epoch, sid_only.
  4. Evaluate beam20 and run calc_plus.
  5. Print a compact metric summary.

Environment overrides:
  CATEGORY        Default: Industrial_and_Scientific
  BASE_MODEL      Default: /root/autodl-tmp/models/Qwen2.5-0.5B
  SAMPLE_SIZE     Default: 10000
  NUM_EPOCHS      Default: 1
  NUM_BEAMS       Default: 20
  MAX_NEW_TOKENS  Default: 0
  SID_VERSIONS    Default: "text_mbk_k256_dedup cs_alpha0.7_k512_dedup"
  RUN_LABEL       Default: noearly. Appended to output/result directories.
  EARLY_STOPPING_PATIENCE Default: 0.
  LOAD_BEST_MODEL_AT_END  Default: False.
  SAVE_DURING_TRAINING    Default: False.
  FORCE_EVAL      Default: 0. Set 1 to rerun eval/calc even if report exists.

Run:
  bash scripts/tmp_run_sidonly_10k_compare.sh
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
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-0.5B}"
SAMPLE_SIZE="${SAMPLE_SIZE:-10000}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_BEAMS="${NUM_BEAMS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
SID_VERSIONS="${SID_VERSIONS:-text_mbk_k256_dedup cs_alpha0.7_k512_dedup}"
RUN_LABEL="${RUN_LABEL:-noearly}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-False}"
SAVE_DURING_TRAINING="${SAVE_DURING_TRAINING:-False}"
FORCE_EVAL="${FORCE_EVAL:-0}"
PYTHON="${PYTHON:-python}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CATEGORY SAMPLE_SIZE NUM_EPOCHS NUM_BEAMS RUN_LABEL

output_dir_for_version() {
  local version="$1"
  local suffix=""
  if [[ -n "$RUN_LABEL" ]]; then
    suffix="_${RUN_LABEL}"
  fi
  case "${CATEGORY}:${version}" in
    Industrial_and_Scientific:text_mbk_k256_dedup)
      echo "outputs/sft_sidonly_Industrial_text_mbk_k256_dedup_sample${SAMPLE_SIZE}_ep${NUM_EPOCHS}${suffix}"
      ;;
    Industrial_and_Scientific:cs_alpha0.7_k512_dedup)
      echo "outputs/sft_sidonly_Industrial_cs_alpha0.7_k512_dedup_sample${SAMPLE_SIZE}_ep${NUM_EPOCHS}${suffix}"
      ;;
    *)
      echo "outputs/sft_sidonly_${CATEGORY}_${version}_sample${SAMPLE_SIZE}_ep${NUM_EPOCHS}${suffix}"
      ;;
  esac
}

pred_dir_for_version() {
  local version="$1"
  local suffix=""
  if [[ -n "$RUN_LABEL" ]]; then
    suffix="_${RUN_LABEL}"
  fi
  case "${CATEGORY}:${version}" in
    Industrial_and_Scientific:text_mbk_k256_dedup)
      echo "results/eval_sidonly_Industrial_text_mbk_k256_dedup_sample${SAMPLE_SIZE}_ep${NUM_EPOCHS}${suffix}_beam${NUM_BEAMS}"
      ;;
    Industrial_and_Scientific:cs_alpha0.7_k512_dedup)
      echo "results/eval_sidonly_Industrial_cs_alpha0.7_k512_dedup_sample${SAMPLE_SIZE}_ep${NUM_EPOCHS}${suffix}_beam${NUM_BEAMS}"
      ;;
    *)
      echo "results/eval_sidonly_${CATEGORY}_${version}_sample${SAMPLE_SIZE}_ep${NUM_EPOCHS}${suffix}_beam${NUM_BEAMS}"
      ;;
  esac
}

calc_dir_for_version() {
  local version="$1"
  local suffix=""
  if [[ -n "$RUN_LABEL" ]]; then
    suffix="_${RUN_LABEL}"
  fi
  case "${CATEGORY}:${version}" in
    Industrial_and_Scientific:text_mbk_k256_dedup)
      echo "results/calc_plus_sidonly_Industrial_text_mbk_k256_dedup_sample${SAMPLE_SIZE}_ep${NUM_EPOCHS}${suffix}_beam${NUM_BEAMS}"
      ;;
    Industrial_and_Scientific:cs_alpha0.7_k512_dedup)
      echo "results/calc_plus_sidonly_Industrial_cs_alpha0.7_k512_dedup_sample${SAMPLE_SIZE}_ep${NUM_EPOCHS}${suffix}_beam${NUM_BEAMS}"
      ;;
    *)
      echo "results/calc_plus_sidonly_${CATEGORY}_${version}_sample${SAMPLE_SIZE}_ep${NUM_EPOCHS}${suffix}_beam${NUM_BEAMS}"
      ;;
  esac
}

report_path_for_version() {
  local version="$1"
  echo "$(calc_dir_for_version "$version")/eval_report_${version}.json"
}

echo "SID-only 10k compare"
echo "  CATEGORY=${CATEGORY}"
echo "  BASE_MODEL=${BASE_MODEL}"
echo "  SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "  NUM_EPOCHS=${NUM_EPOCHS}"
echo "  NUM_BEAMS=${NUM_BEAMS}"
echo "  SID_VERSIONS=${SID_VERSIONS}"
echo "  RUN_LABEL=${RUN_LABEL}"
echo "  EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}"
echo "  LOAD_BEST_MODEL_AT_END=${LOAD_BEST_MODEL_AT_END}"
echo "  SAVE_DURING_TRAINING=${SAVE_DURING_TRAINING}"

for SID_VERSION in $SID_VERSIONS; do
  OUTPUT_DIR="$(output_dir_for_version "$SID_VERSION")"
  PRED_DIR="$(pred_dir_for_version "$SID_VERSION")"
  CALC_DIR="$(calc_dir_for_version "$SID_VERSION")"
  REPORT_JSON="$(report_path_for_version "$SID_VERSION")"

  echo
  echo "===== Train ${SID_VERSION} ====="
  if [[ -d "${OUTPUT_DIR}/final_checkpoint" ]]; then
    echo "Found final checkpoint, skip training: ${OUTPUT_DIR}/final_checkpoint"
  elif [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output dir exists but final_checkpoint is missing: $OUTPUT_DIR" >&2
    echo "Use a different SAMPLE_SIZE/NUM_EPOCHS or clean this partial directory manually after inspection." >&2
    exit 1
  else
    CATEGORY="$CATEGORY" \
    SID_VERSION="$SID_VERSION" \
    BASE_MODEL="$BASE_MODEL" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    SAMPLE_SIZE="$SAMPLE_SIZE" \
    NUM_EPOCHS="$NUM_EPOCHS" \
    SFT_TASK_MODE=sid_only \
    EARLY_STOPPING_PATIENCE="$EARLY_STOPPING_PATIENCE" \
    LOAD_BEST_MODEL_AT_END="$LOAD_BEST_MODEL_AT_END" \
    SAVE_DURING_TRAINING="$SAVE_DURING_TRAINING" \
    bash scripts/run_sft_smoke_sid_version.sh
  fi

  echo
  echo "===== Evaluate ${SID_VERSION} ====="
  if [[ "$FORCE_EVAL" != "1" && -f "$REPORT_JSON" ]]; then
    echo "Found calc_plus report, skip eval/calc: $REPORT_JSON"
  else
    CATEGORY="$CATEGORY" \
    SID_VERSION="$SID_VERSION" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    NUM_BEAMS="$NUM_BEAMS" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    PRED_DIR="$PRED_DIR" \
    CALC_DIR="$CALC_DIR" \
    bash scripts/eval_smoke_sid_version.sh
  fi
done

echo
echo "===== Summary ====="
"$PYTHON" - "$SID_VERSIONS" <<'PY'
import json
import os
import sys

versions = sys.argv[1].split()
sample_size = os.environ.get("SAMPLE_SIZE", "10000")
num_epochs = os.environ.get("NUM_EPOCHS", "1")
num_beams = os.environ.get("NUM_BEAMS", "20")
run_label = os.environ.get("RUN_LABEL", "noearly")
suffix = f"_{run_label}" if run_label else ""

def report_path(version):
    category = os.environ.get("CATEGORY", "Industrial_and_Scientific")
    if category == "Industrial_and_Scientific" and version == "text_mbk_k256_dedup":
        return f"results/calc_plus_sidonly_Industrial_text_mbk_k256_dedup_sample{sample_size}_ep{num_epochs}{suffix}_beam{num_beams}/eval_report_{version}.json"
    if category == "Industrial_and_Scientific" and version == "cs_alpha0.7_k512_dedup":
        return f"results/calc_plus_sidonly_Industrial_cs_alpha0.7_k512_dedup_sample{sample_size}_ep{num_epochs}{suffix}_beam{num_beams}/eval_report_{version}.json"
    return f"results/calc_plus_sidonly_{category}_{version}_sample{sample_size}_ep{num_epochs}{suffix}_beam{num_beams}/eval_report_{version}.json"

for version in versions:
    path = report_path(version)
    if not os.path.exists(path):
        print(version, "missing report:", path)
        continue
    report = json.load(open(path, "r", encoding="utf-8"))
    sid = report["sid_level_hr_ndcg"]
    print(
        version,
        "invalid=", report["validity"]["invalid_sid_rate"],
        "HR@10=", sid.get("hr@10"),
        "HR@20=", sid.get("hr@20"),
        "NDCG@20=", sid.get("ndcg@20"),
        "prefix_levels=", report.get("prefix_levels"),
    )
PY
