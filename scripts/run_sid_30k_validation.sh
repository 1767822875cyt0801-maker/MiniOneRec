#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run medium-scale SID-only validation for selected SID versions.

Default workflow:
  1. Train/evaluate selected SID versions with sample=30000, 1 epoch, sid_only, no early stopping, beam20.
  2. Run grouped diagnostics for each per_sample_eval.csv.
  3. Write a compact grid summary CSV.
  4. Run per-sample comparisons for the key pairs when available.

Environment overrides:
  CATEGORY       Default: Industrial_and_Scientific
  BASE_MODEL     Default: /root/autodl-tmp/models/Qwen2.5-0.5B
  SID_VERSIONS   Default: "text_mbk_k512_dedup cs_alpha0.2_k512_dedup"
  SAMPLE_SIZE    Default: 30000
  NUM_EPOCHS     Default: 1
  NUM_BEAMS      Default: 20
  RUN_LABEL      Default: noearly
  FORCE_EVAL     Default: 0. Set 1 to rerun eval/calc even if report exists.
  SAVE_DURING_TRAINING Default: False. Keep 30k validation disk-light.

Examples:
  bash scripts/run_sid_30k_validation.sh

  SID_VERSIONS="text_mbk_k512_dedup cs_alpha0.2_k512_dedup cs_alpha0.2_k256_dedup" \
  bash scripts/run_sid_30k_validation.sh
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
SID_VERSIONS="${SID_VERSIONS:-text_mbk_k512_dedup cs_alpha0.2_k512_dedup}"
SAMPLE_SIZE="${SAMPLE_SIZE:-30000}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_BEAMS="${NUM_BEAMS:-20}"
RUN_LABEL="${RUN_LABEL:-noearly}"
FORCE_EVAL="${FORCE_EVAL:-0}"
SAVE_DURING_TRAINING="${SAVE_DURING_TRAINING:-False}"
PYTHON="${PYTHON:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

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

per_sample_for_version() {
  local version="$1"
  echo "$(calc_dir_for_version "$version")/per_sample_eval.csv"
}

echo "SID 30k validation"
echo "  CATEGORY=${CATEGORY}"
echo "  BASE_MODEL=${BASE_MODEL}"
echo "  SID_VERSIONS=${SID_VERSIONS}"
echo "  SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "  NUM_EPOCHS=${NUM_EPOCHS}"
echo "  NUM_BEAMS=${NUM_BEAMS}"
echo "  RUN_LABEL=${RUN_LABEL}"
echo "  SAVE_DURING_TRAINING=${SAVE_DURING_TRAINING}"

CATEGORY="$CATEGORY" \
BASE_MODEL="$BASE_MODEL" \
SAMPLE_SIZE="$SAMPLE_SIZE" \
NUM_EPOCHS="$NUM_EPOCHS" \
NUM_BEAMS="$NUM_BEAMS" \
RUN_LABEL="$RUN_LABEL" \
EARLY_STOPPING_PATIENCE=0 \
LOAD_BEST_MODEL_AT_END=False \
SAVE_DURING_TRAINING="$SAVE_DURING_TRAINING" \
FORCE_EVAL="$FORCE_EVAL" \
SID_VERSIONS="$SID_VERSIONS" \
bash scripts/tmp_run_sidonly_10k_compare.sh

echo
echo "===== Group diagnostics ====="
for version in $SID_VERSIONS; do
  per_sample="$(per_sample_for_version "$version")"
  if [[ ! -f "$per_sample" ]]; then
    echo "Missing per_sample_eval for ${version}: ${per_sample}" >&2
    exit 1
  fi
  out_dir="$(dirname "$per_sample")"
  "$PYTHON" scripts/diagnose_sid_smoke_by_groups.py \
    --per-sample-eval "$per_sample" \
    --output-json "${out_dir}/diagnose_${version}.json" \
    --output-csv "${out_dir}/diagnose_${version}.csv" \
    --focus-k "$NUM_BEAMS"
done

summary_path="results/sidonly_smoke_grid_${CATEGORY}_summary_${SAMPLE_SIZE}.csv"
"$PYTHON" scripts/summarize_sidonly_smoke_grid.py \
  --category "$CATEGORY" \
  --sample-size "$SAMPLE_SIZE" \
  --num-epochs "$NUM_EPOCHS" \
  --run-label "$RUN_LABEL" \
  --num-beams "$NUM_BEAMS" \
  --versions $SID_VERSIONS \
  --output-csv "$summary_path"

echo
echo "===== Key table ====="
"$PYTHON" - "$summary_path" <<'PY'
import pandas as pd
import sys

path = sys.argv[1]
df = pd.read_csv(path)
cols = [
    "sid_version",
    "overall_hr20",
    "overall_ndcg20",
    "len3_hr20",
    "len4_hr20",
    "head_hr20",
    "mid_hr20",
    "tail_hr20",
    "prefix1_hr20_len4",
    "prefix2_hr20_len4",
    "prefix3_hr20_len4",
    "prefix4_hr20_len4",
]
available = [col for col in cols if col in df.columns]
print(df[available].to_string(index=False))
PY

echo
echo "===== Pairwise comparisons ====="
text_k512="$(per_sample_for_version text_mbk_k512_dedup)"
cs02_k512="$(per_sample_for_version cs_alpha0.2_k512_dedup)"
cs02_k256="$(per_sample_for_version cs_alpha0.2_k256_dedup)"

if [[ -f "$text_k512" && -f "$cs02_k512" ]]; then
  "$PYTHON" compare_sft_rl_outputs.py \
    --sft-csv "$text_k512" \
    --rl-csv "$cs02_k512" \
    --output-dir "results/compare_${CATEGORY}_text_k512_vs_cs_alpha02_k512_sidonly_${SAMPLE_SIZE}_${RUN_LABEL}_beam${NUM_BEAMS}" \
    --topk 1 3 5 10 "$NUM_BEAMS" \
    --focus-k "$NUM_BEAMS"
else
  echo "Skipping text k512 vs cs alpha0.2 k512 compare; missing one input."
fi

if [[ -f "$cs02_k256" && -f "$cs02_k512" ]]; then
  "$PYTHON" compare_sft_rl_outputs.py \
    --sft-csv "$cs02_k256" \
    --rl-csv "$cs02_k512" \
    --output-dir "results/compare_${CATEGORY}_cs_alpha02_k256_vs_k512_sidonly_${SAMPLE_SIZE}_${RUN_LABEL}_beam${NUM_BEAMS}" \
    --topk 1 3 5 10 "$NUM_BEAMS" \
    --focus-k "$NUM_BEAMS"
else
  echo "Skipping cs alpha0.2 k256 vs k512 compare; missing one input."
fi

echo
echo "SID 30k validation completed."
