#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the Industrial same-k SID ablation in one command.

Default workflow:
  1. Generate and rewrite:
     - text_mbk_k512_dedup
     - cs_alpha0.7_k256_dedup
  2. Update manifest, run analyze_sid.py and check_sid_stage0.py.
  3. Run 10k SID-only SFT smoke for the two new versions.
  4. Print a four-version summary and run same-k per-sample comparisons.

Existing completed artifacts are reused. Partial output directories cause a
failure unless OVERWRITE=1 is set.

Environment overrides:
  CATEGORY        Default: Industrial_and_Scientific
  BASE_MODEL      Default: /root/autodl-tmp/models/Qwen2.5-0.5B
  MANIFEST        Default: data/Amazon/sid_maps/experiment_manifest.json
  OUTPUT_ROOT     Default: data/Amazon/sid_versions
  CS_EMB_ROOT     Default: data/Amazon/cs_embeddings
  TEXT_EMB_ROOT   Default: data/Amazon/index
  SAMPLE_SIZE     Default: 10000
  NUM_EPOCHS      Default: 1
  NUM_BEAMS       Default: 20
  RUN_LABEL       Default: noearly
  OVERWRITE       Default: 0
  FORCE_EVAL      Default: 0

Run:
  bash scripts/run_sid_k_ablation_10k.sh
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
MANIFEST="${MANIFEST:-data/Amazon/sid_maps/experiment_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/Amazon/sid_versions}"
CS_EMB_ROOT="${CS_EMB_ROOT:-data/Amazon/cs_embeddings}"
TEXT_EMB_ROOT="${TEXT_EMB_ROOT:-data/Amazon/index}"
SAMPLE_SIZE="${SAMPLE_SIZE:-10000}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_BEAMS="${NUM_BEAMS:-20}"
RUN_LABEL="${RUN_LABEL:-noearly}"
OVERWRITE="${OVERWRITE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
PYTHON="${PYTHON:-python}"

NEW_VERSIONS="${NEW_VERSIONS:-text_mbk_k512_dedup cs_alpha0.7_k256_dedup}"
ALL_VERSIONS="${ALL_VERSIONS:-text_mbk_k256_dedup text_mbk_k512_dedup cs_alpha0.7_k256_dedup cs_alpha0.7_k512_dedup}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_DISABLED=true

TRAIN_CSV="${TRAIN_CSV:-$(ls "data/Amazon/train/${CATEGORY}"*.csv | head -1)}"
VALID_CSV="${VALID_CSV:-$(ls "data/Amazon/valid/${CATEGORY}"*.csv | head -1)}"
TEST_CSV="${TEST_CSV:-$(ls "data/Amazon/test/${CATEGORY}"*.csv | head -1)}"
ITEM_JSON="${ITEM_JSON:-data/Amazon/index/${CATEGORY}.item.json}"
ITEM_ORDER="${ITEM_ORDER:-${CS_EMB_ROOT}/${CATEGORY}/${CATEGORY}.item_order.json}"
ROW_INDEX="${ROW_INDEX:-${CS_EMB_ROOT}/${CATEGORY}/${CATEGORY}.row_index.json}"

for required_path in "$TRAIN_CSV" "$VALID_CSV" "$TEST_CSV" "$ITEM_JSON" "$ITEM_ORDER" "$ROW_INDEX" "$MANIFEST"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Required file not found: $required_path" >&2
    exit 1
  fi
done

embedding_path_for_version() {
  local version="$1"
  case "$version" in
    text_mbk*)
      echo "${TEXT_EMB_ROOT}/${CATEGORY}.emb-qwen-td.npy"
      ;;
    cs_alpha*)
      local rest="${version#cs_alpha}"
      local alpha="${rest%%_*}"
      echo "${CS_EMB_ROOT}/${CATEGORY}/${CATEGORY}.cs_emb_alpha${alpha}.npy"
      ;;
    *)
      echo "Unsupported version for this ablation: $version" >&2
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
    echo "256"
  fi
}

version_dir() {
  local version="$1"
  echo "${OUTPUT_ROOT}/${version}/${CATEGORY}"
}

has_generation_outputs() {
  local dir="$1"
  [[ -f "${dir}/index.json" && -f "${dir}/info.txt" && -f "${dir}/item2sid.json" && -f "${dir}/sid2items.json" && -f "${dir}/valid_sid_set.json" && -f "${dir}/item_mapping.json" ]]
}

has_rewritten_csvs() {
  local dir="$1"
  [[ -f "${dir}/train.csv" && -f "${dir}/valid.csv" && -f "${dir}/test.csv" ]]
}

run_generation_and_rewrite() {
  local version="$1"
  local dir
  dir="$(version_dir "$version")"
  local report_dir="${dir}/reports"
  local emb_path
  emb_path="$(embedding_path_for_version "$version")"
  local codebook_size
  codebook_size="$(codebook_size_for_version "$version")"

  if [[ ! -f "$emb_path" ]]; then
    echo "Embedding file not found for ${version}: ${emb_path}" >&2
    exit 1
  fi

  echo
  echo "===== Prepare SID version: ${version} ====="
  if has_generation_outputs "$dir"; then
    echo "Generation artifacts already exist, skip generation: $dir"
  elif [[ -e "$dir" && "$OVERWRITE" != "1" ]]; then
    echo "Partial version dir exists, refusing to continue without OVERWRITE=1: $dir" >&2
    exit 1
  else
    local overwrite_args=()
    if [[ "$OVERWRITE" == "1" ]]; then
      overwrite_args=(--overwrite)
    fi
    "$PYTHON" run_rqkmeans_with_emb.py \
      --category "$CATEGORY" \
      --emb-path "$emb_path" \
      --item-order "$ITEM_ORDER" \
      --row-index "$ROW_INDEX" \
      --sid-version "$version" \
      --output-root "$OUTPUT_ROOT" \
      --item-json "$ITEM_JSON" \
      --num-levels 3 \
      --codebook-size "$codebook_size" \
      --seed 42 \
      --dedup-mode append \
      "${overwrite_args[@]}"
  fi

  if has_rewritten_csvs "$dir"; then
    echo "Rewritten CSVs already exist, skip rewrite: $dir"
  else
    "$PYTHON" rewrite_sid_csv.py \
      --input-csv "$TRAIN_CSV" \
      --output-csv "${dir}/train.csv" \
      --item2sid "${dir}/item2sid.json" \
      --report-path "${report_dir}/rewrite_train_report.json"
    "$PYTHON" rewrite_sid_csv.py \
      --input-csv "$VALID_CSV" \
      --output-csv "${dir}/valid.csv" \
      --item2sid "${dir}/item2sid.json" \
      --report-path "${report_dir}/rewrite_valid_report.json"
    "$PYTHON" rewrite_sid_csv.py \
      --input-csv "$TEST_CSV" \
      --output-csv "${dir}/test.csv" \
      --item2sid "${dir}/item2sid.json" \
      --report-path "${report_dir}/rewrite_test_report.json"
  fi

  "$PYTHON" update_sid_manifest.py \
    --manifest "$MANIFEST" \
    --category "$CATEGORY" \
    --sid-version "$version" \
    --index "${dir}/index.json" \
    --info "${dir}/info.txt" \
    --item2sid "${dir}/item2sid.json" \
    --sid2items "${dir}/sid2items.json" \
    --valid-sid-set "${dir}/valid_sid_set.json" \
    --item-mapping "${dir}/item_mapping.json" \
    --train-csv "${dir}/train.csv" \
    --valid-csv "${dir}/valid.csv" \
    --test-csv "${dir}/test.csv"

  "$PYTHON" analyze_sid.py \
    --manifest "$MANIFEST" \
    --sid-version "$version" \
    --categories "$CATEGORY" \
    --output-dir "${report_dir}/analysis"

  "$PYTHON" check_sid_stage0.py \
    --manifest "$MANIFEST" \
    --category "$CATEGORY" \
    --sid-version "$version" \
    --report-path "${report_dir}/stage0_check_report.json"
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

per_sample_csv_for_version() {
  local version="$1"
  echo "$(calc_dir_for_version "$version")/per_sample_eval.csv"
}

report_json_for_version() {
  local version="$1"
  echo "$(calc_dir_for_version "$version")/eval_report_${version}.json"
}

echo "SID same-k ablation"
echo "  CATEGORY=${CATEGORY}"
echo "  BASE_MODEL=${BASE_MODEL}"
echo "  NEW_VERSIONS=${NEW_VERSIONS}"
echo "  ALL_VERSIONS=${ALL_VERSIONS}"
echo "  SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "  NUM_EPOCHS=${NUM_EPOCHS}"
echo "  NUM_BEAMS=${NUM_BEAMS}"
echo "  RUN_LABEL=${RUN_LABEL}"
echo "  OVERWRITE=${OVERWRITE}"

for version in $NEW_VERSIONS; do
  run_generation_and_rewrite "$version"
done

echo
echo "===== Run 10k SID-only smoke for new versions ====="
CATEGORY="$CATEGORY" \
BASE_MODEL="$BASE_MODEL" \
SAMPLE_SIZE="$SAMPLE_SIZE" \
NUM_EPOCHS="$NUM_EPOCHS" \
NUM_BEAMS="$NUM_BEAMS" \
RUN_LABEL="$RUN_LABEL" \
EARLY_STOPPING_PATIENCE=0 \
LOAD_BEST_MODEL_AT_END=False \
FORCE_EVAL="$FORCE_EVAL" \
SID_VERSIONS="$NEW_VERSIONS" \
bash scripts/tmp_run_sidonly_10k_compare.sh

echo
echo "===== Four-version summary ====="
"$PYTHON" - "$ALL_VERSIONS" "$SAMPLE_SIZE" "$NUM_EPOCHS" "$NUM_BEAMS" "$RUN_LABEL" "$CATEGORY" <<'PY'
import json
import sys
from pathlib import Path

versions = sys.argv[1].split()
sample_size, num_epochs, num_beams, run_label, category = sys.argv[2:7]
suffix = f"_{run_label}" if run_label else ""

def calc_dir(version: str) -> Path:
    if version == "text_mbk_k256_dedup":
        return Path(f"results/calc_plus_sidonly_Industrial_text_mbk_k256_dedup_sample{sample_size}_ep{num_epochs}{suffix}_beam{num_beams}")
    if version == "cs_alpha0.7_k512_dedup":
        return Path(f"results/calc_plus_sidonly_Industrial_cs_alpha0.7_k512_dedup_sample{sample_size}_ep{num_epochs}{suffix}_beam{num_beams}")
    return Path(f"results/calc_plus_sidonly_{category}_{version}_sample{sample_size}_ep{num_epochs}{suffix}_beam{num_beams}")

for version in versions:
    path = calc_dir(version) / f"eval_report_{version}.json"
    print(f"\n== {version}")
    if not path.exists():
        print(f"missing: {path}")
        continue
    report = json.load(open(path, "r", encoding="utf-8"))
    sid = report["sid_level_hr_ndcg"]
    print("invalid:", report["validity"]["invalid_sid_rate"])
    print("HR@1:", sid.get("hr@1"))
    print("HR@10:", sid.get("hr@10"))
    print("HR@20:", sid.get("hr@20"))
    print("NDCG@20:", sid.get("ndcg@20"))
    for level, metrics in report.get("prefix_hit", {}).items():
        print(f"{level}_hit@20:", metrics.get(f"{level}_hit@20"))
PY

echo
echo "===== Same-k comparisons ====="
K256_TEXT="$(per_sample_csv_for_version text_mbk_k256_dedup)"
K256_CS="$(per_sample_csv_for_version cs_alpha0.7_k256_dedup)"
K512_TEXT="$(per_sample_csv_for_version text_mbk_k512_dedup)"
K512_CS="$(per_sample_csv_for_version cs_alpha0.7_k512_dedup)"

if [[ -f "$K256_TEXT" && -f "$K256_CS" ]]; then
  "$PYTHON" compare_sft_rl_outputs.py \
    --sft-csv "$K256_TEXT" \
    --rl-csv "$K256_CS" \
    --output-dir results/compare_text_k256_vs_cs_k256_sidonly_10k_noearly_beam20 \
    --topk 1 3 5 10 20 \
    --focus-k 20
else
  echo "Skipping k256 compare; missing: $K256_TEXT or $K256_CS"
fi

if [[ -f "$K512_TEXT" && -f "$K512_CS" ]]; then
  "$PYTHON" compare_sft_rl_outputs.py \
    --sft-csv "$K512_TEXT" \
    --rl-csv "$K512_CS" \
    --output-dir results/compare_text_k512_vs_cs_k512_sidonly_10k_noearly_beam20 \
    --topk 1 3 5 10 20 \
    --focus-k 20
else
  echo "Skipping k512 compare; missing: $K512_TEXT or $K512_CS"
fi

echo
echo "SID same-k ablation completed."
