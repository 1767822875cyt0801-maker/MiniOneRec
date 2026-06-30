#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run Stage 6 candidate expansion and lightweight rerank for Text-vs-CS SID versions.

Default workflow:
  1. Run exact-only candidate expansion + rerank for Text and CS.
  2. Run prefix@3 candidate expansion + rerank for Text and CS.
  3. Write a compact summary CSV.

Environment overrides:
  CATEGORY       Default: Industrial_and_Scientific
  TEXT_VERSION   Default: text_mbk_k512_dedup
  CS_VERSION     Default: cs_alpha0.2_k512_dedup
  SAMPLE_SIZE    Default: 30000
  NUM_EPOCHS     Default: 1
  RUN_LABEL      Default: noearly
  NUM_BEAMS      Default: 20
  MAX_CANDIDATES Default: 500
  SOURCE_WEIGHT  Default: 4.0
  PYTHON         Default: python

Run:
  CATEGORY=Office_Products bash scripts/run_stage6_candidate_rerank.sh
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
TEXT_VERSION="${TEXT_VERSION:-text_mbk_k512_dedup}"
CS_VERSION="${CS_VERSION:-cs_alpha0.2_k512_dedup}"
SAMPLE_SIZE="${SAMPLE_SIZE:-30000}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
RUN_LABEL="${RUN_LABEL:-noearly}"
NUM_BEAMS="${NUM_BEAMS:-20}"
MAX_CANDIDATES="${MAX_CANDIDATES:-500}"
SOURCE_WEIGHT="${SOURCE_WEIGHT:-4.0}"
PYTHON="${PYTHON:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

suffix=""
if [[ -n "$RUN_LABEL" ]]; then
  suffix="_${RUN_LABEL}"
fi

row_index_path="data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.row_index.json"

prediction_path() {
  local version="$1"
  echo "results/eval_sidonly_${CATEGORY}_${version}_sample${SAMPLE_SIZE}_ep${NUM_EPOCHS}${suffix}_beam${NUM_BEAMS}/predictions.json"
}

embedding_path() {
  local version="$1"
  case "$version" in
    text_mbk*)
      echo "data/Amazon/index/${CATEGORY}.emb-qwen-td.npy"
      ;;
    cs_alpha*)
      local rest="${version#cs_alpha}"
      local alpha="${rest%%_*}"
      echo "data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.cs_emb_alpha${alpha}.npy"
      ;;
    *)
      echo "Unsupported embedding version: ${version}" >&2
      return 1
      ;;
  esac
}

short_name() {
  local version="$1"
  case "$version" in
    text_mbk_k512_dedup) echo "text_k512" ;;
    cs_alpha0.2_k512_dedup) echo "cs_alpha02_k512" ;;
    *) echo "$version" | tr '.' '_' ;;
  esac
}

check_required() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
}

run_one() {
  local version="$1"
  local name="$2"
  local mode="$3"
  local prefix_args=()
  if [[ "$mode" == "p3" ]]; then
    prefix_args=(--prefix-levels 3)
  elif [[ "$mode" != "exact" ]]; then
    echo "Unsupported mode: $mode" >&2
    exit 1
  fi

  local pred
  pred="$(prediction_path "$version")"
  local emb
  emb="$(embedding_path "$version")"
  local version_dir="data/Amazon/sid_versions/${version}/${CATEGORY}"
  local cand_dir="results/candidates_${CATEGORY}_${name}_30k_${mode}_c${MAX_CANDIDATES}_v2"
  local rerank_dir="results/rerank_${CATEGORY}_${name}_30k_${mode}_c${MAX_CANDIDATES}_sourcew${SOURCE_WEIGHT}_v2"

  check_required "$pred"
  check_required "$emb"
  check_required "$row_index_path"
  check_required "${version_dir}/test.csv"
  check_required "${version_dir}/train.csv"
  check_required "${version_dir}/item2sid.json"
  check_required "${version_dir}/sid2items.json"
  check_required "${version_dir}/valid_sid_set.json"

  echo
  echo "===== ${CATEGORY} / ${version} / ${mode} ====="
  "$PYTHON" evaluate_candidates.py \
    --prediction-file "$pred" \
    --test-csv "${version_dir}/test.csv" \
    --item2sid "${version_dir}/item2sid.json" \
    --sid2items "${version_dir}/sid2items.json" \
    --valid-sid-set "${version_dir}/valid_sid_set.json" \
    "${prefix_args[@]}" \
    --max-pred-sids "$NUM_BEAMS" \
    --max-candidates "$MAX_CANDIDATES" \
    --topk 1 3 5 10 20 50 \
    --output-jsonl "${cand_dir}/candidates.jsonl" \
    --output-report "${cand_dir}/report.json"

  "$PYTHON" rerank.py \
    --candidate-jsonl "${cand_dir}/candidates.jsonl" \
    --train-csv "${version_dir}/train.csv" \
    --item-emb "$emb" \
    --row-index "$row_index_path" \
    --source-weight "$SOURCE_WEIGHT" \
    --topk 1 3 5 10 20 50 \
    --output-dir "$rerank_dir"
}

text_name="$(short_name "$TEXT_VERSION")"
cs_name="$(short_name "$CS_VERSION")"

echo "Stage 6 candidate/rerank"
echo "  CATEGORY=${CATEGORY}"
echo "  TEXT_VERSION=${TEXT_VERSION}"
echo "  CS_VERSION=${CS_VERSION}"
echo "  SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "  NUM_EPOCHS=${NUM_EPOCHS}"
echo "  RUN_LABEL=${RUN_LABEL}"
echo "  NUM_BEAMS=${NUM_BEAMS}"
echo "  MAX_CANDIDATES=${MAX_CANDIDATES}"
echo "  SOURCE_WEIGHT=${SOURCE_WEIGHT}"

run_one "$TEXT_VERSION" "$text_name" exact
run_one "$CS_VERSION" "$cs_name" exact
run_one "$TEXT_VERSION" "$text_name" p3
run_one "$CS_VERSION" "$cs_name" p3

summary_path="results/stage6_${CATEGORY}_candidate_rerank_summary.csv"
"$PYTHON" - "$CATEGORY" "$text_name" "$cs_name" "$MAX_CANDIDATES" "$SOURCE_WEIGHT" "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

category, text_name, cs_name, max_candidates, source_weight, summary_path = sys.argv[1:]

rows = []

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def add(name, setting):
    cand = load(f"results/candidates_{category}_{name}_30k_{setting}_c{max_candidates}_v2/report.json")
    rerank = load(
        f"results/rerank_{category}_{name}_30k_{setting}_c{max_candidates}_sourcew{source_weight}_v2/rerank_report.json"
    )
    rows.append({
        "version": name,
        "setting": setting,
        "mean_candidates": cand["candidate_count"]["mean"],
        "candidate_pool_recall": cand["candidate_pool_recall"]["target_in_pool_rate"],
        "target_in_exact_rate": cand["target_in_source"].get("exact", {}).get("rate", 0.0),
        "target_in_prefix3_rate": cand["target_in_source"].get("prefix@3", {}).get("rate", 0.0),
        "avg_rank_before_if_hit": cand["candidate_pool_recall"]["avg_rank_before_rerank_if_hit"],
        "before_hr20": cand["item_level_before_rerank"]["hr@20"],
        "before_ndcg20": cand["item_level_before_rerank"]["ndcg@20"],
        "after_hr20": rerank["after_rerank"]["hr@20"],
        "after_ndcg20": rerank["after_rerank"]["ndcg@20"],
    })

for version_name in [text_name, cs_name]:
    add(version_name, "exact")
    add(version_name, "p3")

df = pd.DataFrame(rows)
Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(summary_path, index=False)
print(df.to_string(index=False))
print(f"\nWrote: {summary_path}")
PY

echo
echo "Stage 6 candidate/rerank completed."
