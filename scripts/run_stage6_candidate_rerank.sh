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
  EVAL_SPLIT     valid or test. Default: test
  CANDIDATE_MODES Default: "exact p3"
  RESULT_ROOT    Default: results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}
  MANIFEST       Default: data/Amazon/sid_maps/experiment_manifest.json
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
MANIFEST="${MANIFEST:-data/Amazon/sid_maps/experiment_manifest.json}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
SAMPLE_SIZE="${SAMPLE_SIZE:-30000}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
RUN_LABEL="${RUN_LABEL:-noearly}"
NUM_BEAMS="${NUM_BEAMS:-20}"
MAX_CANDIDATES="${MAX_CANDIDATES:-500}"
SOURCE_WEIGHT="${SOURCE_WEIGHT:-4.0}"
CANDIDATE_MODES="${CANDIDATE_MODES:-exact p3}"
RESULT_ROOT="${RESULT_ROOT:-results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}}"
PYTHON="${PYTHON:-python}"

if [[ "$EVAL_SPLIT" != "valid" && "$EVAL_SPLIT" != "test" ]]; then
  echo "EVAL_SPLIT must be valid or test, got: $EVAL_SPLIT" >&2
  exit 2
fi

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
  echo "${RESULT_ROOT}/${version}/predictions/predictions.json"
}

manifest_path_or_fallback() {
  local version="$1"
  local key="$2"
  local fallback="$3"
  "$PYTHON" - "$MANIFEST" "$version" "$CATEGORY" "$key" "$fallback" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, version, category, key, fallback = sys.argv[1:6]
entry = {}
if Path(manifest_path).exists():
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    entry = manifest.get("sid_versions", {}).get(version, {}).get(category, {})
print(entry.get(key) or fallback)
PY
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
  local eval_csv
  eval_csv="$(manifest_path_or_fallback "$version" "${EVAL_SPLIT}_csv" "${version_dir}/${EVAL_SPLIT}.csv")"
  local train_csv
  train_csv="$(manifest_path_or_fallback "$version" "train_csv" "${version_dir}/train.csv")"
  local item2sid
  item2sid="$(manifest_path_or_fallback "$version" "item2sid" "${version_dir}/item2sid.json")"
  local sid2items
  sid2items="$(manifest_path_or_fallback "$version" "sid2items" "${version_dir}/sid2items.json")"
  local valid_sid_set
  valid_sid_set="$(manifest_path_or_fallback "$version" "valid_sid_set" "${version_dir}/valid_sid_set.json")"
  local cand_dir="${RESULT_ROOT}/${version}/${mode}/candidates"
  local rerank_dir="${RESULT_ROOT}/${version}/${mode}/rerank"

  check_required "$pred"
  check_required "$emb"
  check_required "$row_index_path"
  check_required "$eval_csv"
  check_required "$train_csv"
  check_required "$item2sid"
  check_required "$sid2items"
  check_required "$valid_sid_set"

  echo
  echo "===== ${CATEGORY} / ${version} / ${mode} ====="
  "$PYTHON" evaluate_candidates.py \
    --prediction-file "$pred" \
    --eval-csv "$eval_csv" \
    --eval-split "$EVAL_SPLIT" \
    --item2sid "$item2sid" \
    --sid2items "$sid2items" \
    --valid-sid-set "$valid_sid_set" \
    "${prefix_args[@]}" \
    --max-pred-sids "$NUM_BEAMS" \
    --max-candidates "$MAX_CANDIDATES" \
    --topk 1 3 5 10 20 50 \
    --output-jsonl "${cand_dir}/candidates.jsonl" \
    --output-report "${cand_dir}/report.json"

  "$PYTHON" rerank.py \
    --candidate-jsonl "${cand_dir}/candidates.jsonl" \
    --train-csv "$train_csv" \
    --eval-split "$EVAL_SPLIT" \
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
echo "  EVAL_SPLIT=${EVAL_SPLIT}"
echo "  RESULT_ROOT=${RESULT_ROOT}"
echo "  SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "  NUM_EPOCHS=${NUM_EPOCHS}"
echo "  RUN_LABEL=${RUN_LABEL}"
echo "  NUM_BEAMS=${NUM_BEAMS}"
echo "  MAX_CANDIDATES=${MAX_CANDIDATES}"
echo "  SOURCE_WEIGHT=${SOURCE_WEIGHT}"
echo "  CANDIDATE_MODES=${CANDIDATE_MODES}"

for mode in $CANDIDATE_MODES; do
  run_one "$TEXT_VERSION" "$text_name" "$mode"
  run_one "$CS_VERSION" "$cs_name" "$mode"
done

summary_path="${RESULT_ROOT}/stage6_candidate_rerank_summary.csv"
"$PYTHON" - "$RESULT_ROOT" "$TEXT_VERSION" "$CS_VERSION" "$EVAL_SPLIT" "$CANDIDATE_MODES" "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

result_root, text_version, cs_version, eval_split, candidate_modes, summary_path = sys.argv[1:]
result_root = Path(result_root)
candidate_modes = candidate_modes.split()

rows = []

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def add(version, setting):
    base = result_root / version / setting
    cand = load(base / "candidates" / "report.json")
    rerank = load(base / "rerank" / "rerank_report.json")
    rows.append({
        "eval_split": eval_split,
        "version": version,
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

for version_name in [text_version, cs_version]:
    for mode in candidate_modes:
        add(version_name, mode)

df = pd.DataFrame(rows)
Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(summary_path, index=False)
print(df.to_string(index=False))
print(f"\nWrote: {summary_path}")
PY

echo
echo "Stage 6 candidate/rerank completed."
