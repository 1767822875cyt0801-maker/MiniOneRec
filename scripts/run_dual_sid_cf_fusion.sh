#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run Dual-SID item-level fusion for Text SID + CF Behavior SID.

This script is CPU post-processing only. It does not train SFT and does not run
evaluate.py. It expects prediction JSON files for both streams to already exist,
then generates exact/prefix@3 candidates if needed, fuses them, runs rerank.py,
and writes a compact summary CSV.

Environment:
  CATEGORY          Default: Industrial_and_Scientific
  TEXT_VERSION      Default: text_mbk_k512_dedup
  BEHAVIOR_VERSION  Default: cf_k512_dedup
  SAMPLE_SIZE       Default: 30000
  NUM_EPOCHS        Default: 1
  RUN_LABEL         Default: noearly
  NUM_BEAMS         Default: 20
  MAX_CANDIDATES    Default: 500
  SOURCE_WEIGHT     Default: 4.0
  LAMBDA_TEXT       Default: 0.5
  K_RRF             Default: 60
  EVAL_SPLIT        valid or test. Default: test
  CANDIDATE_MODES   Default: "exact p3"
  RESULT_ROOT       Default: results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}
  MANIFEST          Default: data/Amazon/sid_maps/experiment_manifest.json
  FORCE_CANDIDATES  Default: 0. Set 1 to regenerate candidates.
  PYTHON            Default: python

Example:
  CATEGORY=Office_Products bash scripts/run_dual_sid_cf_fusion.sh
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
BEHAVIOR_VERSION="${BEHAVIOR_VERSION:-cf_k512_dedup}"
MANIFEST="${MANIFEST:-data/Amazon/sid_maps/experiment_manifest.json}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
SAMPLE_SIZE="${SAMPLE_SIZE:-30000}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
RUN_LABEL="${RUN_LABEL:-noearly}"
NUM_BEAMS="${NUM_BEAMS:-20}"
MAX_CANDIDATES="${MAX_CANDIDATES:-500}"
SOURCE_WEIGHT="${SOURCE_WEIGHT:-4.0}"
LAMBDA_TEXT="${LAMBDA_TEXT:-0.5}"
K_RRF="${K_RRF:-60}"
CANDIDATE_MODES="${CANDIDATE_MODES:-exact p3}"
RESULT_ROOT="${RESULT_ROOT:-results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}}"
FORCE_CANDIDATES="${FORCE_CANDIDATES:-0}"
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

short_name() {
  local version="$1"
  case "$version" in
    text_mbk_k512_dedup) echo "text_k512" ;;
    cf_k512_dedup) echo "cf_k512" ;;
    *) echo "$version" | tr '.' '_' ;;
  esac
}

embedding_path() {
  local version="$1"
  case "$version" in
    text_mbk*)
      echo "data/Amazon/index/${CATEGORY}.emb-qwen-td.npy"
      ;;
    cf*)
      echo "data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.cf_emb.npy"
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

check_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
}

run_candidates() {
  local version="$1"
  local name="$2"
  local mode="$3"
  local pred
  pred="$(prediction_path "$version")"
  local version_dir="data/Amazon/sid_versions/${version}/${CATEGORY}"
  local eval_csv
  eval_csv="$(manifest_path_or_fallback "$version" "${EVAL_SPLIT}_csv" "${version_dir}/${EVAL_SPLIT}.csv")"
  local item2sid
  item2sid="$(manifest_path_or_fallback "$version" "item2sid" "${version_dir}/item2sid.json")"
  local sid2items
  sid2items="$(manifest_path_or_fallback "$version" "sid2items" "${version_dir}/sid2items.json")"
  local valid_sid_set
  valid_sid_set="$(manifest_path_or_fallback "$version" "valid_sid_set" "${version_dir}/valid_sid_set.json")"
  local cand_dir="${RESULT_ROOT}/${version}/${mode}/candidates"
  local prefix_args=()
  if [[ "$mode" == "p3" ]]; then
    prefix_args=(--prefix-levels 3)
  elif [[ "$mode" != "exact" ]]; then
    echo "Unsupported mode: $mode" >&2
    exit 1
  fi

  check_file "$pred"
  check_file "$eval_csv"
  check_file "$item2sid"
  check_file "$sid2items"
  check_file "$valid_sid_set"

  if [[ "$FORCE_CANDIDATES" != "1" && -f "${cand_dir}/candidates.jsonl" && -f "${cand_dir}/report.json" ]]; then
    echo "Found candidates, skip: ${cand_dir}/candidates.jsonl" >&2
  else
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
      --output-report "${cand_dir}/report.json" >&2
  fi

  echo "${cand_dir}/candidates.jsonl"
}

run_fusion_and_rerank() {
  local mode="$1"
  local text_candidates="$2"
  local behavior_candidates="$3"
  local out_dir="${RESULT_ROOT}/dual_sid_${BEHAVIOR_VERSION}/${mode}"
  local behavior_dir="data/Amazon/sid_versions/${BEHAVIOR_VERSION}/${CATEGORY}"
  local train_csv
  train_csv="$(manifest_path_or_fallback "$BEHAVIOR_VERSION" "train_csv" "${behavior_dir}/train.csv")"
  local emb
  emb="$(embedding_path "$BEHAVIOR_VERSION")"
  local row_index="data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.row_index.json"
  local eval_csv
  eval_csv="$(manifest_path_or_fallback "$BEHAVIOR_VERSION" "${EVAL_SPLIT}_csv" "${behavior_dir}/${EVAL_SPLIT}.csv")"

  check_file "$train_csv"
  check_file "$emb"
  check_file "$row_index"
  check_file "$eval_csv"

  "$PYTHON" fuse_dual_sid_candidates.py \
    --text-candidates "$text_candidates" \
    --behavior-candidates "$behavior_candidates" \
    --eval-csv "$eval_csv" \
    --eval-split "$EVAL_SPLIT" \
    --out-dir "$out_dir" \
    --ks 1 3 5 10 20 50 \
    --lambda-text "$LAMBDA_TEXT" \
    --k-rrf "$K_RRF"

  "$PYTHON" rerank.py \
    --candidate-jsonl "${out_dir}/dual_fused_candidates.jsonl" \
    --train-csv "$train_csv" \
    --eval-split "$EVAL_SPLIT" \
    --item-emb "$emb" \
    --row-index "$row_index" \
    --source-weight "$SOURCE_WEIGHT" \
    --topk 1 3 5 10 20 50 \
    --output-dir "${out_dir}/rerank"
}

text_name="$(short_name "$TEXT_VERSION")"
behavior_name="$(short_name "$BEHAVIOR_VERSION")"

echo "Dual-SID CF fusion"
echo "  CATEGORY=${CATEGORY}"
echo "  TEXT_VERSION=${TEXT_VERSION}"
echo "  BEHAVIOR_VERSION=${BEHAVIOR_VERSION}"
echo "  EVAL_SPLIT=${EVAL_SPLIT}"
echo "  RESULT_ROOT=${RESULT_ROOT}"
echo "  SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "  NUM_BEAMS=${NUM_BEAMS}"
echo "  MAX_CANDIDATES=${MAX_CANDIDATES}"
echo "  LAMBDA_TEXT=${LAMBDA_TEXT}"
echo "  K_RRF=${K_RRF}"
echo "  SOURCE_WEIGHT=${SOURCE_WEIGHT}"
echo "  CANDIDATE_MODES=${CANDIDATE_MODES}"

for mode in $CANDIDATE_MODES; do
  text_candidates="$(run_candidates "$TEXT_VERSION" "$text_name" "$mode")"
  behavior_candidates="$(run_candidates "$BEHAVIOR_VERSION" "$behavior_name" "$mode")"
  run_fusion_and_rerank "$mode" "$text_candidates" "$behavior_candidates"
done

summary_path="${RESULT_ROOT}/dual_sid_${BEHAVIOR_VERSION}/summary.csv"
"$PYTHON" - "$RESULT_ROOT" "$CATEGORY" "$BEHAVIOR_VERSION" "$EVAL_SPLIT" "$CANDIDATE_MODES" "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

result_root, category, behavior_version, eval_split, candidate_modes, summary_path = sys.argv[1:]
result_root = Path(result_root)
candidate_modes = candidate_modes.split()
rows = []

for mode in candidate_modes:
    base = result_root / f"dual_sid_{behavior_version}" / mode
    fusion_path = base / "dual_fusion_report.json"
    rerank_path = base / "rerank" / "rerank_report.json"
    if not fusion_path.exists() or not rerank_path.exists():
        continue
    fusion = json.load(open(fusion_path, "r", encoding="utf-8"))
    rerank = json.load(open(rerank_path, "r", encoding="utf-8"))
    union_hr20 = fusion["union_recall"]["union_hit@20"]
    rerank_hr20 = rerank["after_rerank"]["hr@20"]
    rows.append({
        "category": category,
        "eval_split": eval_split,
        "mode": mode,
        "text_hr20": fusion["text_stream"]["hr@20"],
        "behavior_hr20": fusion["behavior_stream"]["hr@20"],
        "union_hr20": union_hr20,
        "union_gain_hr20": fusion["union_recall"]["union_gain_over_text@20"],
        "behavior_only_hit20": fusion["union_recall"]["behavior_only_hit_count@20"],
        "minrank_hr20": fusion["min_rank_fusion"]["hr@20"],
        "rrf_hr20": fusion["rrf_fusion"]["hr@20"],
        "rrf_ndcg20": fusion["rrf_fusion"]["ndcg@20"],
        "rerank_hr20": rerank_hr20,
        "rerank_ndcg20": rerank["after_rerank"]["ndcg@20"],
        "union_rerank_gap_hr20": union_hr20 - rerank_hr20,
    })

df = pd.DataFrame(rows)
Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(summary_path, index=False)
print(df.to_string(index=False))
print(f"\nWrote: {summary_path}")
PY

echo
echo "Dual-SID CF fusion completed."
