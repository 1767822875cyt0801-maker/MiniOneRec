#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run P2-0 rerank headroom diagnosis from the fixed P1 validation artifacts.

Scope is intentionally fixed:
  Industrial_and_Scientific / valid / exact

This script is CPU-only analysis. It does not run inference, train models,
touch test artifacts, or modify frozen baselines.

Environment:
  DRY_RUN      Default: 0. Set 1 to check inputs and print the Python command.
  FORCE        Default: 0. Set 1 to recompute even if valid outputs exist.
  PYTHON       Default: python
  ROOT         Default: results/stage7_validation_protocol/valid/Industrial_and_Scientific
  OUT_DIR      Default: ${ROOT}/p2_headroom

Dry run:
  DRY_RUN=1 bash scripts/run_stage7_p2_headroom_diagnosis.sh

Actual run on AutoDL:
  CUDA_VISIBLE_DEVICES="" bash scripts/run_stage7_p2_headroom_diagnosis.sh
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

CATEGORY="Industrial_and_Scientific"
EVAL_SPLIT="valid"
CANDIDATE_MODE="exact"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
ROOT="${ROOT:-results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}}"
OUT_DIR="${OUT_DIR:-${ROOT}/p2_headroom}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/stage7_p2_headroom_diagnosis.log"

TEXT_CAND_JSONL="${ROOT}/text/candidates/candidates.jsonl"
CF_CAND_JSONL="${ROOT}/cf/candidates/candidates.jsonl"
FUSION_JSONL="${ROOT}/fusion/dual_fused_candidates.jsonl"
FUSION_REPORT="${ROOT}/fusion/dual_fusion_report.json"
RERANK_JSONL="${ROOT}/rerank/reranked_candidates.jsonl"
RERANK_REPORT="${ROOT}/rerank/rerank_report.json"
SUMMARY_JSON="${ROOT}/summary/metrics_summary.json"
CONSISTENCY_JSON="${ROOT}/summary/consistency_report.json"
HEADROOM_JSON="${OUT_DIR}/headroom_summary.json"
HEADROOM_MD="${OUT_DIR}/headroom_summary.md"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

print_cmd() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

check_file() {
  local path="$1"
  [[ -f "$path" ]] || fail "Required file not found: $path"
}

ensure_valid_no_test_path() {
  local path="$1"
  case "$path" in
    results/stage7_validation_protocol/valid/Industrial_and_Scientific/*) ;;
    results/stage7_validation_protocol/valid/Industrial_and_Scientific) ;;
    */results/stage7_validation_protocol/valid/Industrial_and_Scientific/*) ;;
    */results/stage7_validation_protocol/valid/Industrial_and_Scientific) ;;
    *) fail "Path is outside fixed valid Industrial root: $path" ;;
  esac
  case "$path" in
    *"/test/"*|*"test.csv") fail "Path unexpectedly references test: $path" ;;
  esac
}

json_value() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json
import sys

path, dotted = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as f:
    value = json.load(f)
for key in dotted.split("."):
    value = value[key]
print(value)
PY
}

validate_existing_outputs() {
  "$PYTHON" - "$OUT_DIR" "$CATEGORY" "$EVAL_SPLIT" "$CANDIDATE_MODE" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
category, split, candidate_mode = sys.argv[2:5]
required = [
    "headroom_summary.json",
    "headroom_summary.csv",
    "headroom_summary.md",
    "source_coverage.csv",
    "target_rank_distribution.csv",
    "ranking_failures.jsonl",
    "data_protocol_recommendation.md",
]
missing = [name for name in required if not (out_dir / name).is_file()]
if missing:
    raise SystemExit(f"missing outputs: {missing}")
with open(out_dir / "headroom_summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)
if summary.get("overall_ok") is not True:
    raise SystemExit("headroom_summary.json overall_ok is not true")
if summary.get("category") != category:
    raise SystemExit(f"category mismatch: {summary.get('category')}")
if summary.get("split") != split:
    raise SystemExit(f"split mismatch: {summary.get('split')}")
if summary.get("candidate_mode") != candidate_mode:
    raise SystemExit(f"candidate_mode mismatch: {summary.get('candidate_mode')}")
decision = summary.get("recommendation", {}).get("decision", "not_available")
gap = summary.get("topk_metrics", {}).get("recoverable_gap_vs_heuristic@20", "not_available")
print(f"existing_ok decision={decision} recoverable_gap@20={gap}")
PY
}

echo "===== P2-0 Preflight ====="
echo "category=${CATEGORY}"
echo "split=${EVAL_SPLIT}"
echo "candidate_mode=${CANDIDATE_MODE}"
echo "root=${ROOT}"
echo "out_dir=${OUT_DIR}"

ensure_valid_no_test_path "$ROOT"
ensure_valid_no_test_path "$OUT_DIR"
for path in \
  "$TEXT_CAND_JSONL" \
  "$CF_CAND_JSONL" \
  "$FUSION_JSONL" \
  "$FUSION_REPORT" \
  "$RERANK_JSONL" \
  "$RERANK_REPORT" \
  "$SUMMARY_JSON" \
  "$CONSISTENCY_JSON"
do
  ensure_valid_no_test_path "$path"
  check_file "$path"
done

overall_ok="$(json_value "$CONSISTENCY_JSON" overall_ok)"
[[ "$overall_ok" == "True" || "$overall_ok" == "true" ]] || fail "P1 consistency_report overall_ok is not true: $overall_ok"
summary_split="$(json_value "$SUMMARY_JSON" split)"
summary_mode="$(json_value "$SUMMARY_JSON" candidate_mode)"
[[ "$summary_split" == "$EVAL_SPLIT" ]] || fail "Unexpected summary split: $summary_split"
[[ "$summary_mode" == "$CANDIDATE_MODE" ]] || fail "Unexpected candidate mode: $summary_mode"

if [[ "$FORCE" != "1" && -e "$HEADROOM_JSON" ]]; then
  echo "===== Existing P2-0 outputs ====="
  if validate_existing_outputs; then
    echo "Reuse P2-0 headroom diagnosis."
    echo "Summary: $HEADROOM_MD"
    exit 0
  fi
  fail "Existing P2-0 outputs are incomplete or invalid. Set FORCE=1 to regenerate."
fi

echo "===== P2-0 command ====="
cmd=(
  "$PYTHON"
  scripts/diagnose_stage7_rerank_headroom.py
  --root "$ROOT"
  --out-dir "$OUT_DIR"
  --category "$CATEGORY"
  --split "$EVAL_SPLIT"
  --candidate-mode "$CANDIDATE_MODE"
)
print_cmd "${cmd[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only. No P2-0 outputs were written."
  exit 0
fi

"${cmd[@]}"

echo "===== Validate P2-0 outputs ====="
validate_existing_outputs

echo
echo "Stage 7 P2-0 headroom diagnosis completed."
echo "Summary: $HEADROOM_MD"
