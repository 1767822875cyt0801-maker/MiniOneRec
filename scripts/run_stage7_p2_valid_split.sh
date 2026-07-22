#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run P2-1: build the fixed valid_fit/valid_select split manifest.

Scope is fixed:
  Industrial_and_Scientific / valid / exact

This is CPU-only protocol work. It does not train, does not run inference,
does not evaluate test, and does not modify frozen baselines.

Environment:
  DRY_RUN    Default: 0. Set 1 to check inputs and print the Python command.
  FORCE      Default: 0. Set 1 to regenerate even if valid outputs exist.
  PYTHON     Default: python
  ROOT       Default: results/stage7_validation_protocol/valid/Industrial_and_Scientific
  OUT_DIR    Default: ${ROOT}/p2_valid_split
  FIT_RATIO  Default: 0.7
  SEED       Default: 20260706

Dry run:
  DRY_RUN=1 bash scripts/run_stage7_p2_valid_split.sh

Actual run on AutoDL:
  CUDA_VISIBLE_DEVICES="" bash scripts/run_stage7_p2_valid_split.sh
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
FIT_RATIO="${FIT_RATIO:-0.7}"
SEED="${SEED:-20260706}"
ROOT="${ROOT:-results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}}"
OUT_DIR="${OUT_DIR:-${ROOT}/p2_valid_split}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/stage7_p2_valid_split.log"

HEADROOM_JSON="${ROOT}/p2_headroom/headroom_summary.json"
CONSISTENCY_JSON="${ROOT}/summary/consistency_report.json"
SUMMARY_JSON="${ROOT}/summary/metrics_summary.json"
MANIFEST_JSONL="${OUT_DIR}/valid_split_manifest.jsonl"
MANIFEST_CSV="${OUT_DIR}/valid_split_manifest.csv"
SPLIT_SUMMARY_JSON="${OUT_DIR}/valid_split_summary.json"
SPLIT_SUMMARY_MD="${OUT_DIR}/valid_split_summary.md"
SPLIT_BALANCE_CSV="${OUT_DIR}/split_balance.csv"
VALID_FIT_TXT="${OUT_DIR}/valid_fit_row_indices.txt"
VALID_SELECT_TXT="${OUT_DIR}/valid_select_row_indices.txt"

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
  "$PYTHON" - "$OUT_DIR" "$CATEGORY" "$EVAL_SPLIT" "$CANDIDATE_MODE" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
category, split, candidate_mode, seed = sys.argv[2:6]
required = [
    "valid_split_manifest.jsonl",
    "valid_split_manifest.csv",
    "valid_split_summary.json",
    "valid_split_summary.md",
    "split_balance.csv",
    "valid_fit_row_indices.txt",
    "valid_select_row_indices.txt",
]
missing = [name for name in required if not (out_dir / name).is_file()]
if missing:
    raise SystemExit(f"missing outputs: {missing}")
with open(out_dir / "valid_split_summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)
if summary.get("overall_ok") is not True:
    raise SystemExit("valid_split_summary.json overall_ok is not true")
if summary.get("category") != category:
    raise SystemExit(f"category mismatch: {summary.get('category')}")
if summary.get("split") != split:
    raise SystemExit(f"split mismatch: {summary.get('split')}")
if summary.get("candidate_mode") != candidate_mode:
    raise SystemExit(f"candidate_mode mismatch: {summary.get('candidate_mode')}")
if str(summary.get("seed")) != str(seed):
    raise SystemExit(f"seed mismatch: {summary.get('seed')}")
fit = summary.get("valid_fit_count")
select = summary.get("valid_select_count")
with open(out_dir / "valid_split_manifest.jsonl", "r", encoding="utf-8") as f:
    rows = sum(1 for line in f if line.strip())
if rows != summary.get("num_samples"):
    raise SystemExit(f"manifest row count mismatch: {rows} vs {summary.get('num_samples')}")
print(f"existing_ok valid_fit={fit} valid_select={select} rows={rows}")
PY
}

echo "===== P2-1 Preflight ====="
echo "category=${CATEGORY}"
echo "split=${EVAL_SPLIT}"
echo "candidate_mode=${CANDIDATE_MODE}"
echo "root=${ROOT}"
echo "out_dir=${OUT_DIR}"
echo "fit_ratio=${FIT_RATIO}"
echo "seed=${SEED}"

ensure_valid_no_test_path "$ROOT"
ensure_valid_no_test_path "$OUT_DIR"
for path in "$HEADROOM_JSON" "$CONSISTENCY_JSON" "$SUMMARY_JSON"
do
  ensure_valid_no_test_path "$path"
  check_file "$path"
done

p1_ok="$(json_value "$CONSISTENCY_JSON" overall_ok)"
[[ "$p1_ok" == "True" || "$p1_ok" == "true" ]] || fail "P1 consistency overall_ok is not true: $p1_ok"
p2_ok="$(json_value "$HEADROOM_JSON" overall_ok)"
[[ "$p2_ok" == "True" || "$p2_ok" == "true" ]] || fail "P2-0 headroom overall_ok is not true: $p2_ok"
decision="$(json_value "$HEADROOM_JSON" recommendation.decision)"
[[ "$decision" == "continue_to_p2_1" ]] || fail "P2-0 decision is not continue_to_p2_1: $decision"
summary_split="$(json_value "$SUMMARY_JSON" split)"
summary_mode="$(json_value "$SUMMARY_JSON" candidate_mode)"
[[ "$summary_split" == "$EVAL_SPLIT" ]] || fail "Unexpected summary split: $summary_split"
[[ "$summary_mode" == "$CANDIDATE_MODE" ]] || fail "Unexpected candidate mode: $summary_mode"

if [[ "$FORCE" != "1" && -e "$SPLIT_SUMMARY_JSON" ]]; then
  echo "===== Existing P2-1 outputs ====="
  if validate_existing_outputs; then
    echo "Reuse P2-1 valid split manifest."
    echo "Summary: $SPLIT_SUMMARY_MD"
    exit 0
  fi
  fail "Existing P2-1 outputs are incomplete or invalid. Set FORCE=1 to regenerate."
fi

echo "===== P2-1 command ====="
cmd=(
  "$PYTHON"
  scripts/build_stage7_p2_valid_split.py
  --root "$ROOT"
  --out-dir "$OUT_DIR"
  --category "$CATEGORY"
  --split "$EVAL_SPLIT"
  --candidate-mode "$CANDIDATE_MODE"
  --fit-ratio "$FIT_RATIO"
  --seed "$SEED"
)
print_cmd "${cmd[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only. No P2-1 outputs were written."
  exit 0
fi

"${cmd[@]}"

echo "===== Validate P2-1 outputs ====="
validate_existing_outputs

echo
echo "Stage 7 P2-1 valid split manifest completed."
echo "Summary: $SPLIT_SUMMARY_MD"
