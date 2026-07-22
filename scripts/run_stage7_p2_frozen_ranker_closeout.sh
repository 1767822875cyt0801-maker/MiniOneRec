#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run P2-4 closeout: freeze the accepted P2 history-aware ranker.

Scope is fixed:
  Industrial_and_Scientific / valid / exact

This step does not train, rerank, infer, or read test. It validates the P2 chain
and writes a compact frozen manifest for downstream P3 experiments.

Environment:
  DRY_RUN  Default: 0. Set 1 to check inputs and print the Python command.
  FORCE    Default: 0. Set 1 to regenerate even if valid outputs exist.
  PYTHON   Default: python
  ROOT     Default: results/stage7_validation_protocol/valid/Industrial_and_Scientific
  OUT_DIR  Default: ${ROOT}/p2_frozen_ranker

Dry run:
  DRY_RUN=1 bash scripts/run_stage7_p2_frozen_ranker_closeout.sh

Actual run on AutoDL:
  CUDA_VISIBLE_DEVICES="" bash scripts/run_stage7_p2_frozen_ranker_closeout.sh
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
OUT_DIR="${OUT_DIR:-${ROOT}/p2_frozen_ranker}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/stage7_p2_frozen_ranker_closeout.log"

P1_CONSISTENCY_JSON="${ROOT}/summary/consistency_report.json"
P2_HEADROOM_JSON="${ROOT}/p2_headroom/headroom_summary.json"
P2_VALID_SPLIT_JSON="${ROOT}/p2_valid_split/valid_split_summary.json"
P2_MINIMAL_REPORT_JSON="${ROOT}/p2_minimal_ranker/ranker_report.json"
P2_HISTORY_REPORT_JSON="${ROOT}/p2_history_ranker/ranker_report.json"
P2_HISTORY_MODEL_JSON="${ROOT}/p2_history_ranker/model.json"
P2_HISTORY_GRID_CSV="${ROOT}/p2_history_ranker/model_grid.csv"
P2_HISTORY_FEATURE_WEIGHTS_CSV="${ROOT}/p2_history_ranker/feature_weights.csv"

FROZEN_MANIFEST_JSON="${OUT_DIR}/frozen_ranker_manifest.json"
FROZEN_SUMMARY_MD="${OUT_DIR}/frozen_ranker_summary.md"
FROZEN_METRICS_CSV="${OUT_DIR}/frozen_ranker_metrics.csv"
FROZEN_FILES_JSON="${OUT_DIR}/frozen_ranker_files.json"

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
    "frozen_ranker_manifest.json",
    "frozen_ranker_summary.md",
    "frozen_ranker_metrics.csv",
    "frozen_ranker_files.json",
]
missing = [name for name in required if not (out_dir / name).is_file()]
if missing:
    raise SystemExit(f"missing outputs: {missing}")
with open(out_dir / "frozen_ranker_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)
if manifest.get("overall_ok") is not True:
    raise SystemExit("frozen_ranker_manifest.json overall_ok is not true")
if manifest.get("category") != category:
    raise SystemExit(f"category mismatch: {manifest.get('category')}")
if manifest.get("split") != split:
    raise SystemExit(f"split mismatch: {manifest.get('split')}")
if manifest.get("candidate_mode") != candidate_mode:
    raise SystemExit(f"candidate_mode mismatch: {manifest.get('candidate_mode')}")
decision = manifest.get("selection", {}).get("decision", "not_available")
if decision != "freeze_p2_history_ranker":
    raise SystemExit(f"unexpected freeze decision: {decision}")
ranker_id = manifest.get("frozen_ranker", {}).get("ranker_id", "not_available")
hr_gain = manifest.get("selection", {}).get("valid_select_hr20_gain", "not_available")
ndcg_gain = manifest.get("selection", {}).get("valid_select_ndcg20_gain", "not_available")
print(f"existing_ok ranker_id={ranker_id} hr20_gain={hr_gain} ndcg20_gain={ndcg_gain}")
PY
}

echo "===== P2-4 Frozen Ranker Preflight ====="
echo "category=${CATEGORY}"
echo "split=${EVAL_SPLIT}"
echo "candidate_mode=${CANDIDATE_MODE}"
echo "root=${ROOT}"
echo "out_dir=${OUT_DIR}"

ensure_valid_no_test_path "$ROOT"
ensure_valid_no_test_path "$OUT_DIR"
for path in \
  "$P1_CONSISTENCY_JSON" \
  "$P2_HEADROOM_JSON" \
  "$P2_VALID_SPLIT_JSON" \
  "$P2_MINIMAL_REPORT_JSON" \
  "$P2_HISTORY_REPORT_JSON" \
  "$P2_HISTORY_MODEL_JSON" \
  "$P2_HISTORY_GRID_CSV" \
  "$P2_HISTORY_FEATURE_WEIGHTS_CSV"
do
  ensure_valid_no_test_path "$path"
  check_file "$path"
done

p1_ok="$(json_value "$P1_CONSISTENCY_JSON" overall_ok)"
[[ "$p1_ok" == "True" || "$p1_ok" == "true" ]] || fail "P1 consistency overall_ok is not true: $p1_ok"
p20_ok="$(json_value "$P2_HEADROOM_JSON" overall_ok)"
[[ "$p20_ok" == "True" || "$p20_ok" == "true" ]] || fail "P2-0 headroom overall_ok is not true: $p20_ok"
p21_ok="$(json_value "$P2_VALID_SPLIT_JSON" overall_ok)"
[[ "$p21_ok" == "True" || "$p21_ok" == "true" ]] || fail "P2-1 split overall_ok is not true: $p21_ok"
p22_decision="$(json_value "$P2_MINIMAL_REPORT_JSON" selection.freeze_recommendation.decision)"
[[ "$p22_decision" == "do_not_freeze_p2_minimal_ranker" ]] || fail "P2-2 decision must be do_not_freeze_p2_minimal_ranker, got: $p22_decision"
p23_decision="$(json_value "$P2_HISTORY_REPORT_JSON" selection.freeze_recommendation.decision)"
[[ "$p23_decision" == "freeze_p2_history_ranker" ]] || fail "P2-3 decision must be freeze_p2_history_ranker, got: $p23_decision"

if [[ "$FORCE" != "1" && -e "$FROZEN_MANIFEST_JSON" ]]; then
  echo "===== Existing P2-4 outputs ====="
  if validate_existing_outputs; then
    echo "Reuse P2 frozen ranker manifest."
    echo "Summary: $FROZEN_SUMMARY_MD"
    exit 0
  fi
  fail "Existing P2-4 outputs are incomplete or invalid. Set FORCE=1 to regenerate."
fi

echo "===== P2-4 command ====="
cmd=(
  "$PYTHON"
  scripts/freeze_stage7_p2_history_ranker.py
  --root "$ROOT"
  --out-dir "$OUT_DIR"
  --category "$CATEGORY"
  --split "$EVAL_SPLIT"
  --candidate-mode "$CANDIDATE_MODE"
)
print_cmd "${cmd[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only. No P2-4 outputs were written."
  exit 0
fi

"${cmd[@]}"

echo "===== Validate P2-4 outputs ====="
validate_existing_outputs

echo
echo "Stage 7 P2 frozen ranker closeout completed."
echo "Summary: $FROZEN_SUMMARY_MD"
