#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run P2-2: minimal source-aware learned reranker.

Scope is fixed:
  Industrial_and_Scientific / valid / exact

This is CPU-only ranking-side optimization. It trains only on valid_fit, selects
only on valid_select, never reads test, and does not run SID inference.

Environment:
  DRY_RUN       Default: 0. Set 1 to check inputs and print the Python command.
  FORCE         Default: 0. Set 1 to regenerate even if valid outputs exist.
  PYTHON        Default: python
  ROOT          Default: results/stage7_validation_protocol/valid/Industrial_and_Scientific
  OUT_DIR       Default: ${ROOT}/p2_minimal_ranker
  EPOCHS        Default: 18
  MAX_NEGATIVES Default: 80
  SEED          Default: 20260706

Dry run:
  DRY_RUN=1 bash scripts/run_stage7_p2_minimal_ranker.sh

Actual run on AutoDL:
  CUDA_VISIBLE_DEVICES="" bash scripts/run_stage7_p2_minimal_ranker.sh
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
EPOCHS="${EPOCHS:-18}"
MAX_NEGATIVES="${MAX_NEGATIVES:-80}"
SEED="${SEED:-20260706}"
ROOT="${ROOT:-results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}}"
OUT_DIR="${OUT_DIR:-${ROOT}/p2_minimal_ranker}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/stage7_p2_minimal_ranker.log"

P1_CONSISTENCY_JSON="${ROOT}/summary/consistency_report.json"
SPLIT_SUMMARY_JSON="${ROOT}/p2_valid_split/valid_split_summary.json"
SPLIT_MANIFEST_JSONL="${ROOT}/p2_valid_split/valid_split_manifest.jsonl"
FUSION_JSONL="${ROOT}/fusion/dual_fused_candidates.jsonl"
HEURISTIC_JSONL="${ROOT}/rerank/reranked_candidates.jsonl"

RANKER_REPORT_JSON="${OUT_DIR}/ranker_report.json"
RANKER_REPORT_MD="${OUT_DIR}/ranker_report.md"
RANKER_REPORT_CSV="${OUT_DIR}/ranker_report.csv"
MODEL_JSON="${OUT_DIR}/model.json"
MODEL_GRID_CSV="${OUT_DIR}/model_grid.csv"
LEARNED_JSONL="${OUT_DIR}/learned_reranked_candidates.jsonl"
PER_SAMPLE_CSV="${OUT_DIR}/per_sample_metrics.csv"

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
  "$PYTHON" - "$OUT_DIR" "$CATEGORY" "$EVAL_SPLIT" "$CANDIDATE_MODE" "$SPLIT_SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
category, split, candidate_mode, split_summary_path = sys.argv[2:6]
required = [
    "learned_reranked_candidates.jsonl",
    "per_sample_metrics.csv",
    "ranker_report.json",
    "ranker_report.csv",
    "ranker_report.md",
    "model.json",
    "model_grid.csv",
]
missing = [name for name in required if not (out_dir / name).is_file()]
if missing:
    raise SystemExit(f"missing outputs: {missing}")
with open(out_dir / "ranker_report.json", "r", encoding="utf-8") as f:
    report = json.load(f)
with open(split_summary_path, "r", encoding="utf-8") as f:
    split_summary = json.load(f)
if report.get("overall_ok") is not True:
    raise SystemExit("ranker_report.json overall_ok is not true")
if report.get("category") != category:
    raise SystemExit(f"category mismatch: {report.get('category')}")
if report.get("split") != split:
    raise SystemExit(f"split mismatch: {report.get('split')}")
if report.get("candidate_mode") != candidate_mode:
    raise SystemExit(f"candidate_mode mismatch: {report.get('candidate_mode')}")
rows = 0
with open(out_dir / "learned_reranked_candidates.jsonl", "r", encoding="utf-8") as f:
    rows = sum(1 for line in f if line.strip())
if rows != split_summary.get("num_samples"):
    raise SystemExit(f"learned rows mismatch: {rows} vs {split_summary.get('num_samples')}")
selection = report.get("selection", {}).get("freeze_recommendation", {})
decision = selection.get("decision", "not_available")
selected = report.get("selected_config_id", "not_available")
select_metrics = report.get("split_metrics", {}).get("valid_select", {})
learned = select_metrics.get("learned_rerank", {}).get("hr@20", "not_available")
heuristic = select_metrics.get("heuristic_rerank", {}).get("hr@20", "not_available")
print(f"existing_ok selected={selected} decision={decision} valid_select_learned_hr20={learned} heuristic_hr20={heuristic}")
PY
}

echo "===== P2-2 Preflight ====="
echo "category=${CATEGORY}"
echo "split=${EVAL_SPLIT}"
echo "candidate_mode=${CANDIDATE_MODE}"
echo "root=${ROOT}"
echo "out_dir=${OUT_DIR}"
echo "epochs=${EPOCHS}"
echo "max_negatives=${MAX_NEGATIVES}"
echo "seed=${SEED}"

ensure_valid_no_test_path "$ROOT"
ensure_valid_no_test_path "$OUT_DIR"
for path in \
  "$P1_CONSISTENCY_JSON" \
  "$SPLIT_SUMMARY_JSON" \
  "$SPLIT_MANIFEST_JSONL" \
  "$FUSION_JSONL" \
  "$HEURISTIC_JSONL"
do
  ensure_valid_no_test_path "$path"
  check_file "$path"
done

p1_ok="$(json_value "$P1_CONSISTENCY_JSON" overall_ok)"
[[ "$p1_ok" == "True" || "$p1_ok" == "true" ]] || fail "P1 consistency overall_ok is not true: $p1_ok"
split_ok="$(json_value "$SPLIT_SUMMARY_JSON" overall_ok)"
[[ "$split_ok" == "True" || "$split_ok" == "true" ]] || fail "P2-1 split overall_ok is not true: $split_ok"
summary_split="$(json_value "$SPLIT_SUMMARY_JSON" split)"
summary_mode="$(json_value "$SPLIT_SUMMARY_JSON" candidate_mode)"
[[ "$summary_split" == "$EVAL_SPLIT" ]] || fail "Unexpected split: $summary_split"
[[ "$summary_mode" == "$CANDIDATE_MODE" ]] || fail "Unexpected candidate mode: $summary_mode"

if [[ "$FORCE" != "1" && -e "$RANKER_REPORT_JSON" ]]; then
  echo "===== Existing P2-2 outputs ====="
  if validate_existing_outputs; then
    echo "Reuse P2-2 minimal ranker."
    echo "Summary: $RANKER_REPORT_MD"
    exit 0
  fi
  fail "Existing P2-2 outputs are incomplete or invalid. Set FORCE=1 to regenerate."
fi

echo "===== P2-2 command ====="
cmd=(
  "$PYTHON"
  scripts/train_stage7_p2_minimal_ranker.py
  --root "$ROOT"
  --out-dir "$OUT_DIR"
  --category "$CATEGORY"
  --split "$EVAL_SPLIT"
  --candidate-mode "$CANDIDATE_MODE"
  --epochs "$EPOCHS"
  --max-negatives-per-positive "$MAX_NEGATIVES"
  --seed "$SEED"
)
print_cmd "${cmd[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only. No P2-2 outputs were written."
  exit 0
fi

"${cmd[@]}"

echo "===== Validate P2-2 outputs ====="
validate_existing_outputs

echo
echo "Stage 7 P2-2 minimal ranker completed."
echo "Summary: $RANKER_REPORT_MD"
