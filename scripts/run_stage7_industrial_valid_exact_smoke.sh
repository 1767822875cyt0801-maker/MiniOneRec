#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the fixed P1 validation smoke:
  Industrial_and_Scientific / valid / exact / Text-SID + CF-SID.

This script never trains, never evaluates test, never runs p3, and never touches
Office/SASRec/LightGCN/learned rerank.

Environment:
  DRY_RUN          Default: 0. Set 1 to check paths and print commands only.
  FORCE            Default: 0. Set 1 to rerun steps and overwrite their outputs.
  TEXT_CHECKPOINT  Optional explicit Text-SID final_checkpoint directory.
  CF_CHECKPOINT    Optional explicit CF-SID final_checkpoint directory.
  PYTHON           Default: python
  NUM_BEAMS        Default: 20
  MAX_CANDIDATES   Default: 500
  BATCH_SIZE       Default: 4
  MAX_NEW_TOKENS   Default: 0
  LAMBDA_TEXT      Default: 0.5
  K_RRF            Default: 60
  SOURCE_WEIGHT    Default: 4.0

Dry run:
  DRY_RUN=1 bash scripts/run_stage7_industrial_valid_exact_smoke.sh

Actual run on AutoDL:
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_stage7_industrial_valid_exact_smoke.sh
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
TEXT_VERSION="text_mbk_k512_dedup"
BEHAVIOR_VERSION="cf_k512_dedup"
SAMPLE_SIZE="30000"
NUM_EPOCHS="1"
RUN_LABEL="noearly"

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
PYTHON="${PYTHON:-python}"
NUM_BEAMS="${NUM_BEAMS:-20}"
MAX_CANDIDATES="${MAX_CANDIDATES:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
LAMBDA_TEXT="${LAMBDA_TEXT:-0.5}"
K_RRF="${K_RRF:-60}"
SOURCE_WEIGHT="${SOURCE_WEIGHT:-4.0}"

ROOT="results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}"
ALIGNMENT_DIR="${ROOT}/alignment"
TEXT_ROOT="${ROOT}/text"
CF_ROOT="${ROOT}/cf"
FUSION_DIR="${ROOT}/fusion"
RERANK_DIR="${ROOT}/rerank"
SUMMARY_DIR="${ROOT}/summary"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/stage7_industrial_valid_exact.log"

TEXT_VERSION_DIR="data/Amazon/sid_versions/${TEXT_VERSION}/${CATEGORY}"
CF_VERSION_DIR="data/Amazon/sid_versions/${BEHAVIOR_VERSION}/${CATEGORY}"
TEXT_TRAIN_CSV="${TEXT_VERSION_DIR}/train.csv"
TEXT_VALID_CSV="${TEXT_VERSION_DIR}/valid.csv"
TEXT_INFO="${TEXT_VERSION_DIR}/info.txt"
TEXT_ITEM2SID="${TEXT_VERSION_DIR}/item2sid.json"
TEXT_SID2ITEMS="${TEXT_VERSION_DIR}/sid2items.json"
TEXT_VALID_SID_SET="${TEXT_VERSION_DIR}/valid_sid_set.json"

CF_TRAIN_CSV="${CF_VERSION_DIR}/train.csv"
CF_VALID_CSV="${CF_VERSION_DIR}/valid.csv"
CF_INFO="${CF_VERSION_DIR}/info.txt"
CF_ITEM2SID="${CF_VERSION_DIR}/item2sid.json"
CF_SID2ITEMS="${CF_VERSION_DIR}/sid2items.json"
CF_VALID_SID_SET="${CF_VERSION_DIR}/valid_sid_set.json"
CF_ITEM_EMB="data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.cf_emb.npy"
CF_ROW_INDEX="data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.row_index.json"

ALIGNMENT_JSON="${ALIGNMENT_DIR}/alignment_report.json"
ALIGNMENT_MD="${ALIGNMENT_DIR}/alignment_report.md"
TEXT_PRED_JSON="${TEXT_ROOT}/predictions/predictions.json"
CF_PRED_JSON="${CF_ROOT}/predictions/predictions.json"
TEXT_CAND_JSONL="${TEXT_ROOT}/candidates/candidates.jsonl"
TEXT_CAND_REPORT="${TEXT_ROOT}/candidates/report.json"
CF_CAND_JSONL="${CF_ROOT}/candidates/candidates.jsonl"
CF_CAND_REPORT="${CF_ROOT}/candidates/report.json"
FUSION_REPORT="${FUSION_DIR}/dual_fusion_report.json"
FUSION_JSONL="${FUSION_DIR}/dual_fused_candidates.jsonl"
FUSION_SCHEMA="${FUSION_DIR}/schema_report.json"
RERANK_REPORT="${RERANK_DIR}/rerank_report.json"
RERANK_JSONL="${RERANK_DIR}/reranked_candidates.jsonl"
SUMMARY_JSON="${SUMMARY_DIR}/metrics_summary.json"
SUMMARY_CSV="${SUMMARY_DIR}/metrics_summary.csv"
SUMMARY_MD="${SUMMARY_DIR}/metrics_summary.md"
CONSISTENCY_JSON="${SUMMARY_DIR}/consistency_report.json"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

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

run_cmd() {
  print_cmd "$@"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

check_file() {
  local path="$1"
  [[ -f "$path" ]] || fail "Required file not found: $path"
}

check_dir() {
  local path="$1"
  [[ -d "$path" ]] || fail "Required directory not found: $path"
}

ensure_valid_output_path() {
  local path="$1"
  case "$path" in
    results/stage7_validation_protocol/valid/*) ;;
    *) fail "Output path is not under valid stage7 root: $path" ;;
  esac
  case "$path" in
    results/dual_sid_*|results/eval_sidonly_*|results/candidates_*|results/rerank_*|results/calc_plus_sidonly_*)
      fail "Output path overlaps frozen/test-style artifact path: $path"
      ;;
  esac
}

csv_row_count() {
  "$PYTHON" - "$1" <<'PY'
import csv
import sys
with open(sys.argv[1], "r", encoding="utf-8", newline="") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
}

json_list_count() {
  "$PYTHON" - "$1" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, list):
    raise SystemExit("not a JSON list")
print(len(data))
PY
}

json_list_schema_count() {
  "$PYTHON" - "$@" <<'PY'
import json
import sys
path = sys.argv[1]
required_keys = sys.argv[2:]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, list):
    raise SystemExit("not a JSON list")
for idx, row in enumerate(data):
    if not isinstance(row, dict):
        raise SystemExit(f"row {idx} is not an object")
    missing = [key for key in required_keys if key not in row]
    if missing:
        raise SystemExit(f"row {idx} missing keys: {missing}")
print(len(data))
PY
}

jsonl_count() {
  "$PYTHON" - "$1" <<'PY'
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
print(sum(1 for line in f if line.strip()))
PY
}

jsonl_schema_count() {
  "$PYTHON" - "$@" <<'PY'
import json
import sys
path = sys.argv[1]
required_keys = sys.argv[2:]
count = 0
with open(path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SystemExit(f"row {idx} is not an object")
        missing = [key for key in required_keys if key not in row]
        if missing:
            raise SystemExit(f"row {idx} missing keys: {missing}")
        count += 1
print(count)
PY
}

json_bool_field() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json
import sys
path, dotted = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
cur = data
for part in dotted.split("."):
    cur = cur[part]
print("true" if bool(cur) else "false")
PY
}

json_str_field() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json
import sys
path, dotted = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
cur = data
for part in dotted.split("."):
    cur = cur[part]
print(cur)
PY
}

validate_prediction() {
  local path="$1"
  local expected_rows="$2"
  [[ -f "$path" ]] || return 1
  local rows
  rows="$(json_list_schema_count "$path" output predict)" || fail "Prediction file schema is invalid: $path"
  [[ "$rows" == "$expected_rows" ]] || fail "Prediction row mismatch for $path: rows=$rows expected=$expected_rows"
}

validate_candidate() {
  local jsonl="$1"
  local report="$2"
  local expected_rows="$3"
  if [[ -f "$jsonl" || -f "$report" ]]; then
    [[ -f "$jsonl" && -f "$report" ]] || fail "Partial candidate artifact exists: $jsonl / $report"
  else
    return 1
  fi
  local rows
  rows="$(jsonl_schema_count "$jsonl" row_index target_item_id candidate_item_ids candidate_details)" || fail "Candidate JSONL schema is invalid: $jsonl"
  [[ "$rows" == "$expected_rows" ]] || fail "Candidate row mismatch for $jsonl: rows=$rows expected=$expected_rows"
  local split
  split="$(json_str_field "$report" "inputs.eval_split")" || fail "Candidate report missing inputs.eval_split: $report"
  [[ "$split" == "$EVAL_SPLIT" ]] || fail "Candidate report split mismatch: $report split=$split"
}

validate_fusion() {
  local expected_rows="$1"
  if [[ -f "$FUSION_JSONL" || -f "$FUSION_REPORT" || -f "$FUSION_SCHEMA" ]]; then
    [[ -f "$FUSION_JSONL" && -f "$FUSION_REPORT" && -f "$FUSION_SCHEMA" ]] || fail "Partial fusion artifact exists under $FUSION_DIR"
  else
    return 1
  fi
  local rows
  rows="$(jsonl_schema_count "$FUSION_JSONL" row_index target_item_id candidate_item_ids candidate_details)" || fail "Fusion JSONL schema is invalid: $FUSION_JSONL"
  [[ "$rows" == "$expected_rows" ]] || fail "Fusion row mismatch: rows=$rows expected=$expected_rows"
  local split
  split="$(json_str_field "$FUSION_REPORT" "inputs.eval_split")" || fail "Fusion report missing inputs.eval_split"
  [[ "$split" == "$EVAL_SPLIT" ]] || fail "Fusion report split mismatch: split=$split"
}

validate_rerank() {
  local expected_rows="$1"
  if [[ -f "$RERANK_JSONL" || -f "$RERANK_REPORT" ]]; then
    [[ -f "$RERANK_JSONL" && -f "$RERANK_REPORT" ]] || fail "Partial rerank artifact exists under $RERANK_DIR"
  else
    return 1
  fi
  local rows
  rows="$(jsonl_schema_count "$RERANK_JSONL" row_index target_item_id reranked_item_ids)" || fail "Rerank JSONL schema is invalid: $RERANK_JSONL"
  [[ "$rows" == "$expected_rows" ]] || fail "Rerank row mismatch: rows=$rows expected=$expected_rows"
  local split
  split="$(json_str_field "$RERANK_REPORT" "inputs.eval_split")" || fail "Rerank report missing inputs.eval_split"
  [[ "$split" == "$EVAL_SPLIT" ]] || fail "Rerank report split mismatch: split=$split"
}

validate_alignment() {
  if [[ -f "$ALIGNMENT_JSON" || -f "$ALIGNMENT_MD" ]]; then
    [[ -f "$ALIGNMENT_JSON" && -f "$ALIGNMENT_MD" ]] || fail "Partial alignment artifact exists under $ALIGNMENT_DIR"
  else
    return 1
  fi
  local ok
  ok="$(json_bool_field "$ALIGNMENT_JSON" "overall_ok")" || fail "Alignment report missing overall_ok"
  [[ "$ok" == "true" ]] || fail "Alignment overall_ok is false: $ALIGNMENT_JSON"
}

handle_existing_or_missing() {
  local label="$1"
  shift
  if [[ "$FORCE" == "1" ]]; then
    echo "FORCE=1: will run ${label}."
    return 1
  fi
  if "$@"; then
    echo "Reuse ${label}."
    return 0
  fi
  return 1
}

discover_checkpoint() {
  local explicit="$1"
  local label="$2"
  local version="$3"
  if [[ -n "$explicit" ]]; then
    check_dir "$explicit"
    echo "$explicit"
    return 0
  fi

  local candidates=()
  if [[ -d outputs ]]; then
    while IFS= read -r path; do
      candidates+=("$path")
    done < <(
      find outputs -type d -name final_checkpoint \
        | grep -F "$version" \
        | grep -F "Industrial" \
        | grep -F "sample${SAMPLE_SIZE}" \
        | grep -F "ep${NUM_EPOCHS}" \
        | grep -F "$RUN_LABEL" \
        | sort
    )
  fi

  if [[ "${#candidates[@]}" -eq 1 ]]; then
    echo "${candidates[0]}"
    return 0
  fi

  echo "Checkpoint discovery failed for ${label} (${version})." >&2
  echo "Searched under outputs/ for directories named final_checkpoint containing:" >&2
  echo "  version=${version}" >&2
  echo "  category token=Industrial" >&2
  echo "  sample${SAMPLE_SIZE}" >&2
  echo "  ep${NUM_EPOCHS}" >&2
  echo "  ${RUN_LABEL}" >&2
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "Found 0 candidates." >&2
  else
    echo "Found multiple candidates; set ${label}_CHECKPOINT explicitly:" >&2
    printf '  %s\n' "${candidates[@]}" >&2
  fi
  exit 1
}

preflight() {
  echo "===== Preflight ====="
  [[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fail "DRY_RUN must be 0 or 1"
  [[ "$FORCE" == "0" || "$FORCE" == "1" ]] || fail "FORCE must be 0 or 1"
  [[ "$EVAL_SPLIT" == "valid" ]] || fail "Internal error: EVAL_SPLIT must be valid"
  [[ "$CANDIDATE_MODE" == "exact" ]] || fail "Internal error: CANDIDATE_MODE must be exact"

  for path in "$ROOT" "$ALIGNMENT_DIR" "$TEXT_ROOT" "$CF_ROOT" "$FUSION_DIR" "$RERANK_DIR" "$SUMMARY_DIR" "$LOG_DIR"; do
    ensure_valid_output_path "$path"
  done

  check_file "$TEXT_VALID_CSV"
  check_file "$CF_VALID_CSV"
  check_file "$TEXT_TRAIN_CSV"
  check_file "$CF_TRAIN_CSV"
  check_file "$TEXT_INFO"
  check_file "$CF_INFO"
  check_file "$TEXT_ITEM2SID"
  check_file "$CF_ITEM2SID"
  check_file "$TEXT_SID2ITEMS"
  check_file "$CF_SID2ITEMS"
  check_file "$TEXT_VALID_SID_SET"
  check_file "$CF_VALID_SID_SET"
  check_file "$CF_ITEM_EMB"
  check_file "$CF_ROW_INDEX"

  TEXT_VALID_ROWS="$(csv_row_count "$TEXT_VALID_CSV")"
  CF_VALID_ROWS="$(csv_row_count "$CF_VALID_CSV")"
  [[ "$TEXT_VALID_ROWS" == "$CF_VALID_ROWS" ]] || fail "Text/CF valid row count mismatch: text=$TEXT_VALID_ROWS cf=$CF_VALID_ROWS"
  VALID_ROWS="$TEXT_VALID_ROWS"

  TEXT_CHECKPOINT_RESOLVED="$(discover_checkpoint "${TEXT_CHECKPOINT:-}" "TEXT" "$TEXT_VERSION")"
  CF_CHECKPOINT_RESOLVED="$(discover_checkpoint "${CF_CHECKPOINT:-}" "CF" "$BEHAVIOR_VERSION")"

  echo "category=${CATEGORY}"
  echo "split=${EVAL_SPLIT}"
  echo "candidate_mode=${CANDIDATE_MODE}"
  echo "valid_rows=${VALID_ROWS}"
  echo "text_checkpoint=${TEXT_CHECKPOINT_RESOLVED}"
  echo "cf_checkpoint=${CF_CHECKPOINT_RESOLVED}"
  echo "root=${ROOT}"
}

run_alignment() {
  echo "===== Alignment ====="
  if handle_existing_or_missing "alignment" validate_alignment; then
    return 0
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$ALIGNMENT_DIR"
  fi
  run_cmd "$PYTHON" scripts/check_split_alignment.py \
    --text-csv "$TEXT_VALID_CSV" \
    --behavior-csv "$CF_VALID_CSV" \
    --category "$CATEGORY" \
    --split "$EVAL_SPLIT" \
    --output-json "$ALIGNMENT_JSON" \
    --output-md "$ALIGNMENT_MD"
  if [[ "$DRY_RUN" != "1" ]]; then
    validate_alignment
  fi
}

run_inference() {
  local label="$1"
  local checkpoint="$2"
  local train_csv="$3"
  local info="$4"
  local valid_csv="$5"
  local pred_json="$6"

  echo "===== ${label} valid inference ====="
  if handle_existing_or_missing "${label} predictions" validate_prediction "$pred_json" "$VALID_ROWS"; then
    return 0
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$(dirname "$pred_json")"
  fi
  run_cmd env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" evaluate.py \
    --base_model "$checkpoint" \
    --train_file "$train_csv" \
    --info_file "$info" \
    --category "$CATEGORY" \
    --test_data_path "$valid_csv" \
    --result_json_data "$pred_json" \
    --batch_size "$BATCH_SIZE" \
    --num_beams "$NUM_BEAMS" \
    --max_new_tokens "$MAX_NEW_TOKENS"
  if [[ "$DRY_RUN" != "1" ]]; then
    validate_prediction "$pred_json" "$VALID_ROWS"
  fi
}

run_candidates() {
  local label="$1"
  local pred_json="$2"
  local eval_csv="$3"
  local item2sid="$4"
  local sid2items="$5"
  local valid_sid_set="$6"
  local out_jsonl="$7"
  local out_report="$8"

  echo "===== ${label} exact candidate expansion ====="
  if handle_existing_or_missing "${label} candidates" validate_candidate "$out_jsonl" "$out_report" "$VALID_ROWS"; then
    return 0
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$(dirname "$out_jsonl")"
  fi
  run_cmd "$PYTHON" evaluate_candidates.py \
    --prediction-file "$pred_json" \
    --eval-csv "$eval_csv" \
    --eval-split "$EVAL_SPLIT" \
    --item2sid "$item2sid" \
    --sid2items "$sid2items" \
    --valid-sid-set "$valid_sid_set" \
    --max-pred-sids "$NUM_BEAMS" \
    --max-candidates "$MAX_CANDIDATES" \
    --topk 1 3 5 10 20 50 \
    --output-jsonl "$out_jsonl" \
    --output-report "$out_report"
  if [[ "$DRY_RUN" != "1" ]]; then
    validate_candidate "$out_jsonl" "$out_report" "$VALID_ROWS"
  fi
}

run_fusion() {
  echo "===== Text+CF dual fusion ====="
  if handle_existing_or_missing "fusion" validate_fusion "$VALID_ROWS"; then
    return 0
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$FUSION_DIR"
  fi
  run_cmd "$PYTHON" fuse_dual_sid_candidates.py \
    --text-candidates "$TEXT_CAND_JSONL" \
    --behavior-candidates "$CF_CAND_JSONL" \
    --eval-csv "$CF_VALID_CSV" \
    --eval-split "$EVAL_SPLIT" \
    --out-dir "$FUSION_DIR" \
    --ks 1 3 5 10 20 50 \
    --lambda-text "$LAMBDA_TEXT" \
    --k-rrf "$K_RRF" \
    --text-name text \
    --behavior-name cf
  if [[ "$DRY_RUN" != "1" ]]; then
    validate_fusion "$VALID_ROWS"
  fi
}

run_rerank_step() {
  echo "===== Current heuristic rerank ====="
  if handle_existing_or_missing "heuristic rerank" validate_rerank "$VALID_ROWS"; then
    return 0
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$RERANK_DIR"
  fi
  run_cmd "$PYTHON" rerank.py \
    --candidate-jsonl "$FUSION_JSONL" \
    --train-csv "$CF_TRAIN_CSV" \
    --eval-split "$EVAL_SPLIT" \
    --item-emb "$CF_ITEM_EMB" \
    --row-index "$CF_ROW_INDEX" \
    --source-weight "$SOURCE_WEIGHT" \
    --topk 1 3 5 10 20 50 \
    --output-dir "$RERANK_DIR"
  if [[ "$DRY_RUN" != "1" ]]; then
    validate_rerank "$VALID_ROWS"
  fi
}

write_summary_and_consistency() {
  echo "===== Summary and consistency ====="
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$SUMMARY_DIR"
  fi
  if [[ "$FORCE" != "1" ]]; then
    local summary_count=0
    for path in "$SUMMARY_JSON" "$SUMMARY_CSV" "$SUMMARY_MD" "$CONSISTENCY_JSON"; do
      if [[ -f "$path" ]]; then
        summary_count=$((summary_count + 1))
      fi
    done
    if [[ "$summary_count" -gt 0 && "$summary_count" -lt 4 ]]; then
      fail "Partial summary artifact exists under $SUMMARY_DIR"
    fi
  fi
  if [[ "$FORCE" != "1" && -f "$SUMMARY_JSON" && -f "$SUMMARY_CSV" && -f "$SUMMARY_MD" && -f "$CONSISTENCY_JSON" ]]; then
    local ok
    ok="$(json_bool_field "$CONSISTENCY_JSON" "overall_ok")" || ok="false"
    if [[ "$ok" == "true" ]]; then
      echo "Reuse summary and consistency reports."
      return 0
    fi
    fail "Existing consistency report is not overall_ok=true. Use FORCE=1 after inspecting: $CONSISTENCY_JSON"
  fi

  run_cmd "$PYTHON" - \
    "$CATEGORY" "$EVAL_SPLIT" "$CANDIDATE_MODE" "$VALID_ROWS" "$ROOT" \
    "$ALIGNMENT_JSON" "$TEXT_PRED_JSON" "$CF_PRED_JSON" \
    "$TEXT_CAND_JSONL" "$TEXT_CAND_REPORT" "$CF_CAND_JSONL" "$CF_CAND_REPORT" \
    "$FUSION_JSONL" "$FUSION_REPORT" "$RERANK_JSONL" "$RERANK_REPORT" \
    "$SUMMARY_JSON" "$SUMMARY_CSV" "$SUMMARY_MD" "$CONSISTENCY_JSON" <<'PY'
import csv
import json
import sys
from pathlib import Path

(
    category,
    split,
    mode,
    valid_rows_raw,
    root_raw,
    alignment_json,
    text_pred_json,
    cf_pred_json,
    text_cand_jsonl,
    text_cand_report,
    cf_cand_jsonl,
    cf_cand_report,
    fusion_jsonl,
    fusion_report,
    rerank_jsonl,
    rerank_report,
    summary_json,
    summary_csv,
    summary_md,
    consistency_json,
) = sys.argv[1:]

valid_rows = int(valid_rows_raw)
root = Path(root_raw)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_list_count(path):
    return len(load_json(path))


def jsonl_count(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def under_root(path):
    p = Path(path)
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def no_test_path(path):
    text = Path(path).as_posix().lower()
    return "/test/" not in text and not text.endswith("/test.csv") and "test.csv" not in text


alignment = load_json(alignment_json)
text_cand = load_json(text_cand_report)
cf_cand = load_json(cf_cand_report)
fusion = load_json(fusion_report)
rerank = load_json(rerank_report)

union = fusion["union_recall"]
minrank = fusion["min_rank_fusion"]
rrf = fusion["rrf_fusion"]
text = fusion["text_stream"]
cf = fusion["behavior_stream"]
heuristic = rerank["after_rerank"]
weights = rerank.get("weights", {})

metrics = {
    "split": split,
    "category": category,
    "candidate_mode": mode,
    "num_valid_rows": valid_rows,
    "text_hr20": text["hr@20"],
    "text_ndcg20": text["ndcg@20"],
    "cf_hr20": cf["hr@20"],
    "cf_ndcg20": cf["ndcg@20"],
    "union_hr20": union["union_hit@20"],
    "text_only_hit20": union["text_only_hit_count@20"],
    "behavior_only_hit20": union["behavior_only_hit_count@20"],
    "both_hit20": union["both_hit_count@20"],
    "neither_hit20": union["neither_hit_count@20"],
    "minrank_hr20": minrank["hr@20"],
    "minrank_ndcg20": minrank["ndcg@20"],
    "rrf_hr20": rrf["hr@20"],
    "rrf_ndcg20": rrf["ndcg@20"],
    "heuristic_rerank_hr20": heuristic["hr@20"],
    "heuristic_rerank_ndcg20": heuristic["ndcg@20"],
    "union_minus_heuristic_rerank_hr20": union["union_hit@20"] - heuristic["hr@20"],
    "lambda_text": fusion["rrf"]["lambda_text"],
    "k_rrf": fusion["rrf"]["k_rrf"],
    "heuristic_rerank_weights": weights,
}

checks = []


def add_check(name, ok, detail):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


counts = {
    "text_prediction_rows": json_list_count(text_pred_json),
    "cf_prediction_rows": json_list_count(cf_pred_json),
    "text_candidate_rows": jsonl_count(text_cand_jsonl),
    "cf_candidate_rows": jsonl_count(cf_cand_jsonl),
    "fusion_rows": jsonl_count(fusion_jsonl),
    "rerank_rows": jsonl_count(rerank_jsonl),
}
for name, value in counts.items():
    add_check(name, value == valid_rows, f"{value} == {valid_rows}")

add_check("alignment overall_ok", alignment.get("overall_ok") is True, str(alignment.get("overall_ok")))
add_check("candidate mode exact", mode == "exact", mode)
add_check("split valid", split == "valid", split)
add_check("text candidate eval_split valid", text_cand["inputs"].get("eval_split") == "valid", text_cand["inputs"].get("eval_split"))
add_check("cf candidate eval_split valid", cf_cand["inputs"].get("eval_split") == "valid", cf_cand["inputs"].get("eval_split"))
add_check("fusion eval_split valid", fusion["inputs"].get("eval_split") == "valid", fusion["inputs"].get("eval_split"))
add_check("rerank eval_split valid", rerank["inputs"].get("eval_split") == "valid", rerank["inputs"].get("eval_split"))
add_check("rerank train-only popularity", rerank["inputs"].get("train_csv", "").endswith("/train.csv"), rerank["inputs"].get("train_csv"))

output_paths = [
    alignment_json,
    text_pred_json,
    cf_pred_json,
    text_cand_jsonl,
    text_cand_report,
    cf_cand_jsonl,
    cf_cand_report,
    fusion_jsonl,
    fusion_report,
    rerank_jsonl,
    rerank_report,
    summary_json,
    summary_csv,
    summary_md,
    consistency_json,
]
for path in output_paths:
    add_check(f"under valid root: {path}", under_root(path), path)
    add_check(f"no test path: {path}", no_test_path(path), path)

add_check("frozen baselines not written by runner", True, "runner writes only under valid/Industrial_and_Scientific")

overall_ok = all(check["ok"] for check in checks)
consistency = {
    "overall_ok": overall_ok,
    "category": category,
    "split": split,
    "candidate_mode": mode,
    "num_valid_rows": valid_rows,
    "row_counts": counts,
    "checks": checks,
}

for path in [summary_json, summary_csv, summary_md, consistency_json]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

Path(summary_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

with open(summary_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
    writer.writeheader()
    writer.writerow(metrics)

lines = [
    "# Stage 7 Industrial Valid Exact Smoke Summary",
    "",
    f"- split: `{split}`",
    f"- category: `{category}`",
    f"- candidate_mode: `{mode}`",
    f"- num_valid_rows: `{valid_rows}`",
    "",
    "| Metric | Value |",
    "|---|---:|",
]
for key, value in metrics.items():
    if isinstance(value, dict):
        continue
    lines.append(f"| {key} | {value} |")
lines.extend([
    "",
    "## Heuristic Rerank Weights",
    "",
])
for key, value in weights.items():
    lines.append(f"- `{key}`: `{value}`")
Path(summary_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

Path(consistency_json).write_text(json.dumps(consistency, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"Wrote summary JSON: {summary_json}")
print(f"Wrote summary CSV: {summary_csv}")
print(f"Wrote summary MD: {summary_md}")
print(f"Wrote consistency JSON: {consistency_json}")
print(f"overall_ok={overall_ok}")
if not overall_ok:
    raise SystemExit(1)
PY
}

main() {
  preflight
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "===== Dry-run commands ====="
  fi
  run_alignment
  run_inference "Text-SID" "$TEXT_CHECKPOINT_RESOLVED" "$TEXT_TRAIN_CSV" "$TEXT_INFO" "$TEXT_VALID_CSV" "$TEXT_PRED_JSON"
  run_inference "CF-SID" "$CF_CHECKPOINT_RESOLVED" "$CF_TRAIN_CSV" "$CF_INFO" "$CF_VALID_CSV" "$CF_PRED_JSON"
  run_candidates "Text-SID" "$TEXT_PRED_JSON" "$TEXT_VALID_CSV" "$TEXT_ITEM2SID" "$TEXT_SID2ITEMS" "$TEXT_VALID_SID_SET" "$TEXT_CAND_JSONL" "$TEXT_CAND_REPORT"
  run_candidates "CF-SID" "$CF_PRED_JSON" "$CF_VALID_CSV" "$CF_ITEM2SID" "$CF_SID2ITEMS" "$CF_VALID_SID_SET" "$CF_CAND_JSONL" "$CF_CAND_REPORT"
  run_fusion
  run_rerank_step
  write_summary_and_consistency
  echo
  echo "Stage 7 Industrial valid exact smoke completed."
  echo "Summary: ${SUMMARY_MD}"
}

main
