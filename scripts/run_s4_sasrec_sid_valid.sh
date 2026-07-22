#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  BASE_MODEL=/path/to/base/model \
  STAGE=parity \
  CONFIG_MODE=smoke \
  DRY_RUN=1 \
  bash scripts/run_s4_sasrec_sid_valid.sh

Stages:
  env-preflight Validate Python/torchrun/import environment before torchrun.
  parity      Validate baseline/treatment inputs and write S4 parity manifest.
  train       Run or print isolated stream-specific SFT command.
  candidates  Run or print stream-specific valid generation + exact candidate evaluation.
  audit-existing  Classify existing stream/scope candidate artifacts without modifying them.
  summarize   Summarize treatment valid candidate report with MRR.

Defaults are valid-only, exact-only, no resume, no overwrite, and DRY_RUN=1.
For STAGE=train/candidates/summarize, set TRAIN_STREAM=baseline or TRAIN_STREAM=treatment.
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

PYTHON="${PYTHON:-python3}"
STAGE="${STAGE:-parity}"
TRAIN_STREAM="${TRAIN_STREAM:-treatment}"
DRY_RUN="${DRY_RUN:-1}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
BASE_MODEL="${BASE_MODEL:?BASE_MODEL is required and must be shared by baseline and treatment}"
SID_ROOT="${SID_ROOT:-data/Amazon/sid_versions}"
BASELINE_VERSION="${BASELINE_VERSION:-cf_k512_dedup}"
TREATMENT_VERSION="${TREATMENT_VERSION:-sasrec_v3_k512_dedup}"
CONFIG_MODE="${CONFIG_MODE:-smoke}"
SEED="${SEED:-42}"
RESULTS_ROOT="${RESULTS_ROOT:-results/s4_sasrec_sid_valid}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/s4_sasrec_sid_valid}"
NUM_GPUS="${NUM_GPUS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
CUTOFF_LEN="${CUTOFF_LEN:-512}"
BF16="${BF16:-True}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
NUM_BEAMS="${NUM_BEAMS:-50}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.0}"
MAX_PRED_SIDS="${MAX_PRED_SIDS:-50}"
MAX_CANDIDATES="${MAX_CANDIDATES:-1000}"
ALLOW_MISSING_BASE_MODEL="${ALLOW_MISSING_BASE_MODEL:-0}"
OVERWRITE="${OVERWRITE:-0}"
REUSE_COMPLETE="${REUSE_COMPLETE:-0}"
CANDIDATE_ROW_LIMIT="${CANDIDATE_ROW_LIMIT:-0}"

case "$CONFIG_MODE" in
  smoke)
    DEFAULT_SAMPLE=2000
    DEFAULT_NUM_TRAIN_EPOCHS=1
    ;;
  formal)
    DEFAULT_SAMPLE=30000
    DEFAULT_NUM_TRAIN_EPOCHS=1
    ;;
  *)
    echo "CONFIG_MODE must be smoke or formal, got: $CONFIG_MODE" >&2
    exit 2
    ;;
esac

SAMPLE="${SAMPLE:-$DEFAULT_SAMPLE}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-$DEFAULT_NUM_TRAIN_EPOCHS}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-6}"

if [[ "$CONFIG_MODE" == "smoke" && "$SAMPLE" == "-1" ]]; then
  echo "S4 smoke must use a bounded sample budget; refusing SAMPLE=-1" >&2
  exit 2
fi
if [[ "$MAX_NEW_TOKENS" -le 0 ]]; then
  echo "S4 MAX_NEW_TOKENS must be > 0; default is 6 for 4-level SID plus newline plus EOS" >&2
  exit 2
fi

COMMON_ARGS=(
  --category "$CATEGORY"
  --sid-root "$SID_ROOT"
  --baseline-version "$BASELINE_VERSION"
  --treatment-version "$TREATMENT_VERSION"
  --base-model "$BASE_MODEL"
  --results-root "$RESULTS_ROOT"
  --outputs-root "$OUTPUTS_ROOT"
  --seed "$SEED"
  --config-mode "$CONFIG_MODE"
  --num-gpus "$NUM_GPUS"
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --cutoff-len "$CUTOFF_LEN"
  --num-train-epochs "$NUM_TRAIN_EPOCHS"
  --bf16 "$BF16"
  --sample "$SAMPLE"
  --gradient-checkpointing "$GRADIENT_CHECKPOINTING"
  --num-beams "$NUM_BEAMS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --length-penalty "$LENGTH_PENALTY"
  --max-pred-sids "$MAX_PRED_SIDS"
  --max-candidates "$MAX_CANDIDATES"
)

if [[ "$ALLOW_MISSING_BASE_MODEL" == "1" ]]; then
  COMMON_ARGS+=(--allow-missing-base-model)
fi

echo "S4 SASRec Strong Behavior-SID runner"
echo "  STAGE=$STAGE"
echo "  TRAIN_STREAM=$TRAIN_STREAM"
echo "  DRY_RUN=$DRY_RUN"
echo "  CATEGORY=$CATEGORY"
echo "  CONFIG_MODE=$CONFIG_MODE"
echo "  SAMPLE=$SAMPLE"
echo "  NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS"
echo "  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "  BASE_MODEL=$BASE_MODEL"
echo "  BASELINE_VERSION=$BASELINE_VERSION"
echo "  TREATMENT_VERSION=$TREATMENT_VERSION"
echo "  CANDIDATE_ROW_LIMIT=$CANDIDATE_ROW_LIMIT"
echo "  OVERWRITE=$OVERWRITE"
echo "  REUSE_COMPLETE=$REUSE_COMPLETE"
echo "  SPLIT=valid"
echo "  TEST_READ=false"

case "$STAGE" in
  env-preflight)
    "$PYTHON" scripts/s4_sasrec_sid_valid_pipeline.py --action env-preflight "${COMMON_ARGS[@]}"
    ;;
  parity)
    ACTION="parity-audit"
    if [[ "$DRY_RUN" == "1" ]]; then
      ACTION="dry-run"
    fi
    "$PYTHON" scripts/s4_sasrec_sid_valid_pipeline.py --action "$ACTION" "${COMMON_ARGS[@]}"
    ;;
  train)
    if [[ "$TRAIN_STREAM" != "baseline" && "$TRAIN_STREAM" != "treatment" ]]; then
      echo "TRAIN_STREAM must be baseline or treatment, got: $TRAIN_STREAM" >&2
      exit 2
    fi
    PLAN_TEXT="$("$PYTHON" scripts/s4_sasrec_sid_valid_pipeline.py --action dry-run "${COMMON_ARGS[@]}")"
    echo "$PLAN_TEXT"
    CMD_KEY="${TRAIN_STREAM}_train"
    CMD_LINE="$(printf '%s\n' "$PLAN_TEXT" | sed -n "s/^  \\[$CMD_KEY\\] //p")"
    if [[ -z "$CMD_LINE" ]]; then
      echo "Could not resolve $CMD_KEY command" >&2
      exit 1
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "DRY_RUN: would execute $CMD_KEY:"
      echo "$CMD_LINE"
    else
      "$PYTHON" scripts/s4_sasrec_sid_valid_pipeline.py --action env-preflight "${COMMON_ARGS[@]}"
      eval "$CMD_LINE"
    fi
    ;;
  candidates)
    if [[ "$TRAIN_STREAM" != "baseline" && "$TRAIN_STREAM" != "treatment" ]]; then
      echo "TRAIN_STREAM must be baseline or treatment, got: $TRAIN_STREAM" >&2
      exit 2
    fi
    PLAN_FILE="$(mktemp /tmp/s4_candidate_plan.XXXXXX.json)"
    PLAN_ARGS=(--action plan-candidates --stream "$TRAIN_STREAM" --candidate-row-limit "$CANDIDATE_ROW_LIMIT")
    if [[ "$OVERWRITE" == "1" ]]; then
      PLAN_ARGS+=(--overwrite-existing)
    fi
    if [[ "$REUSE_COMPLETE" == "1" ]]; then
      PLAN_ARGS+=(--reuse-complete)
    fi
    "$PYTHON" scripts/s4_sasrec_sid_valid_pipeline.py "${PLAN_ARGS[@]}" "${COMMON_ARGS[@]}" > "$PLAN_FILE"
    GEN_KEY="${TRAIN_STREAM}_valid_generation"
    EVAL_KEY="${TRAIN_STREAM}_exact_candidate_eval"
    SUMMARY_KEY="${TRAIN_STREAM}_summary"
    GEN_CMD="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["commands"]["generation"])' "$PLAN_FILE")"
    EVAL_CMD="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["commands"]["candidate_eval"])' "$PLAN_FILE")"
    SUMMARY_CMD="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["commands"]["summary"])' "$PLAN_FILE")"
    if [[ -z "$GEN_CMD" || -z "$EVAL_CMD" || -z "$SUMMARY_CMD" ]]; then
      echo "Could not resolve candidate generation/evaluation commands" >&2
      exit 1
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "DRY_RUN: would execute $GEN_KEY:"
      echo "$GEN_CMD"
      echo "DRY_RUN: would execute $EVAL_KEY:"
      echo "$EVAL_CMD"
      echo "DRY_RUN: would execute $SUMMARY_KEY:"
      echo "$SUMMARY_CMD"
      echo "DRY_RUN: artifact audit:"
      "$PYTHON" -m json.tool "$PLAN_FILE"
    else
      "$PYTHON" scripts/s4_sasrec_sid_valid_pipeline.py --action env-preflight "${COMMON_ARGS[@]}"
      PREP_FILE="$(mktemp /tmp/s4_candidate_prepare.XXXXXX.json)"
      PREP_ARGS=(--action prepare-candidates --stream "$TRAIN_STREAM" --candidate-row-limit "$CANDIDATE_ROW_LIMIT")
      if [[ "$OVERWRITE" == "1" ]]; then
        PREP_ARGS+=(--overwrite-existing)
      fi
      if [[ "$REUSE_COMPLETE" == "1" ]]; then
        PREP_ARGS+=(--reuse-complete)
      fi
      "$PYTHON" scripts/s4_sasrec_sid_valid_pipeline.py "${PREP_ARGS[@]}" "${COMMON_ARGS[@]}" > "$PREP_FILE"
      REUSED="$("$PYTHON" -c 'import json,sys; print("1" if json.load(open(sys.argv[1])).get("reused") else "0")' "$PREP_FILE")"
      if [[ "$REUSED" == "1" ]]; then
        echo "S4 candidate artifacts already complete and explicitly reused:"
        "$PYTHON" -m json.tool "$PREP_FILE"
        exit 0
      fi
      GEN_CMD="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["commands"]["generation"])' "$PREP_FILE")"
      EVAL_CMD="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["commands"]["candidate_eval"])' "$PREP_FILE")"
      SUMMARY_CMD="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["commands"]["summary"])' "$PREP_FILE")"
      PRED_FILE="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["prediction_file"])' "$PREP_FILE")"
      EVAL_CSV="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["eval_csv"])' "$PREP_FILE")"
      if ! eval "$GEN_CMD"; then
        "$PYTHON" -c 'import json,sys,pathlib; d=json.load(open(sys.argv[1])); paths=[d["prediction_file"],d["candidates_jsonl"],d["candidate_report"]]; print({"generation_failed": True, "partial_artifacts": {p: pathlib.Path(p).exists() for p in paths}})' "$PREP_FILE" >&2
        exit 1
      fi
      "$PYTHON" scripts/s4_sasrec_sid_valid_pipeline.py --action validate-predictions --stream "$TRAIN_STREAM" --candidate-row-limit "$CANDIDATE_ROW_LIMIT" --prediction-file "$PRED_FILE" --eval-csv "$EVAL_CSV" "${COMMON_ARGS[@]}"
      eval "$EVAL_CMD"
      eval "$SUMMARY_CMD"
      "$PYTHON" scripts/s4_sasrec_sid_valid_pipeline.py --action finalize-candidates --stream "$TRAIN_STREAM" --candidate-row-limit "$CANDIDATE_ROW_LIMIT" "${COMMON_ARGS[@]}"
    fi
    ;;
  audit-existing)
    if [[ "$TRAIN_STREAM" != "baseline" && "$TRAIN_STREAM" != "treatment" ]]; then
      echo "TRAIN_STREAM must be baseline or treatment, got: $TRAIN_STREAM" >&2
      exit 2
    fi
    "$PYTHON" scripts/s4_sasrec_sid_valid_pipeline.py --action audit-existing --stream "$TRAIN_STREAM" --candidate-row-limit "$CANDIDATE_ROW_LIMIT" "${COMMON_ARGS[@]}"
    ;;
  summarize)
    "$PYTHON" scripts/s4_sasrec_sid_valid_pipeline.py --action summarize-candidates --stream "$TRAIN_STREAM" --candidate-row-limit "$CANDIDATE_ROW_LIMIT" "${COMMON_ARGS[@]}"
    ;;
  *)
    echo "Unknown STAGE: $STAGE" >&2
    usage
    exit 2
    ;;
esac
