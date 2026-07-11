#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run Stage 7 P3 valid-only fusion sweep with the frozen P2 history ranker.

This is CPU post-processing over existing valid exact candidates. It refuses
test paths, does not train a ranker, and writes one output directory per
lambda/source_weight configuration.

Environment:
  DRY_RUN              Default: 1. Print commands and do not write outputs.
  FORCE                Default: 0. Set 1 to allow overwriting config dirs.
  CATEGORY             Default: Industrial_and_Scientific
  EVAL_SPLIT           Fixed/default: valid
  CANDIDATE_MODE       Fixed/default: exact
  STAGE7_ROOT          Default: results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}
  TEXT_CANDIDATES      Default: ${STAGE7_ROOT}/text_mbk_k512_dedup/exact/candidates/candidates.jsonl
  BEHAVIOR_CANDIDATES  Default: ${STAGE7_ROOT}/cf_k512_dedup/exact/candidates/candidates.jsonl
  FROZEN_RANKER        Default: ${STAGE7_ROOT}/p2_frozen_ranker/frozen_ranker_manifest.json,
                       falling back to ${STAGE7_ROOT}/p2_history_ranker/model.json when the manifest is absent.
  LAMBDAS              Default: "0.4 0.5 0.6"
  SOURCE_WEIGHTS       Default: "2.0 4.0 6.0"
  OUT_DIR              Default: ${STAGE7_ROOT}/p3_fusion_sweep
  PYTHON               Default: python

Examples:
  DRY_RUN=1 bash scripts/run_stage7_p3_valid_fusion_sweep.sh
  DRY_RUN=0 bash scripts/run_stage7_p3_valid_fusion_sweep.sh
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
EVAL_SPLIT="${EVAL_SPLIT:-valid}"
CANDIDATE_MODE="${CANDIDATE_MODE:-exact}"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"
STAGE7_ROOT="${STAGE7_ROOT:-results/stage7_validation_protocol/${EVAL_SPLIT}/${CATEGORY}}"
TEXT_CANDIDATES="${TEXT_CANDIDATES:-${STAGE7_ROOT}/text_mbk_k512_dedup/exact/candidates/candidates.jsonl}"
BEHAVIOR_CANDIDATES="${BEHAVIOR_CANDIDATES:-${STAGE7_ROOT}/cf_k512_dedup/exact/candidates/candidates.jsonl}"
OUT_DIR="${OUT_DIR:-${STAGE7_ROOT}/p3_fusion_sweep}"
LAMBDAS="${LAMBDAS:-0.4 0.5 0.6}"
SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-2.0 4.0 6.0}"
K_RRF="${K_RRF:-60}"

if [[ "$EVAL_SPLIT" != "valid" ]]; then
  echo "P3 sweep is valid-only; EVAL_SPLIT must be valid, got: $EVAL_SPLIT" >&2
  exit 2
fi
if [[ "$CANDIDATE_MODE" != "exact" ]]; then
  echo "P3 sweep defaults to exact candidates only; got: $CANDIDATE_MODE" >&2
  exit 2
fi

EVAL_CSV="${EVAL_CSV:-data/Amazon/sid_versions/cf_k512_dedup/${CATEGORY}/${EVAL_SPLIT}.csv}"
if [[ ! -f "$EVAL_CSV" ]]; then
  EVAL_CSV="data/Amazon/${EVAL_SPLIT}/${CATEGORY}_5_2016-10-2018-11.csv"
fi
TRAIN_CSV="${TRAIN_CSV:-data/Amazon/sid_versions/cf_k512_dedup/${CATEGORY}/train.csv}"
if [[ ! -f "$TRAIN_CSV" ]]; then
  TRAIN_CSV="data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv"
fi
ITEM_EMB="${ITEM_EMB:-data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.cf_emb.npy}"
ROW_INDEX="${ROW_INDEX:-data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.row_index.json}"
FROZEN_RANKER="${FROZEN_RANKER:-${STAGE7_ROOT}/p2_frozen_ranker/frozen_ranker_manifest.json}"
if [[ ! -f "$FROZEN_RANKER" ]]; then
  FROZEN_RANKER="${STAGE7_ROOT}/p2_history_ranker/model.json"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

cmd=(
  "$PYTHON" scripts/run_stage7_p3_valid_fusion_sweep.py
  --category "$CATEGORY"
  --eval-split "$EVAL_SPLIT"
  --candidate-mode "$CANDIDATE_MODE"
  --text-candidates "$TEXT_CANDIDATES"
  --behavior-candidates "$BEHAVIOR_CANDIDATES"
  --eval-csv "$EVAL_CSV"
  --train-csv "$TRAIN_CSV"
  --item-emb "$ITEM_EMB"
  --row-index "$ROW_INDEX"
  --frozen-ranker "$FROZEN_RANKER"
  --out-dir "$OUT_DIR"
  --k-rrf "$K_RRF"
  --python "$PYTHON"
  --lambdas
)
for value in $LAMBDAS; do
  cmd+=("$value")
done
cmd+=(--source-weights)
for value in $SOURCE_WEIGHTS; do
  cmd+=("$value")
done
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ "$FORCE" == "1" ]]; then
  cmd+=(--force)
fi

echo "===== Stage 7 P3 Valid Fusion Sweep ====="
echo "category=${CATEGORY}"
echo "split=${EVAL_SPLIT}"
echo "candidate_mode=${CANDIDATE_MODE}"
echo "stage7_root=${STAGE7_ROOT}"
echo "out_dir=${OUT_DIR}"
echo "frozen_ranker=${FROZEN_RANKER}"
echo "dry_run=${DRY_RUN}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
