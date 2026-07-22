#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the Stage 7 P3-0 behavior representation audit.

This is CPU/read-only with respect to existing Stage 7 artifacts:
it does not train SASRec, rebuild CF embeddings, regenerate SIDs, run GPU
inference, or evaluate test. The test CSV is read only for split-provenance
overlap counts.

Environment:
  DRY_RUN       Default: 0. Set 1 to print the audit command only.
  PYTHON        Default: python
  CATEGORY      Default: Industrial_and_Scientific
  SPLIT         Default: valid
  CANDIDATE_MODE Default: exact
  CF_VERSION    Default: cf_k512_dedup
  TEXT_VERSION  Default: text_mbk_k512_dedup
  STAGE7_ROOT   Default: results/stage7_validation_protocol/${SPLIT}/${CATEGORY}
  OUT_DIR       Default: ${STAGE7_ROOT}/p3_behavior_audit

Examples:
  DRY_RUN=1 bash scripts/run_stage7_p3_behavior_audit.sh
  CUDA_VISIBLE_DEVICES="" bash scripts/run_stage7_p3_behavior_audit.sh
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
SPLIT="${SPLIT:-valid}"
CANDIDATE_MODE="${CANDIDATE_MODE:-exact}"
CF_VERSION="${CF_VERSION:-cf_k512_dedup}"
TEXT_VERSION="${TEXT_VERSION:-text_mbk_k512_dedup}"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"
STAGE7_ROOT="${STAGE7_ROOT:-results/stage7_validation_protocol/${SPLIT}/${CATEGORY}}"
OUT_DIR="${OUT_DIR:-${STAGE7_ROOT}/p3_behavior_audit}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

echo "===== Stage 7 P3-0 Behavior Audit ====="
echo "category=${CATEGORY}"
echo "split=${SPLIT}"
echo "candidate_mode=${CANDIDATE_MODE}"
echo "cf_version=${CF_VERSION}"
echo "text_version=${TEXT_VERSION}"
echo "stage7_root=${STAGE7_ROOT}"
echo "out_dir=${OUT_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"

required_files=(
  "build_cs_embeddings.py"
  "run_rqkmeans_with_emb.py"
  "rewrite_sid_csv.py"
  "sasrec.py"
  "SASRecModules_ori.py"
  "data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv"
  "data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv"
  "data/Amazon/test/${CATEGORY}_5_2016-10-2018-11.csv"
  "data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.cf_emb.npy"
  "data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.cs_embedding_report.json"
  "data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.item_order.json"
  "data/Amazon/cs_embeddings/${CATEGORY}/${CATEGORY}.row_index.json"
  "data/Amazon/sid_versions/${CF_VERSION}/${CATEGORY}/reports/generation_report.json"
  "data/Amazon/sid_versions/${CF_VERSION}/${CATEGORY}/item2sid.json"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Required audit input missing: $path" >&2
    exit 1
  fi
done

p2_manifest="${STAGE7_ROOT}/p2_frozen_ranker/frozen_ranker_manifest.json"
if [[ -f "$p2_manifest" ]]; then
  echo "P2 frozen manifest found: ${p2_manifest}"
else
  echo "WARNING: P2 frozen manifest not found locally: ${p2_manifest}" >&2
  echo "         Audit will still record the frozen ranker id from the P2 closeout contract." >&2
fi

cmd=(
  "$PYTHON" "scripts/audit_stage7_p3_behavior_representation.py"
  --repo-root "."
  --category "$CATEGORY"
  --split "$SPLIT"
  --candidate-mode "$CANDIDATE_MODE"
  --cf-version "$CF_VERSION"
  --text-version "$TEXT_VERSION"
  --stage7-root "$STAGE7_ROOT"
  --out-dir "$OUT_DIR"
)

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete."
  exit 0
fi

"${cmd[@]}"

echo
echo "P3-0 audit artifacts:"
echo "  ${OUT_DIR}/p3_behavior_audit_report.md"
echo "  ${OUT_DIR}/p3_behavior_audit_report.json"
echo "  ${OUT_DIR}/p3_behavior_audit_checks.csv"
