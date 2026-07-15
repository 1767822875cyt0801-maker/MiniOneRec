#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  DRY_RUN=1 bash scripts/run_s5_final_test_confirmation.sh --action dry-run
  CONFIRM_FINAL_TEST=1 DRY_RUN=0 bash scripts/run_s5_final_test_confirmation.sh --action run-final-test

Actions:
  dry-run         Verify frozen release hashes and print final test command plan.
  audit           Same as dry-run; intended for read-only audit logs.
  run-final-test  Execute the frozen final test exactly as configured.
  finalize-test   Finalize already generated test candidates and metrics.
  audit-partial-final-test
                  Read-only audit for the CF-complete/SASRec-missing partial state.
  resume-final-test-after-cf
                  Resume the already-started final test from SASRec generation only.

The runner does not accept lambda/bonus/projection/beam/token overrides.
EOF
}

ACTION="dry-run"
CONFIG="configs/s5_auxiliary_fusion/frozen_release_config.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --action) ACTION="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

PYTHON="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-1}"
CONFIRM_FINAL_TEST="${CONFIRM_FINAL_TEST:-0}"
CONFIRM_FINAL_TEST_RESUME="${CONFIRM_FINAL_TEST_RESUME:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PROJECT_ROOT

echo "S5 final test confirmation runner"
echo "  ACTION=$ACTION"
echo "  CONFIG=$CONFIG"
echo "  DRY_RUN=$DRY_RUN"
echo "  CONFIRM_FINAL_TEST=$CONFIRM_FINAL_TEST"
echo "  CONFIRM_FINAL_TEST_RESUME=$CONFIRM_FINAL_TEST_RESUME"
echo "  PROJECT_ROOT=<runtime-resolved>"
echo "  Frozen parameters are read only from $CONFIG"

case "$ACTION" in
  dry-run|audit|audit-partial-final-test)
    "$PYTHON" scripts/s5_final_test_confirmation.py --action "$ACTION" --config "$CONFIG"
    ;;
  resume-final-test-after-cf)
    if [[ "$DRY_RUN" == "0" && "$CONFIRM_FINAL_TEST_RESUME" != "1" ]]; then
      echo "Partial final-test resume requires CONFIRM_FINAL_TEST_RESUME=1 and DRY_RUN=0." >&2
      exit 2
    fi
    CONFIRM_FINAL_TEST_RESUME="$CONFIRM_FINAL_TEST_RESUME" DRY_RUN="$DRY_RUN" \
      "$PYTHON" scripts/s5_final_test_confirmation.py --action "$ACTION" --config "$CONFIG"
    ;;
  run-final-test|finalize-test)
    if [[ "$DRY_RUN" != "0" || "$CONFIRM_FINAL_TEST" != "1" ]]; then
      echo "Final test execution requires CONFIRM_FINAL_TEST=1 and DRY_RUN=0." >&2
      exit 2
    fi
    CONFIRM_FINAL_TEST="$CONFIRM_FINAL_TEST" DRY_RUN="$DRY_RUN" \
      "$PYTHON" scripts/s5_final_test_confirmation.py --action "$ACTION" --config "$CONFIG"
    ;;
  *)
    echo "Unknown ACTION: $ACTION" >&2
    usage
    exit 2
    ;;
esac
