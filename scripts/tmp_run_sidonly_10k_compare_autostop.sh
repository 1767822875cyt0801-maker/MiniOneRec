#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Run the temporary SID-only 10k comparison, then stop the AutoDL instance.

Important:
  Launch this script with nohup so it keeps running after your local computer disconnects:

    mkdir -p logs
    nohup bash scripts/tmp_run_sidonly_10k_compare_autostop.sh \
      > logs/sidonly_10k_compare_noearly_autostop.log 2>&1 &
    echo $! > logs/sidonly_10k_compare_noearly_autostop.pid

Environment:
  AUTO_STOP             Default: 1. Set 0 to only run and report, without shutdown.
  AUTOSTOP_CMD          Default: shutdown -h now
  STOP_DELAY_SECONDS    Default: 120
  All variables accepted by tmp_run_sidonly_10k_compare.sh are forwarded.
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

AUTO_STOP="${AUTO_STOP:-1}"
AUTOSTOP_CMD="${AUTOSTOP_CMD:-shutdown -h now}"
STOP_DELAY_SECONDS="${STOP_DELAY_SECONDS:-120}"

echo "Auto-stop SID-only compare wrapper"
echo "  started_at=$(date -Is)"
echo "  AUTO_STOP=${AUTO_STOP}"
echo "  AUTOSTOP_CMD=${AUTOSTOP_CMD}"
echo "  STOP_DELAY_SECONDS=${STOP_DELAY_SECONDS}"
echo "  pid=$$"

bash scripts/tmp_run_sidonly_10k_compare.sh
status=$?

echo
echo "SID-only compare finished"
echo "  finished_at=$(date -Is)"
echo "  exit_status=${status}"

if [[ "$AUTO_STOP" == "1" ]]; then
  echo "Auto-stop enabled. Instance will stop in ${STOP_DELAY_SECONDS}s."
  echo "Cancel manually with: kill $$"
  sleep "$STOP_DELAY_SECONDS"
  echo "Running autostop command: ${AUTOSTOP_CMD}"
  # shellcheck disable=SC2086
  $AUTOSTOP_CMD
else
  echo "AUTO_STOP=0, leaving instance running."
fi

exit "$status"
