#!/bin/bash
# Restart the Edge2 worker without matching this shell's own cmdline.
#
# Board status: envelope output on the six-stem RK model, roles policy, auto listening
# mode, DMIC evidence from the 100 ms evidence stream, environment contrast layer on.
# Deploy: scp tools/restart_board_worker.sh khadas@<board>:~/restart_board_worker.sh
set -u
cd "$HOME/khadas-ows-node"

for pid in $(ls /proc | grep -E '^[0-9]+$'); do
    comm=$(cat "/proc/$pid/comm" 2>/dev/null || true)
    case "$comm" in
        python*)
            cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
            case "$cmd" in
                *khadas_ows.worker*) kill "$pid" 2>/dev/null || true ;;
            esac
            ;;
    esac
done
sleep 1.5

(setsid env PYTHONPATH=python .venv/bin/python -m khadas_ows.worker \
    --mode npu --model artifacts/rk-fused.rknn --window-frames 345 \
    --output-gain unity --output-mode envelope --policy roles \
    --listening-mode auto --env-decision on \
    --attack-ms 300 --release-ms 2000 --gate-floor-db -100 --gate-full-db -90 \
    --evidence-file /tmp/dmic-latest.json \
    --node-status-url http://127.0.0.1:8787/v1/status \
    --eq-depth 1.3 --masking-dose 1.0 --max-positive-db 3.0 \
    >> logs/worker-npu-rk6.log 2>&1 < /dev/null &)

sleep 4
ss -lntp 2>/dev/null | grep 9202 || echo NO-LISTENER
