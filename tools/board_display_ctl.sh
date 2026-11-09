#!/bin/sh
# Board status-page controller. One PID file, no process-name patterns that can
# hit the caller's own shell (that mistake cost three failed launches).
#
#   sh tools/board_display_ctl.sh start [--vt 0]
#   sh tools/board_display_ctl.sh stop
#   sh tools/board_display_ctl.sh status
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PY="$ROOT/.venv/bin/python"
PAGE="$ROOT/tools/board_display.py"
PIDFILE="$ROOT/logs/display.pid"
LOG="$ROOT/logs/display.log"

is_running() {
    [ -f "$PIDFILE" ] || return 1
    pid=$(cat "$PIDFILE" 2>/dev/null || true)
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null
}

case "${1:-}" in
    start)
        shift
        if is_running; then
            echo "already running pid $(cat "$PIDFILE")"
            exit 0
        fi
        mkdir -p "$ROOT/logs"
        cd "$ROOT"
        setsid "$PY" "$PAGE" "$@" >"$LOG" 2>&1 </dev/null &
        echo $! > "$PIDFILE"
        sleep 2
        if is_running; then
            echo "started pid $(cat "$PIDFILE")"
        else
            echo "failed to start; log:" >&2
            tail -10 "$LOG" >&2 || true
            rm -f "$PIDFILE"
            exit 1
        fi
        ;;
    stop)
        if is_running; then
            pid=$(cat "$PIDFILE")
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
            rm -f "$PIDFILE"
            echo "stopped pid $pid"
        else
            echo "not running"
            rm -f "$PIDFILE"
        fi
        ;;
    status)
        if is_running; then
            echo "running pid $(cat "$PIDFILE")"
        else
            echo "not running"
        fi
        ;;
    *)
        echo "usage: $0 start|stop|status" >&2
        exit 2
        ;;
esac
