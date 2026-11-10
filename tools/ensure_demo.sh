#!/bin/bash
# Edge2 演示栈看门狗：node/worker/DMIC evidence 任一掉线，或 tuning 不再更新时，
# 调 start_demo.sh 整栈重启；否则只负责把 actual_mode 保持在 npu。
# 由板端 crontab 每 2 分钟调用（不需要 sudo）；要临时停掉自动恢复：
#   crontab -e  # 注释掉 ensure_demo 那一行
set -u

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd "$HOME/khadas-ows-node" || exit 0
mkdir -p logs
log() { echo "$(date -u '+%F %T') $*" >> logs/watchdog.log; }

status_json=$(curl -s --max-time 2 http://127.0.0.1:8787/v1/status 2>/dev/null)

need_restart=0
[ -n "$status_json" ] || need_restart=1
ss -lnt 2>/dev/null | grep -q ':9202' || need_restart=1
if [ "$need_restart" = 0 ]; then
    evidence_age=$(printf '%s' "$status_json" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("last_evidence_age_ms",99999))' 2>/dev/null || echo 99999)
    case "$evidence_age" in ''|*[!0-9]*) evidence_age=99999 ;; esac
    [ "$evidence_age" -gt 10000 ] && need_restart=1
fi

if [ "$need_restart" = 1 ]; then
    log "detected down/stale -> restart demo stack"
    bash "$HOME/start_demo.sh"
    log "restart done"
    exit 0
fi

actual=$(printf '%s' "$status_json" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("actual_mode",""))' 2>/dev/null || echo "")
if [ "$actual" != "npu" ]; then
    log "actual_mode=${actual:-?} -> POST npu"
    curl -s --max-time 2 -X POST --data npu http://127.0.0.1:8787/v1/mode >/dev/null 2>&1 || true
fi
