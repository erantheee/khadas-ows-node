#!/bin/bash
# Edge2 演示栈自启动：node + NPU worker + DMIC evidence + 切 npu。
# 由板端 crontab @reboot 调用（不需要 sudo）；也可以手工执行（幂等：先停旧的再起）。
#
#   ssh khadas@172.16.20.18 'bash ~/start_demo.sh'
#
# 配置沿用 2026-11-09 跑通的口径：T88 fp16（--window-frames 88）、--policy s3、
# evidence 100 ms -> /tmp/dmic-latest.json、音频 9101；node 超时 15000/6000。
set -u

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd "$HOME/khadas-ows-node" || exit 1
mkdir -p logs
exec >>"logs/boot.log" 2>&1

echo "=== start_demo $(date -u '+%F %T') UTC ==="

kill_pat() {
    pids=$(ps -eo pid,args | grep -F "$1" | grep -v grep | awk '{print $1}')
    [ -n "$pids" ] && kill $pids 2>/dev/null
    return 0
}
for pat in "build/khadas-ows-node" "khadas_ows.worker" "tools/dmic_evidence.py"; do
    kill_pat "$pat"
done
sleep 1

export KHADAS_MODEL_HASH=$(sha256sum artifacts/candidate-v0-t88.fp.rknn | cut -d' ' -f1)
export KHADAS_CLIENT_TIMEOUT_MS=15000
export KHADAS_WORKER_TIMEOUT_MS=6000

# 1) node（8787 状态 / 9100 音频）
setsid nohup ./build/khadas-ows-node >> logs/server.log 2>&1 < /dev/null &
echo $! > logs/server.pid
for i in $(seq 1 40); do
    curl -sf --max-time 1 http://127.0.0.1:8787/v1/status >/dev/null 2>&1 && break
    sleep 0.5
done
if ! curl -sf --max-time 2 http://127.0.0.1:8787/v1/status >/dev/null 2>&1; then
    echo "!! node 没起来，放弃"
    exit 1
fi

# 2) DMIC evidence（ALSA 设备开机可能晚到；最多试 3 轮）
for attempt in 1 2 3; do
    setsid nohup .venv/bin/python tools/dmic_evidence.py >> logs/dmic.log 2>&1 < /dev/null &
    echo $! > logs/dmic.pid
    sleep 3
    if [ -f /tmp/dmic-latest.json ] && [ $(( $(date +%s) - $(stat -c %Y /tmp/dmic-latest.json) )) -le 3 ]; then
        break
    fi
    echo "!! dmic evidence 未就绪（第 $attempt 次），重试"
    kill "$(cat logs/dmic.pid)" 2>/dev/null
    sleep 1
done

# 3) NPU worker（S3 policy；脚本内部先杀旧 worker，并等 9202）
bash "$HOME/restart_board_worker.sh"

# 4) 切 npu（worker 就绪后才有意义；失败重试）
for i in $(seq 1 20); do
    out=$(curl -s --max-time 2 -X POST --data npu http://127.0.0.1:8787/v1/mode 2>/dev/null)
    case "$out" in
        *npu*) break ;;
    esac
    sleep 0.5
done

sleep 1
echo "--- status ---"
curl -s --max-time 2 http://127.0.0.1:8787/v1/status | head -c 500
echo
echo "=== start_demo done ==="
