#!/usr/bin/env bash
# Idempotent deployment of khadas-ows-node to a Khadas Edge2 board from the Mac.
#
# What it does (in order):
#   1. ssh preflight (aarch64 check, remote home, sudo availability)   [needs ssh key]
#   2. rsync the repository (no .venv/build/models binaries/git/audio evidence)
#   3. remote bootstrap: tools/bootstrap_edge2.sh (apt/cmake/venv) unless --skip-bootstrap
#   4. remote release build + ctest
#   5. copy the .rknn artifact (and optional CPU checkpoint) with sha256 verification
#   6. install RKNN-Toolkit-Lite2 wheel if given, else print the exact instruction
#   7. (re)start worker(s) + C++ server with nohup, logs and pid files under logs/
#   8. curl GET /v1/status and print it
#
# It deliberately does NOT assume sudo: tools/bootstrap_edge2.sh uses sudo internally; if
# sudo is unavailable this script reports it and tells you to run the bootstrap manually.
# It also does NOT fake success: any failing remote step aborts with a nonzero exit code.
#
# Usage:
#   tools/deploy_edge2.sh --host khadas@192.168.0.50
#   tools/deploy_edge2.sh --host khadas@192.168.0.50 \
#       --rknn artifacts/models/candidate-v0.int8.rknn \
#       --lite2-wheel ~/Downloads/rknn_toolkit_lite2-2.5.0-cp311-cp311-linux_aarch64.whl
#   tools/deploy_edge2.sh --host khadas@192.168.0.50 --skip-bootstrap   # faster redeploy
#   tools/deploy_edge2.sh --dry-run                                    # print the plan only
#
# Exit codes: 0 success, 2 usage/validation, 3 remote bootstrap/build failed,
#             4 service start or /v1/status check failed.
set -euo pipefail

HOST_PLACEHOLDER="user@edge2-board"
HOST=""
REMOTE_PATH="~/khadas-ows-node"
RKNN=""
CPU_MODEL=""
LITE2_WHEEL=""
AUDIO_PORT="9100"
HTTP_PORT="8787"
CPU_PORT="9201"
NPU_PORT="9202"
DRY_RUN=0
SKIP_BOOTSTRAP=0
SKIP_TESTS=0
NO_START=0
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_OPTS_STR="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new}"

usage() {
  cat <<'EOF'
Usage: tools/deploy_edge2.sh --host [user@]board [options]

Required (unless --dry-run):
  --host USER@HOST          Edge2 ssh target, e.g. khadas@192.168.0.50

Options:
  --path DIR                remote installation directory (default ~/khadas-ows-node)
  --rknn FILE               .int8.rknn artifact to copy to the board (from the rknn-convert
                            GitHub Actions artifact; sha256 is verified on both sides)
  --cpu-model FILE          optional CPU checkpoint (e.g. models/candidate-v0.pt) needed for
                            cpu mode; without it the board runs passthrough/npu only
  --lite2-wheel FILE        RKNN-Toolkit-Lite2 2.5.0 aarch64 wheel to upload and install
  --audio-port PORT         node PCM TCP port (default 9100)
  --http-port PORT          node HTTP status port (default 8787)
  --cpu-port PORT           CPU worker loopback port (default 9201)
  --npu-port PORT           NPU worker loopback port (default 9202)
  --skip-bootstrap          skip tools/bootstrap_edge2.sh (board already provisioned)
  --skip-tests              skip ctest on the board
  --no-start                sync/build/install but do not (re)start services
  --local-repo DIR          local repo root (default: parent of this script)
  --dry-run                 print the full command plan; touch nothing
  -h, --help                show this help

Environment:
  SSH_OPTS                  override ssh options string
EOF
}

step() { printf '\n=== %s ===\n' "$*"; }
info() { printf '[deploy] %s\n' "$*"; }
warn() { printf '[deploy] WARNING: %s\n' "$*" >&2; }
die() { local message="$1" code="${2:-2}"; printf '[deploy] ERROR: %s\n' "$message" >&2; exit "$code"; }
dry() { printf '[dry-run] %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:?--host needs a value}"; shift 2 ;;
    --path) REMOTE_PATH="${2:?--path needs a value}"; shift 2 ;;
    --rknn) RKNN="${2:?--rknn needs a value}"; shift 2 ;;
    --cpu-model) CPU_MODEL="${2:?--cpu-model needs a value}"; shift 2 ;;
    --lite2-wheel) LITE2_WHEEL="${2:?--lite2-wheel needs a value}"; shift 2 ;;
    --audio-port) AUDIO_PORT="${2:?--audio-port needs a value}"; shift 2 ;;
    --http-port) HTTP_PORT="${2:?--http-port needs a value}"; shift 2 ;;
    --cpu-port) CPU_PORT="${2:?--cpu-port needs a value}"; shift 2 ;;
    --npu-port) NPU_PORT="${2:?--npu-port needs a value}"; shift 2 ;;
    --skip-bootstrap) SKIP_BOOTSTRAP=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --no-start) NO_START=1; shift ;;
    --local-repo) LOCAL_REPO="${2:?--local-repo needs a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

for port_name in AUDIO_PORT HTTP_PORT CPU_PORT NPU_PORT; do
  port_value="${!port_name}"
  [[ "$port_value" =~ ^[0-9]+$ ]] || die "$port_name must be numeric, got '$port_value'"
done

HOST="${HOST:-$HOST_PLACEHOLDER}"
[[ -f "$LOCAL_REPO/CMakeLists.txt" ]] || die "local repo not found at $LOCAL_REPO (no CMakeLists.txt)"
[[ -f "$LOCAL_REPO/tools/run_stream.py" ]] || warn "tools/run_stream.py missing in local repo"

if [[ -n "$RKNN" && ! -f "$RKNN" ]]; then
  if ((DRY_RUN)); then
    dry "NOTE: --rknn file does not exist locally: $RKNN (the real run would fail here)"
  else
    die "--rknn file not found: $RKNN"
  fi
fi
if [[ -n "$LITE2_WHEEL" && ! -f "$LITE2_WHEEL" ]]; then
  if ((DRY_RUN)); then
    dry "NOTE: --lite2-wheel file does not exist locally: $LITE2_WHEEL (real run would fail)"
  else
    die "--lite2-wheel file not found: $LITE2_WHEEL"
  fi
fi
if [[ -n "$CPU_MODEL" && ! -f "$CPU_MODEL" ]]; then
  if ((DRY_RUN)); then
    dry "NOTE: --cpu-model file does not exist locally: $CPU_MODEL (real run would fail)"
  else
    die "--cpu-model file not found: $CPU_MODEL"
  fi
fi

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else die "no sha256sum/shasum available locally"; fi
}

RKNN_BASENAME=""; RKNN_SHA=""
if [[ -n "$RKNN" ]]; then
  RKNN_BASENAME="$(basename "$RKNN")"
  RKNN_SHA="$(sha256_file "$RKNN")"
fi
CPU_BASENAME=""; CPU_SHA=""
if [[ -n "$CPU_MODEL" ]]; then
  CPU_BASENAME="$(basename "$CPU_MODEL")"
  CPU_SHA="$(sha256_file "$CPU_MODEL")"
fi
LITE2_BASENAME=""
if [[ -n "$LITE2_WHEEL" ]]; then LITE2_BASENAME="$(basename "$LITE2_WHEEL")"; fi

# ------------------------------------------------------------------ ssh/scp helpers

run_remote() {
  local command="$1"
  if ((DRY_RUN)); then
    dry "ssh $SSH_OPTS_STR $HOST $command"
  else
    # shellcheck disable=SC2086 # SSH_OPTS_STR is intentionally word-split
    ssh $SSH_OPTS_STR "$HOST" "$command"
  fi
}

run_remote_script() {
  local script="$1"; shift
  local quoted_args="" arg
  for arg in "$@"; do quoted_args+=" '${arg//\'/\'\\\'\'}'"; done
  if ((DRY_RUN)); then
    dry "ssh $SSH_OPTS_STR $HOST bash -s --$quoted_args   (script below)"
    printf '%s\n' "$script" | sed 's/^/[dry-run]   /'
  else
    # shellcheck disable=SC2086 # SSH_OPTS_STR is intentionally word-split
    ssh $SSH_OPTS_STR "$HOST" "bash -s --$quoted_args" <<<"$script"
  fi
}

run_rsync() {
  local -a command=(rsync -rlptz
    --exclude '.git' --exclude '.venv' --exclude 'build' --exclude 'build-*'
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache'
    --exclude 'models/*.pt' --exclude 'models/*.pth' --exclude 'models/*.onnx'
    --exclude 'models/*.onnx.data' --exclude 'models/*.rknn' --exclude 'models/calib'
    --exclude '*.wav' --exclude '*.log' --exclude '.DS_Store'
    --exclude 'docs/soak-report-*.json' --exclude 'docs/listening-records')
  if ((DRY_RUN)); then
    dry "${command[*]} $LOCAL_REPO/ $HOST:\${REMOTE_HOME}/khadas-ows-node/"
  else
    # shellcheck disable=SC2086
    "${command[@]}" -e "ssh $SSH_OPTS_STR" "$LOCAL_REPO/" "$HOST:$REMOTE_ABS/"
  fi
}

run_scp() {
  local source="$1" destination="$2"
  if ((DRY_RUN)); then
    dry "scp $SSH_OPTS_STR $source $HOST:$destination"
  else
    # shellcheck disable=SC2086
    scp $SSH_OPTS_STR "$source" "$HOST:$destination"
  fi
}

# ------------------------------------------------------------------------ preflight

step "Edge2 deployment plan"
info "board       : $HOST (remote path $REMOTE_PATH)"
info "local repo  : $LOCAL_REPO"
info "ports       : audio=$AUDIO_PORT http=$HTTP_PORT cpu-worker=$CPU_PORT npu-worker=$NPU_PORT"
info "rknn        : ${RKNN:-(not given)}"
info "cpu model   : ${CPU_MODEL:-(not given)}"
info "lite2 wheel : ${LITE2_WHEEL:-(not given)}"
info "flags       : skip_bootstrap=$SKIP_BOOTSTRAP skip_tests=$SKIP_TESTS no_start=$NO_START dry_run=$DRY_RUN"

if ((DRY_RUN)); then
  REMOTE_HOME='${REMOTE_HOME}'
  REMOTE_ABS='${REMOTE_HOME}/khadas-ows-node'
  if [[ "$REMOTE_PATH" != "~" && "$REMOTE_PATH" != "~/"* ]]; then
    REMOTE_ABS="$REMOTE_PATH"
  fi
  HAS_ENV=1
  info "dry run: no ssh/scp/rsync will be executed; plan follows"
else
  if [[ "$HOST" == "$HOST_PLACEHOLDER" ]]; then
    die "refusing to deploy to the placeholder host; pass --host user@board-ip (use --dry-run to only print the plan)"
  fi
  command -v ssh >/dev/null 2>&1 || die "ssh not available locally"
  command -v rsync >/dev/null 2>&1 || die "rsync not available locally"
  step "ssh preflight"
  # shellcheck disable=SC2086
  PREFLIGHT="$(ssh $SSH_OPTS_STR "$HOST" 'printf "HOME=%s\nARCH=%s\nSUDO=%s\n" "$HOME" "$(uname -m)" "$(command -v sudo >/dev/null 2>&1 && echo yes || echo no)"')" \
    || die "ssh preflight failed: cannot log in to $HOST (check IP/user/ssh key)" 3
  REMOTE_HOME="$(printf '%s\n' "$PREFLIGHT" | sed -n 's/^HOME=//p')"
  REMOTE_ARCH="$(printf '%s\n' "$PREFLIGHT" | sed -n 's/^ARCH=//p')"
  REMOTE_SUDO="$(printf '%s\n' "$PREFLIGHT" | sed -n 's/^SUDO=//p')"
  [[ -n "$REMOTE_HOME" ]] || die "could not determine remote home directory" 3
  [[ "$REMOTE_ARCH" == "aarch64" ]] || warn "board reports arch '$REMOTE_ARCH'; bootstrap_edge2.sh expects aarch64"
  [[ "$REMOTE_SUDO" == "yes" ]] || warn "sudo not found on the board; tools/bootstrap_edge2.sh needs it for apt-get (skip with --skip-bootstrap if the board is already provisioned)"
  if [[ "$REMOTE_PATH" == "~" ]]; then REMOTE_ABS="$REMOTE_HOME"
  elif [[ "$REMOTE_PATH" == "~/"* ]]; then REMOTE_ABS="$REMOTE_HOME/${REMOTE_PATH#\~/}"
  else REMOTE_ABS="$REMOTE_PATH"; fi
  info "remote home : $REMOTE_HOME"
  info "remote path : $REMOTE_ABS"
  info "board arch  : $REMOTE_ARCH"
fi

# ----------------------------------------------------------------------------- sync

step "sync repository (excluding .venv/build/models binaries/audio evidence)"
run_rsync   # uses REMOTE_ABS in real mode

# ------------------------------------------------------------------ provision/build

step "provision + build on the board"
PROVISION_SCRIPT="$(cat <<'REMOTE_SCRIPT'
set -euo pipefail
ABS="$1"
SKIP_BOOTSTRAP="$2"
SKIP_TESTS="$3"
cd "$ABS"
if [ "$SKIP_BOOTSTRAP" != "1" ]; then
  echo "[remote] running tools/bootstrap_edge2.sh (sudo apt/cmake/venv)"
  sh tools/bootstrap_edge2.sh
else
  echo "[remote] skipping tools/bootstrap_edge2.sh (--skip-bootstrap)"
fi
echo "[remote] release build"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
if [ "$SKIP_TESTS" != "1" ]; then
  echo "[remote] ctest"
  ctest --test-dir build --output-on-failure
fi
echo "[remote] python venv + pip install -e"
if [ ! -x .venv/bin/pip ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -e '.[test]'
mkdir -p logs models
if grep -q 'KHADAS_HTTP_PORT' src/main.cpp 2>/dev/null; then
  echo "HAS_ENV=1"
else
  echo "HAS_ENV=0"
fi
echo "PROVISION_OK=1"
REMOTE_SCRIPT
)"
if ((DRY_RUN)); then
  run_remote_script "$PROVISION_SCRIPT" "$REMOTE_ABS" "$SKIP_BOOTSTRAP" "$SKIP_TESTS"
  HAS_ENV=1
else
  PROVISION_OUT="$(run_remote_script "$PROVISION_SCRIPT" "$REMOTE_ABS" "$SKIP_BOOTSTRAP" "$SKIP_TESTS")" \
    || die "remote provision/build failed (see output above; if it was sudo/apt, run tools/bootstrap_edge2.sh manually on the board)" 3
  printf '%s\n' "$PROVISION_OUT"
  HAS_ENV="$(printf '%s\n' "$PROVISION_OUT" | sed -n 's/^HAS_ENV=//p' | tail -1)"
  [[ -n "$HAS_ENV" ]] || HAS_ENV=0
  [[ "$HAS_ENV" == "1" ]] || warn "this build does not read KHADAS_AUDIO_PORT/KHADAS_HTTP_PORT; the server will use its built-in defaults"
  if [[ "$HAS_ENV" != "1" && "$HTTP_PORT" != "8787" ]]; then
    die "build has no KHADAS_HTTP_PORT support but --http-port is $HTTP_PORT; use --http-port 8787 or rebuild after pulling the env-port change"
  fi
fi

# ------------------------------------------------------------------- artifacts copy

step "copy model artifacts"
if [[ -n "$RKNN" ]]; then
  run_remote "mkdir -p '$REMOTE_ABS/models'"
  run_scp "$RKNN" "$REMOTE_ABS/models/$RKNN_BASENAME"
  info "rknn sha256 (local): $RKNN_SHA"
else
  warn "no --rknn given: NPU mode will not be startable until the .rknn artifact is on the board"
  info "obtain it on the Mac with: gh run list --workflow=rknn.yml --repo erantheee/khadas-ows-node"
  info "then: gh run download <run-id> --repo erantheee/khadas-ows-node -n rknn-artifacts -D artifacts"
  info "and rerun: tools/deploy_edge2.sh --host $HOST --rknn artifacts/models/candidate-v0.int8.rknn --skip-bootstrap"
fi
if [[ -n "$CPU_MODEL" ]]; then
  run_scp "$CPU_MODEL" "$REMOTE_ABS/models/$CPU_BASENAME"
  info "cpu checkpoint sha256 (local): $CPU_SHA"
fi
if [[ -n "$LITE2_WHEEL" ]]; then
  run_scp "$LITE2_WHEEL" "$REMOTE_ABS/$LITE2_BASENAME"
  run_remote "cd '$REMOTE_ABS' && .venv/bin/pip install '$LITE2_BASENAME'"
else
  info "no --lite2-wheel given; NPU mode needs RKNN-Toolkit-Lite2 matching the board image."
  info "Install on the board one of:"
  info "  .venv/bin/pip install rknn-toolkit-lite2==2.5.0"
  info "  or upload the Rockchip aarch64 wheel and: .venv/bin/pip install ./rknn_toolkit_lite2-2.5.0-cp311-cp311-linux_aarch64.whl"
  info "  (must match RKNN-Toolkit2 2.5.0 used for conversion and the board's RKNPU driver)"
fi

# verify uploaded hashes (honest integrity check)
if ((!DRY_RUN)) && [[ -n "$RKNN" ]]; then
  REMOTE_SHA="$(run_remote "sha256sum '$REMOTE_ABS/models/$RKNN_BASENAME' | awk '{print \$1}'")"
  [[ "$REMOTE_SHA" == "$RKNN_SHA" ]] || die "rknn sha256 mismatch after upload: local=$RKNN_SHA remote=$REMOTE_SHA"
  info "rknn sha256 verified on board: $REMOTE_SHA"
fi

# --------------------------------------------------------------------- start services

if ((NO_START)); then
  step "services not started (--no-start)"
else
  step "start worker(s) + node (nohup, logs/ and pid files)"
  MODEL_HASH="$RKNN_SHA"
  START_SCRIPT="$(cat <<'REMOTE_SCRIPT'
set -euo pipefail
ABS="$1"; AUDIO_PORT="$2"; HTTP_PORT="$3"; CPU_PORT="$4"; NPU_PORT="$5"
HAS_ENV="$6"; MODEL_HASH="$7"; RKNN_NAME="$8"; CPU_NAME="$9"
cd "$ABS"
mkdir -p logs
for pidfile in logs/server.pid logs/worker-cpu.pid logs/worker-npu.pid; do
  if [ -f "$pidfile" ]; then
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; fi
    rm -f "$pidfile"
  fi
done
sleep 1
pkill -f "build/khadas-ows-node" 2>/dev/null || true
pkill -f "khadas_ows.worker" 2>/dev/null || true
sleep 1
if [ -n "$CPU_NAME" ] && [ -f "models/$CPU_NAME" ]; then
  echo "[remote] starting cpu worker on 127.0.0.1:$CPU_PORT"
  PYTHONPATH=python nohup .venv/bin/python -m khadas_ows.worker --mode cpu \
      --model "models/$CPU_NAME" --port "$CPU_PORT" >> logs/worker-cpu.log 2>&1 < /dev/null &
  echo $! > logs/worker-cpu.pid
elif [ -f "models/candidate-v0.pt" ]; then
  echo "[remote] starting cpu worker on 127.0.0.1:$CPU_PORT (existing models/candidate-v0.pt)"
  PYTHONPATH=python nohup .venv/bin/python -m khadas_ows.worker --mode cpu \
      --model models/candidate-v0.pt --port "$CPU_PORT" >> logs/worker-cpu.log 2>&1 < /dev/null &
  echo $! > logs/worker-cpu.pid
else
  echo "[remote] no cpu checkpoint on board: cpu mode will fall back to passthrough"
fi
if [ -n "$RKNN_NAME" ] && [ -f "models/$RKNN_NAME" ]; then
  echo "[remote] starting npu worker on 127.0.0.1:$NPU_PORT"
  PYTHONPATH=python nohup .venv/bin/python -m khadas_ows.worker --mode npu \
      --model "models/$RKNN_NAME" --port "$NPU_PORT" >> logs/worker-npu.log 2>&1 < /dev/null &
  echo $! > logs/worker-npu.pid
else
  echo "[remote] no .rknn on board: npu mode will fall back to passthrough"
fi
echo "[remote] starting node: audio=$AUDIO_PORT http=$HTTP_PORT"
if [ "$HAS_ENV" = "1" ]; then
  env KHADAS_AUDIO_PORT="$AUDIO_PORT" KHADAS_HTTP_PORT="$HTTP_PORT" \
      KHADAS_CPU_PORT="$CPU_PORT" KHADAS_NPU_PORT="$NPU_PORT" \
      KHADAS_MODEL_HASH="$MODEL_HASH" \
      nohup ./build/khadas-ows-node >> logs/server.log 2>&1 < /dev/null &
else
  echo "[remote] NOTE: build has no env port support; http stays at 8787"
  nohup ./build/khadas-ows-node "$AUDIO_PORT" >> logs/server.log 2>&1 < /dev/null &
fi
echo $! > logs/server.pid
sleep 2
if command -v curl >/dev/null 2>&1; then
  echo "REMOTE_STATUS_BEGIN"
  curl -sS --max-time 3 "http://127.0.0.1:$HTTP_PORT/v1/status" || echo "REMOTE_STATUS_FAIL"
  echo
  echo "REMOTE_STATUS_END"
else
  echo "REMOTE_STATUS_FAIL (no curl on board)"
fi
echo "[remote] server log tail:"
tail -n 5 logs/server.log 2>/dev/null || true
echo "[remote] worker-npu log tail:"
tail -n 5 logs/worker-npu.log 2>/dev/null || true
REMOTE_SCRIPT
)"
  if ((DRY_RUN)); then
    run_remote_script "$START_SCRIPT" "$REMOTE_ABS" "$AUDIO_PORT" "$HTTP_PORT" \
      "$CPU_PORT" "$NPU_PORT" "$HAS_ENV" "$MODEL_HASH" "$RKNN_BASENAME" "$CPU_BASENAME"
  else
    START_OUT="$(run_remote_script "$START_SCRIPT" "$REMOTE_ABS" "$AUDIO_PORT" "$HTTP_PORT" \
      "$CPU_PORT" "$NPU_PORT" "$HAS_ENV" "$MODEL_HASH" "$RKNN_BASENAME" "$CPU_BASENAME")" \
      || die "failed to start services on the board (see output above)" 4
    printf '%s\n' "$START_OUT" | sed '/^REMOTE_STATUS_BEGIN$/,/^REMOTE_STATUS_END$/d'
  fi
fi

# ------------------------------------------------------------------- status from Mac

step "verify /v1/status from the Mac"
HOST_ADDR="${HOST#*@}"
STATUS_URL="http://$HOST_ADDR:$HTTP_PORT/v1/status"
if ((DRY_RUN)); then
  if ((NO_START)); then
    info "--no-start: services would not be started, so no /v1/status check would run"
  else
    dry "curl -sS --max-time 5 '$STATUS_URL'"
    dry "(fallback) ssh $SSH_OPTS_STR $HOST 'curl -sS http://127.0.0.1:$HTTP_PORT/v1/status'"
    dry "curl -sS -X POST --data npu 'http://$HOST_ADDR:$HTTP_PORT/v1/mode'"
  fi
  info "dry run complete: nothing was executed"
  exit 0
fi
if ((NO_START)); then
  info "services were not started; skipping status check"
  exit 0
fi
if STATUS_JSON="$(curl -sS --max-time 5 "$STATUS_URL" 2>/dev/null)"; then
  info "status via network: $STATUS_JSON"
else
  warn "curl from the Mac to $STATUS_URL failed (firewall/interface?); trying on the board"
  STATUS_JSON="$(ssh $SSH_OPTS_STR "$HOST" "curl -sS --max-time 5 'http://127.0.0.1:$HTTP_PORT/v1/status'")" \
    || die "could not read /v1/status on the board either; check $REMOTE_ABS/logs/server.log" 4
  info "status via ssh loopback: $STATUS_JSON"
fi

step "deployment complete"
info "board       : $HOST ($REMOTE_ABS)"
info "status json : $STATUS_JSON"
info "start server: (already started; pid in $REMOTE_ABS/logs/server.pid)"
info "switch mode : curl -sS -X POST --data npu http://$HOST_ADDR:$HTTP_PORT/v1/mode"
info "stream test : .venv/bin/python tools/run_stream.py --host $HOST_ADDR --audio-port $AUDIO_PORT --http-port $HTTP_PORT --mode npu --seconds 30 --json /tmp/stream.json"
info "30-min soak : .venv/bin/python tools/soak_edge2.py --host $HOST_ADDR --audio-port $AUDIO_PORT --http-port $HTTP_PORT --mode npu --minutes 30 --ssh $HOST --json-report docs/soak-report-\$(date +%Y%m%d-%H%M%S).json"
info "logs        : $REMOTE_ABS/logs/{server,worker-npu,worker-cpu}.log"
info "stop all    : ssh $HOST 'cd $REMOTE_ABS && for f in logs/*.pid; do kill \$(cat \$f) 2>/dev/null; done'"
info "NOTE: deployment success is NOT NPU success and NOT listening acceptance."
info "      Confirm actual_mode=npu in /v1/status and run the 30-minute soak before claiming NPU."
printf '\n'
