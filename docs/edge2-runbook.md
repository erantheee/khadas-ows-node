# Edge2 端到端验收 Runbook

本文是 **物理 Khadas Edge2 可用之后** 的完整执行手册：板端 NPU 运行、30 分钟连续运行 soak、
人工盲听 A/B/C 三条证据链。所有命令都与仓库中脚本的当前参数一致（以下命令可按需替换
`<board-ip>`、端口和路径）。

> **当前状态（诚实声明）**：本 checkout 开发机上**没有**可访问的 Edge2。
> 因此本文给出的板端命令中，只有"Mac 侧本地预演"部分被真实执行过（见文末
> §13 已验证/未验证清单）。板端 NPU 执行、30 分钟 soak、人工听音**都还没有发生**，
> 任何"PASS"都必须来自下面命令的真实输出，不能由文档或 CI 推断。

## 0. 证据边界（每一步只证明它自己）

| 步骤 | 产出 | 证明了什么 | **不能**证明什么 |
| --- | --- | --- | --- |
| checkpoint audit | `tools/verify_model_parity.py` | 权重严格加载、mask 复现 | 转换/板端/听感 |
| ONNX export parity | `tools/export_onnx.py` + CI ONNX parity 检查（见 `docs/rknn-verification.md`） | torch ↔ onnxruntime 数值一致 | RKNN 转换 |
| RKNN conversion + simulator | `rknn-convert` GitHub Actions 产物 | 能转换、模拟器上形状/数值合理 | **板端 NPU 能跑** |
| Edge2 on-board NPU | 本 runbook §7/§9 | 板端实际 `actual_mode=npu`、持续实时 | 听感质量 |
| 30-min soak | `tools/soak_edge2.py` JSON | 30 分钟连续性/时限/不降级/不崩溃 | 听感质量 |
| 人工盲听 | `tools/listening_ab.py` + `docs/listening-template.md` | **人**对音质的判断与记录 | 工程实时性 |

工程状态与听音状态相互独立：soak PASS ≠ 听感通过；听感通过 ≠ 板端实时性通过。
禁止用任何 AI/指标/SDR 代替人工听音结论。

## 1. 前置假设与检查

- 板卡：Khadas Edge2，刷写 **Khadas 官方 Ubuntu aarch64 镜像**（下文按 Ubuntu 22.04+ 假设），
  能通过局域网 ssh 登录（`khadas@<board-ip>`），与 Mac 同一网段。
- Mac：本仓库 + `.venv`（Python 3.11，numpy/torch）+ `ssh`/`scp`/`rsync`/`curl`/`gh`。
- 板端需要 **Python ≥ 3.11**（`pyproject.toml` 要求 `>=3.11,<3.13`）。
  ⚠️ **风险**：部分 Khadas Ubuntu 镜像自带 Python 3.10，`python3 -m venv` + `pip install -e .`
  会因 `requires-python` 失败。执行前先确认：`python3 --version`；不满足时先升级/安装
  Python 3.11（或在单独决策后放宽版本要求），否则 NPU worker 无法安装 torch/numpy 依赖。
- 板端可访问外网（用于 `apt-get`、`pip install -e '.[test]'`）。若不能联网，需要离线 wheel 缓存
  （本仓库未提供，属于已知风险）。
- RKNN 工具链版本：转换用 **RKNN-Toolkit2 2.5.0**（CI），板端用 **RKNN-Toolkit-Lite2 2.5.0**，
  且必须与板子内核里的 **RKNPU 驱动** 匹配。版本不匹配时 NPU worker 会启动失败并自动降级。
- 端口：节点默认 PCM TCP `9100` / HTTP `8787`，可用环境变量
  `KHADAS_AUDIO_PORT`/`KHADAS_HTTP_PORT` 覆盖（当前 `src/main.cpp` 支持），worker 端口
  `KHADAS_CPU_PORT`/`KHADAS_NPU_PORT`（默认 9201/9202）。
- 时间：`date -u` 大致同步即可（证据 JSON 记录 UTC 时间戳）；不需要 root 权限的业务命令
  （**部署脚本不假设 sudo**，只有 `tools/bootstrap_edge2.sh` 内部用 sudo 装 apt 包）。

```bash
# Mac → 板 连通性
ping -c 3 <board-ip>
ssh khadas@<board-ip> 'uname -m; uname -r; python3 --version; date -u'
# 期望：aarch64 ... ; Python 3.11.x ; 时间偏差可接受
```

## 2. Mac 侧准备与本地预演（无板即可执行）

```bash
cd /path/to/khadas-ows-node
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
ctest --test-dir build --output-on-failure
.venv/bin/pytest -q
```

本地 passthrough 预演（macOS/Linux；若 `build/khadas-ows-node` 缺失或过期，可用
`cmake -S . -B build-lead -DCMAKE_BUILD_TYPE=Release && cmake --build build-lead --parallel`，
把下面命令中的二进制换成 `build-lead/khadas-ows-node`）：

```bash
# 终端 1：启动本地节点（端口被占用时用 KHADAS_AUDIO_PORT/KHADAS_HTTP_PORT 换端口）
./build/khadas-ows-node

# 终端 2：5 秒实时流探针（PASS=exit 0）
.venv/bin/python tools/run_stream.py --mode passthrough --seconds 5 \
    --json /tmp/stream-local.json --output /tmp/stream-local.wav

# 0.2 分钟 soak 冒烟（验证报告/判定路径；PASS=exit 0）
.venv/bin/python tools/soak_edge2.py --host 127.0.0.1 --mode passthrough \
    --minutes 0.2 --json-report /tmp/soak-local.json

# 听音工具机制自检（只验证响度匹配/盲标/CSV 机器，不评价听感；PASS=exit 0）
.venv/bin/python tools/listening_ab.py --self-test
```

本地探针失败语义（必须理解，否则会误读板端结果）：

| 退出码 | 含义 |
| --- | --- |
| 0 | 所有配置的工程判据满足（PASS） |
| 1 | 判据不满足（FAIL，如 p95 超限、deadline miss、模式未启用） |
| 2 | 前置条件失败（板/服务不可达、WAV 不合法）：**没有产生任何测量证据，不是 PASS** |

> 说明：`run_stream.py` 的 deadline miss 定义为"单块往返时间 > 块时长"——严格同步协议下
> 只有这样才能维持实时；`--max-p95-ms` 缺省等于一个块时长。`--reconnect-every N` 会每 N 秒
> 用 RST 强行断开并重连（模拟网络消失），用于验证节点不因客户端异常退出而崩溃。

## 3. 获取 `.rknn` 产物（GitHub Actions，x86_64 转换 + 模拟器）

转换在 CI (`rknn-convert` workflow, `workflow_dispatch`) 的 ubuntu-22.04 x86_64 runner 上完成；
**模拟器通过不等于板端 NPU 运行**。产物（以 workflow 实际上传名为准）包含
`models/candidate-v0.fp.rknn`、`models/candidate-v0.int8.rknn`、校验报告 JSON、`SHA256SUMS`。

```bash
gh run list --repo erantheee/khadas-ows-node --workflow rknn.yml --limit 5
gh run watch <run-id> --repo erantheee/khadas-ows-node --exit-status
gh run download <run-id> --repo erantheee/khadas-ows-node -n rknn-artifacts -D artifacts
shasum -a 256 artifacts/models/candidate-v0.int8.rknn
grep -E 'cosine|max_abs|shape|toolkit' artifacts/*report*.json   # 记录模拟器 parity 数字
```

备选：板端直接下载（板能访问 GitHub 时）：
`gh run download <run-id> -n rknn-artifacts -D ~/khadas-ows-node/artifacts`。

## 4. 板端基础 bootstrap（首次部署；`deploy_edge2.sh` 会自动执行）

```bash
ssh khadas@<board-ip>
# 仓库会被 deploy_edge2.sh rsync 到 ~/khadas-ows-node；也可以手动 clone/rsync
cd ~/khadas-ows-node
sh tools/bootstrap_edge2.sh
```

`tools/bootstrap_edge2.sh` 做的事：检查 aarch64 → `sudo apt-get install build-essential cmake
python3 python3-venv python3-pip` → Release 构建 → `ctest` → `.venv` → `pip install -e '.[test]'`。
如果板端 **没有 sudo**：不要用这一步，改为手工安装 cmake/python3-venv，或使用已
provision 好的镜像并给 `deploy_edge2.sh` 加 `--skip-bootstrap`。

## 5. 安装 RKNN-Toolkit-Lite2 2.5.0（板端 runtime）

必须与 §3 转换所用 `rknn-toolkit2==2.5.0` 及板端 RKNPU 驱动匹配。

```bash
cd ~/khadas-ows-node
# 方式 A：板端 pip 源有 aarch64 wheel
.venv/bin/pip install rknn-toolkit-lite2==2.5.0
# 方式 B：从 Rockchip rknn-toolkit2 v2.5.0 release 下载 aarch64 wheel 后安装
.venv/bin/pip install ./rknn_toolkit_lite2-2.5.0-cp311-cp311-linux_aarch64.whl
# 验证 import 与 NPU 驱动
.venv/bin/python -c "from rknnlite.api import RKNNLite; print('rknnlite ok')"
dmesg | grep -i rknpu | tail -5
cat /sys/kernel/debug/rknpu/load 2>/dev/null || echo '(需要 root/该节点不存在)'
```

也可以用 `deploy_edge2.sh --lite2-wheel <本地 whl 路径>` 自动上传安装。
**失败语义**：`import rknnlite` 或 `init_runtime` 失败 → worker 直接退出非零 → 节点的
`actual_mode` 保持 `passthrough` 且 `fallback_reason=worker_unavailable`，不会伪装成功。

## 6. 部署（Mac 上执行，幂等可重复）

```bash
# 首次/完整部署（包含 bootstrap；需要 board 上 sudo 可用）
tools/deploy_edge2.sh --host khadas@<board-ip> \
    --rknn artifacts/models/candidate-v0.int8.rknn \
    --lite2-wheel /path/to/rknn_toolkit_lite2-2.5.0-cp311-cp311-linux_aarch64.whl

# 板已 provision 好之后的复跑（更快）
tools/deploy_edge2.sh --host khadas@<board-ip> \
    --rknn artifacts/models/candidate-v0.int8.rknn --skip-bootstrap

# CPU 对比模式需要板上 checkpoint（可选；不给则 cpu 模式会自动降级 passthrough）
tools/deploy_edge2.sh --host khadas@<board-ip> \
    --rknn artifacts/models/candidate-v0.int8.rknn \
    --cpu-model models/candidate-v0.pt --skip-bootstrap

# 只看命令计划，不碰板（不需要真实主机名）
tools/deploy_edge2.sh --dry-run
```

脚本行为要点：

- `rsync` 排除 `.git/.venv/build*/models 二进制/*.wav` 等，模型文件单独 `scp` 并**双侧 sha256 校验**；
- 远端 Release 构建 + `ctest`；`.venv/bin/pip install -e '.[test]'`；
- 用 `nohup` 启动 CPU/NPU worker 与节点，日志/pid 在 `~/khadas-ows-node/logs/`；
  重复执行会先按 pid 文件停旧进程再启动（幂等）；
- 自动探测 `src/main.cpp` 是否支持 `KHADAS_AUDIO_PORT/KHADAS_HTTP_PORT`；旧构建则回落到
  默认 8787 并给出警告（`--http-port` 非默认时会直接报错）；
- 最后 `curl /v1/status` 打印——**部署成功 ≠ NPU 成功**，必须看 §7 的 `actual_mode`。

常用参数：`--path`（远端目录，默认 `~/khadas-ows-node`）、`--audio-port/--http-port/
--cpu-port/--npu-port`、`--skip-bootstrap`、`--skip-tests`、`--no-start`、`--dry-run`、
`--local-repo`；ssh 选项可用环境变量 `SSH_OPTS` 覆盖。

## 7. 冒烟验证：NPU 模式是否真的启用

```bash
# 1) 状态：部署后 requested=actual=passthrough 是正常的（还没请求模式）
curl -sS http://<board-ip>:8787/v1/status

# 2) 请求 NPU 模式
curl -sS -X POST --data npu http://<board-ip>:8787/v1/mode

# 3) 关键证据：actual_mode 只有在 worker 真正处理了数据包后才会变成 npu
.venv/bin/python tools/run_stream.py --host <board-ip> --audio-port 9100 \
    --http-port 8787 --mode npu --seconds 30 --chunk-ms 100 \
    --json docs/evidence/npu-30s-stream.json \
    --output docs/listening-renders/npu.wav
curl -sS http://<board-ip>:8787/v1/status
ssh khadas@<board-ip> 'tail -n 30 ~/khadas-ows-node/logs/worker-npu.log'
```

`run_stream.py` JSON 关键字段（`pass` 由下列配置判据计算，不看音质）：

| 字段 | 含义 | 判定建议 |
| --- | --- | --- |
| `requested_mode` / `status_end.actual_mode` | 请求/实际模式 | 非 passthrough 模式下必须相等 |
| `latency_ms.p50/p95/p99` | 每块往返时延 | p95 应 ≤ `--chunk-ms`（严格同步实时判据） |
| `deadline_misses` / `miss_ratio` | RTT 超过块时长的块数/比例 | 接近 0；`--max-miss-ratio` 默认 0.02 |
| `reconnects.forced` / `recoveries` | RST 断连次数/恢复次数 | `--reconnect-every` 测试时二者应相等 |
| `rtf_estimate` | ΣRTT/音频秒（客户端估计） | <1 表示处理不慢于实时；注意含 TCP |
| `realtime_ratio` | 成功音频秒/挂钟秒 | 应 ≈1.0 |
| `fallbacks_delta` / `rejected_delta` | 服务端 fallback/拒包增量 | fallback 必须为 0 才能声称 NPU 生效 |
| `passthrough_bit_exact` | passthrough 回包是否逐位一致 | 应为 all_exact=true |

**只有** `run_stream` exit 0 且 `actual_mode=npu`、`fallbacks_delta=0`、`worker-npu.log` 正常，
才能说"板端 NPU 这一次运行成功"。30 分钟结论以 §9 的 soak 报告为准。

> ⚠️ **协议注意**：当前 C++ 节点与 worker 的协议是**严格同步**的（一次请求/一次回包）。
> 如果 NPU worker 是窗口式/阻塞式实现（例如凑满 4 秒窗口才推理），单块往返时延会超过块时长，
> 探针会**如实 FAIL**——这是"该协议下不能维持实时"的工程结论，**不要调大判据来掩饰**。
> 需要的话应另行决策异步/管线化协议（属于协议设计变更，不在本 runbook 范围的放水项）。

## 8. 准备三份盲听 render（同一输入/时长）

监听比较需要 passthrough / CPU / NPU 三份**相同输入、相同处理规则**的输出；`run_stream.py`
的 `--output` 会把收到的音频写成 WAV：

```bash
mkdir -p docs/listening-renders
# passthrough
.venv/bin/python tools/run_stream.py --host <board-ip> --mode passthrough \
    --seconds 20 --chunk-ms 100 --output docs/listening-renders/passthrough.wav \
    --json docs/evidence/render-passthrough.json
# cpu（板上 CPU worker 可用时；否则可在 Mac 上用同一 fixture 生成并明确记录）
.venv/bin/python tools/run_stream.py --host <board-ip> --mode cpu \
    --seconds 20 --chunk-ms 100 --output docs/listening-renders/cpu.wav \
    --json docs/evidence/render-cpu.json
# npu（只有 §7 确认 actual_mode=npu 后才做）
.venv/bin/python tools/run_stream.py --host <board-ip> --mode npu \
    --seconds 20 --chunk-ms 100 --output docs/listening-renders/npu.wav \
    --json docs/evidence/render-npu.json
```

注意：

- 若 NPU worker 是**窗口式**实现（例如 4 秒窗口），其输出相对输入有数秒延迟/首尾零填充，
  这是已知的吞吐探针行为，不是产品级低延迟；盲听记录中应注明。
- `--seconds 20` 是建议值；三份必须一致，输入默认是 `tools/make_fixture.py` 的确定性 fixture。
- 任何一次 render 若 `realtime_ratio` 明显 < 1 或出现 fallback，应记录并视为该 render 的缺陷，
  不要静默继续。

## 9. 30 分钟连续运行 soak（核心工程证据）

```bash
.venv/bin/python tools/soak_edge2.py \
    --host <board-ip> --audio-port 9100 --http-port 8787 \
    --mode npu --minutes 30 --chunk-ms 100 \
    --ssh khadas@<board-ip> \
    --max-fallbacks 0 --max-unrecovered-seconds 5 \
    --json-report docs/soak-report-$(date +%Y%m%d-%H%M%S).json
```

驱动行为：子进程运行 `tools/run_stream.py` 连续推 30 分钟音频；每秒轮询 `/v1/status`
（requested/actual mode、fallbacks、rejected、buffer_samples、last_rtf、model_hash）；
用探针的 `--progress-jsonl` 存活时间检测"不可恢复的流丢失"；`--ssh` 时每 60 秒读取板端
thermal zone / CPU 频率 / loadavg / throttling 事件。

判定（任何一条不满足 → **FAIL**，退出码 1；板/服务不可达 → 退出码 2，且报告写明
"NO SOAK WAS PERFORMED"）：

1. 探针能连上且在 `--startup-grace` 内产生第一个成功块；
2. 连续 `--max-unrecovered-seconds` 秒没有成功块进展 → unrecoverable stream loss（FAIL）；
3. `/v1/status` 连续 `--max-status-poll-failures` 次失败（崩溃/挂起）→ FAIL；
4. `run_stream` 自身判据（p95、deadline miss ratio、模式未启用）→ FAIL；
5. 服务端 `fallbacks` 增量 > `--max-fallbacks`（=0 时任何 fallback 都失败）→ FAIL；
6. `rejected` 增量 > 0 → FAIL；
7. 请求 cpu/npu 时 `actual_mode` 必须全程等于请求模式（`requested_mode_engaged_ratio=1.0`）；
8. `realtime_ratio ≥ --min-realtime-ratio`（默认 0.98，即挂钟不能慢于实时 2% 以上）；
9. 实际传输音频秒数 ≥ 请求的 98%；探针提前退出 → FAIL；
10. 墙钟预算（`--wall-grace-factor`，默认 1.5×）超时 → FAIL。

报告与摘要区分：

- `report["verdict"]`、`report["failures"]` 是工程结论；
- `report["listening_quality"]` 固定为 **"NOT ASSESSED..."**，听感必须另行人工盲听；
- thermal 是**辅助证据**（无 `--ssh` 时报告明确警告"throttling 未验证"，但不会因此伪造 PASS）。

证据落点：`docs/soak-report-*.json`（含 status 采样序列、thermal 采样、streamer 子报告、
失败/警告列表、模型 hash）；探针日志在 `/tmp/soak-edge2-*/streamer.log`（路径写在报告
`evidence_dir`）。失败的排查线索：`fallback_reason`、`worker-npu.log`、thermal 曲线、
`last_rtf` 高企、`status_end.actual_mode`。

30 分钟测试期间不要同时跑其他重负载实验（会污染实时性证据）。

> 若 soak FAIL 的原因是 worker 阻塞/窗口时延（见 §7 注意），把它记录为协议不兼容的工程发现；
> 只有在明确决策后才可以更换协议或降低负载模型，**不要**通过放宽 `--max-fallbacks`/
> `--max-unrecovered-seconds`/p95 判据来获得 PASS。

## 10. 人工盲听 A/B/C（结论属于人）

```bash
# 1) 先验证机器（非交互；只验证响度匹配/盲标/CSV，不评价听感）
.venv/bin/python tools/listening_ab.py --self-test

# 2) 交互式盲听（Mac 需能播放：afplay；Linux 板：aplay；都没有则 --no-play 手动播放）
.venv/bin/python tools/listening_ab.py \
    --render docs/listening-renders \
    --out docs/listening-records \
    --build-hash "git $(git rev-parse --short HEAD); rknn sha256 $(shasum -a 256 artifacts/models/candidate-v0.int8.rknn | cut -d' ' -f1)"
```

工具做什么 / 不做什么：

- **做**：用 ITU-R BS.1770 积分响度（LUFS，K 加权 + 门限）把三份输出对齐到最安静的一份
  （**只衰减、不提升**，不会削波）；用种子化随机把条件映射到 A/B/C 盲标（结束才揭示
  key）；调用系统播放器；收集 leakage/pumping/artifacts/preference 打分与自由笔记；
  写 CSV + key.json + summary.md。
- **不做**：不评价音质、不给出偏好结论、不替代听者。响度匹配只是电平对齐，不是质量指标。
- 记录位置默认 `docs/listening-records/<时间戳>/`：`A.wav B.wav C.wav`（盲标命名）、
  `scores.csv`（盲标分数）、`key.json`（结束后揭示的映射 + LUFS/增益/输入 sha256）、
  `summary.md`（人工结论 + 免责声明）。也可用 `--out`/`--a --b --c`/`--seed` 自定义。
- 响度实现的绝对刻度已与本机 `ffmpeg -filter_complex ebur128` 做过临时交叉验证：44.1k/48k
  素材上积分响度差 ≤ 0.01 LU（这不是仓库 CI 的一部分，仅说明 BS.1770 实现的刻度正确；
  电平匹配准确不代表音质判断）。
- 填写记录时同步更新 `docs/listening-template.md`（保留"工程状态 ≠ 听音状态"的分离）。

## 11. 证据地图

| 文件/命令 | 由谁产生 | 证明 |
| --- | --- | --- |
| `docs/soak-report-*.json` | `tools/soak_edge2.py` | 板端 30 分钟工程连续性与实时性（PASS/FAIL） |
| `docs/evidence/npu-30s-stream.json` | `tools/run_stream.py` | 单次 NPU 流：时延/丢块/fallback/bit-exact |
| `docs/listening-renders/*.wav` | `run_stream.py --output` | 盲听用的三份电平可比的 render |
| `docs/listening-records/<ts>/` | `tools/listening_ab.py` | **人**的盲听打分与结论（CSV/key/summary） |
| `worker-npu.log` / `server.log` | deploy 启动的进程 | 板端运行/降级原因 |
| CI `rknn-artifacts` | GitHub Actions | 转换 + 模拟器 parity（**非板端**） |

## 12. 降级与失败语义

- `actual_mode`：节点最近一次处理数据包时实际使用的模式；请求 cpu/npu 但 worker 不可用/
  超时/协议错/非有限值/ring overflow 时会退回 passthrough 并递增 `fallbacks`，
  `fallback_reason` 给出原因（`worker_unavailable`/`worker_timeout`/`worker_protocol`/
  `worker_short_read`/`worker_non_finite`/`ring_overflow`/`invalid_packet`，以实际输出为准）。
- **fallback 出现 = 不能声称 NPU 成功**；soak 默认 `--max-fallbacks 0`，任何 fallback 即 FAIL。
- 网络断开：探针会有限次自动重连（默认 10 秒窗口、最多 3 个重试块），超过即判 unrecoverable
  stream loss 并 FAIL；soak 会同时把 "自动重连发生过" 记为 warning。
- 板子重启/服务崩溃：`/v1/status` 连续失败或探针进度停滞 → soak FAIL，报告写明原因。
- 重跑：`deploy_edge2.sh --skip-bootstrap` 幂等停旧起新；只重启服务可手动：
  `ssh khadas@<board-ip> 'cd ~/khadas-ows-node && for f in logs/*.pid; do kill $(cat $f) 2>/dev/null; done'`
  然后重新执行 §6 的 deploy（或按 §5/§7 手动起 worker/节点）。

## 13. 调试命令（板端/网络/热）

```bash
# 网络健康：macOS 用
ping -c 20 -s 1400 <board-ip>
# Linux 主机可用（禁分片以检测 MTU 问题）
ping -M do -s 1400 -c 20 <board-ip>
iperf3 -c <board-ip> -t 20                 # 双方都装了 iperf3 时
# 板端网卡错误计数
ethtool eth0 | grep -iE 'speed|duplex|link'; ethtool -S eth0 | grep -iE 'err|drop|fail'
# 板端抓包（需要 root；只抓少量样本，避免污染实时性证据）
sudo tcpdump -nn -i any 'tcp port 9100' -c 50 -w /tmp/kows.pcap
sudo tcpdump -nn -i any 'tcp port 8787' -c 20
# 连接/端口状态
ss -tnp | grep -E '9100|8787|9201|9202'
# 热/频率/负载
for z in /sys/class/thermal/thermal_zone*; do echo "$z $(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
cat /proc/loadavg; dmesg | grep -iE 'throttl|thermal' | tail -20
# NPU
dmesg | grep -i rknpu | tail -20
cat /proc/device-tree/model; uname -a
# 服务日志/进程
tail -n 50 ~/khadas-ows-node/logs/server.log ~/khadas-ows-node/logs/worker-npu.log
ps -o pid,etime,pcpu,pmem,cmd -p "$(cat ~/khadas-ows-node/logs/server.pid)"
```

> `ping -M do` 是 Linux iputils 的参数，macOS 不支持；tcpdump/ethtool 在板端执行。

## 14. 已验证 / 未验证清单（随执行更新）

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| Mac 本地 passthrough `run_stream.py`（含 RST 重连） | ✅ 已本地执行 | `/tmp/stream-local*.json`（本 checkout 开发机） |
| Mac 本地 `soak_edge2.py --minutes 0.2` PASS | ✅ 已本地执行 | `/tmp/soak-local*.json` |
| Mac 本地 `listening_ab.py --self-test` PASS | ✅ 已本地执行 | 11/11 checks（响度参考/匹配/盲标/CSV） |
| BS.1770 LUFS vs ffmpeg `ebur128`（临时交叉验证） | ✅ 已本地执行 | 44.1k/48k 素材偏差 ≤ 0.01 LU |
| `deploy_edge2.sh --dry-run` 计划输出 + `bash -n` | ✅ 已本地执行 | 命令计划输出 |
| RKNN 转换 + 模拟器 parity | ⏳ 以 CI 报告为准 | `rknn-convert` run + artifacts |
| 板端 RKNPU 驱动 / Lite2 安装 | ❌ 未验证（无板） | 需 §5/§7 真实输出 |
| 板端 `actual_mode=npu` 单次流 | ❌ 未验证（无板） | 需 §7 JSON |
| 30 分钟板端 soak | ❌ 未验证（无板） | 需 §9 JSON |
| 人工盲听 A/B/C 结论 | ❌ 未验证（需人） | 需 §10 记录 + 听者签字/结论 |

**在完成 §7、§9、§10 并把真实输出写入 `docs/` 之前，仓库不得声称"Edge2 NPU 通过"或"听感通过"。**
