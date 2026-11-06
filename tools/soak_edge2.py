#!/usr/bin/env python3
"""30-minute Edge2 engineering soak driver for khadas-ows-node.

This tool answers ONE engineering question: does the node keep streaming audio without
crashing, losing the stream unrecoverably, silently falling back to passthrough, or
falling behind realtime for the requested soak duration? It writes a JSON report and a
human summary table with a PASS/FAIL verdict derived only from configured engineering
criteria.

It explicitly does NOT judge audio quality. Listening acceptance is a separate human
A/B decision (see tools/listening_ab.py and docs/listening-template.md). A PASS here is
not a listening pass, and a FAIL here says nothing about perceived audio quality.

Continuity model:
  * The streaming probe (tools/run_stream.py) is launched as a child process and paced
    against a monotonic realtime clock; it retries TCP failures for a bounded time.
  * This driver polls GET /v1/status every second (requested/actual mode, fallbacks,
    rejected, buffer_samples, last_rtf, model_hash) and watches the probe's
    --progress-jsonl liveness timestamps.
  * Unrecoverable stream loss = no successful chunk progress for longer than
    --max-unrecovered-seconds, or the probe exits before the requested audio duration.
  * Optional --ssh reads board thermal zones / CPU scaling / /proc/loadavg; absent ssh
    it degrades to a warning (thermal behaviour unverified), never to a silent pass.

Exit codes:
  0  PASS (all configured engineering criteria met)
  1  FAIL (engineering criteria not met)
  2  FAIL before the soak: board/server not reachable or unusable (no soak was performed)

Usage:
    .venv/bin/python tools/soak_edge2.py --host 192.168.0.50 --mode npu --minutes 30 \
        --ssh khadas@192.168.0.50 --json-report docs/soak-report-20261106-180000.json
Local short verification (Mac, passthrough, generic server on default ports):
    .venv/bin/python tools/soak_edge2.py --host 127.0.0.1 --mode passthrough --minutes 0.2
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "tools/soak_edge2.py"
STREAMER = REPO_ROOT / "tools" / "run_stream.py"

# One ssh round trip gathers everything we want from the board. Kept POSIX-sh only so a
# stock Khadas Ubuntu image works. Output is line oriented and parsed defensively.
REMOTE_THERMAL_PAYLOAD = r"""
for zone in /sys/class/thermal/thermal_zone*; do
  [ -r "$zone/temp" ] || continue
  printf 'TEMP %s %s\n' "$(cat "$zone/type" 2>/dev/null)" "$(cat "$zone/temp" 2>/dev/null)"
done
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
  cur=$(cat "$cpu/cpufreq/scaling_cur_freq" 2>/dev/null)
  max=$(cat "$cpu/cpufreq/cpuinfo_max_freq" 2>/dev/null)
  [ -n "$cur$max" ] && printf 'FREQ %s %s %s\n' "$(basename "$cpu")" "$cur" "$max"
done
printf 'LOAD %s\n' "$(cat /proc/loadavg 2>/dev/null)"
printf 'NPU %s\n' "$(cat /sys/kernel/debug/rknpu/load 2>/dev/null || echo unavailable)"
if command -v dmesg >/dev/null 2>&1; then
  throttled=$(dmesg 2>/dev/null | grep -ciE 'throttl|thermal.*(trip|critical)' 2>/dev/null || true)
  printf 'THROTTLE_EVENTS %s\n' "${throttled:-unavailable}"
else
  printf 'THROTTLE_EVENTS unavailable\n'
fi
""".strip()


def log(message: str = "") -> None:
    print(message, flush=True)


class PreconditionError(RuntimeError):
    pass


# --------------------------------------------------------------------------- HTTP

def _http(host: str, port: int, method: str, path: str, body: str | None = None,
          timeout: float = 2.0) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        payload = body.encode() if body is not None else None
        headers = {"Content-Length": str(len(payload))} if payload is not None else {}
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def poll_status(host: str, port: int, timeout: float = 2.0) -> dict:
    code, payload = _http(host, port, "GET", "/v1/status", timeout=timeout)
    if code != 200:
        raise ConnectionError(f"GET /v1/status -> HTTP {code}")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ConnectionError(f"GET /v1/status -> non-JSON {payload[:120]!r}") from exc


def wait_for_status(host: str, port: int, timeout_s: float, interval_s: float = 0.5) -> dict:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return poll_status(host, port)
        except (OSError, ConnectionError, http.client.HTTPException) as exc:
            last_error = exc
            time.sleep(interval_s)
    raise PreconditionError(
        f"cannot reach http://{host}:{port}/v1/status within {timeout_s:.0f}s ({last_error}); "
        "the board/server is absent, powered off, or not deployed. NO SOAK WAS PERFORMED."
    )


# ------------------------------------------------------------------- progress/sample

def read_last_progress(path: Path) -> dict | None:
    """Read the newest complete JSONL progress record (probe rewrites ~1 line/s)."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 8192))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [line for line in tail.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "wall_unix" in record:
            return record
    return None


def _parse_thermal_stdout(stdout: str) -> dict:
    sample: dict = {"temps_c": {}, "cpu_freq": {}, "load": None, "npu": None,
                    "throttle_events": None, "raw_lines": []}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        sample["raw_lines"].append(line[:300])
        parts = line.split()
        tag = parts[0]
        if tag == "TEMP" and len(parts) >= 3:
            try:
                sample["temps_c"][parts[1]] = round(int(parts[2]) / 1000.0, 2)
            except (ValueError, IndexError):
                pass
        elif tag == "FREQ" and len(parts) >= 4:
            try:
                sample["cpu_freq"][parts[1]] = {
                    "cur_khz": int(parts[2]),
                    "max_khz": int(parts[3]),
                    "pct_of_max": round(100.0 * int(parts[2]) / int(parts[3]), 1)
                    if int(parts[3]) else None,
                }
            except (ValueError, IndexError):
                pass
        elif tag == "LOAD" and len(parts) >= 2:
            sample["load"] = " ".join(parts[1:])
        elif tag == "NPU" and len(parts) >= 2:
            sample["npu"] = " ".join(parts[1:])
        elif tag == "THROTTLE_EVENTS" and len(parts) >= 2:
            sample["throttle_events"] = parts[1]
    return sample


def sample_thermal(args: argparse.Namespace) -> dict:
    """Return one thermal/CPU sample; never raises (records the error instead)."""
    started = time.monotonic()
    try:
        if args.thermal_command is not None and not args.ssh:
            completed = subprocess.run(
                ["sh", "-c", args.thermal_command], capture_output=True, text=True,
                timeout=args.thermal_timeout_s,
            )
        else:
            assert args.ssh is not None
            completed = subprocess.run(
                ["ssh", *args.ssh_opts.split(), args.ssh, args.thermal_command
                 if args.thermal_command is not None else REMOTE_THERMAL_PAYLOAD],
                capture_output=True, text=True, timeout=args.thermal_timeout_s,
            )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "elapsed_ms": None}
    sample = _parse_thermal_stdout(completed.stdout)
    sample["returncode"] = completed.returncode
    sample["elapsed_ms"] = round((time.monotonic() - started) * 1000.0, 1)
    if completed.returncode != 0:
        sample["error"] = (completed.stderr or "").strip()[:500] or f"exit {completed.returncode}"
    return sample


def summarize_thermal(samples: list[dict]) -> dict:
    temps: list[float] = []
    zone_max: dict[str, float] = {}
    throttled_samples = 0
    reduced_freq_samples = 0
    errors = 0
    for sample in samples:
        if "error" in sample:
            errors += 1
        for name, value in (sample.get("temps_c") or {}).items():
            temps.append(value)
            zone_max[name] = max(zone_max.get(name, value), value)
        freqs = sample.get("cpu_freq") or {}
        if any((entry.get("pct_of_max") or 100.0) < 80.0 for entry in freqs.values()):
            reduced_freq_samples += 1
        try:
            throttled_samples += int(sample.get("throttle_events") or 0)
        except (TypeError, ValueError):
            pass
    return {
        "samples": len(samples),
        "errors": errors,
        "temperature_max_c": max(temps) if temps else None,
        "temperature_zone_max_c": zone_max,
        "throttle_event_count_seen": throttled_samples,
        "samples_with_reduced_cpu_freq": reduced_freq_samples,
        "note": "thermal data is advisory evidence from the board OS; it is not a verdict "
                "criterion by itself, but a PASS without thermal data means throttling was "
                "not observable during the soak.",
    }


# ---------------------------------------------------------------------- stream child

def start_streamer(args: argparse.Namespace, temp_dir: Path
                    ) -> tuple[subprocess.Popen, dict, object]:
    streamer_json = temp_dir / "streamer-report.json"
    progress_jsonl = temp_dir / "progress.jsonl"
    streamer_log = temp_dir / "streamer.log"
    command = [
        sys.executable, str(STREAMER),
        "--host", args.host,
        "--audio-port", str(args.audio_port),
        "--http-port", str(args.http_port),
        "--mode", args.mode,
        "--seconds", f"{args.minutes * 60.0:.3f}",
        "--chunk-ms", f"{args.chunk_ms:.3f}",
        "--json", str(streamer_json),
        "--progress-jsonl", str(progress_jsonl),
    ]
    if args.input is not None:
        command += ["--input", str(args.input)]
    if args.output is not None:
        command += ["--output", str(args.output)]
    if args.max_p95_ms is not None:
        command += ["--max-p95-ms", f"{args.max_p95_ms:.3f}"]
    if args.max_miss_ratio is not None:
        command += ["--max-miss-ratio", f"{args.max_miss_ratio:.6f}"]
    if args.require_bit_exact:
        command += ["--require-bit-exact"]
    log_paths = {"streamer_json": streamer_json, "progress_jsonl": progress_jsonl,
                 "streamer_log": streamer_log}
    handle = streamer_log.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
    return process, log_paths, handle


def stop_streamer(process: subprocess.Popen, grace_s: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)  # probe finalises its evidence on SIGINT
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def tail_log(path: Path, limit: int = 30) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-limit:]


# --------------------------------------------------------------------------- driver

def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def run_soak(args: argparse.Namespace) -> tuple[dict, int]:
    started_at = datetime.now(timezone.utc).isoformat()
    report: dict = {
        "schema": 1,
        "tool": TOOL,
        "started_at": started_at,
        "host": args.host,
        "audio_port": args.audio_port,
        "http_port": args.http_port,
        "requested_mode": args.mode,
        "duration_requested_s": args.minutes * 60.0,
        "chunk_ms": args.chunk_ms,
        "listening_quality": (
            "NOT ASSESSED. This report covers engineering stream continuity/timing only. "
            "Audio quality requires independent human blind listening "
            "(tools/listening_ab.py + docs/listening-template.md)."
        ),
        "criteria": {
            "max_fallbacks": args.max_fallbacks,
            "max_unrecovered_seconds": args.max_unrecovered_seconds,
            "min_realtime_ratio": args.min_realtime_ratio,
            "max_p95_ms": args.max_p95_ms,
            "max_miss_ratio": args.max_miss_ratio,
            "wall_grace_factor": args.wall_grace_factor,
            "require_bit_exact": bool(args.require_bit_exact),
            "thermal_ssh": args.ssh,
        },
        "failures": [],
        "warnings": [],
        "board_reachable": False,
        "status_samples": [],
        "thermal_samples": [],
        "streamer_report": None,
        "streamer_log_tail": [],
    }

    # ---------------------------------------------------------------- preflight
    try:
        status_start = wait_for_status(args.host, args.http_port, args.preflight_timeout)
    except PreconditionError as exc:
        report["failures"].append(f"server_unreachable: {exc}")
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["verdict"] = "FAIL"
        report["pass"] = False
        return report, 2
    report["board_reachable"] = True
    report["status_start"] = status_start
    log(f"[soak] board reachable: http://{args.host}:{args.http_port} "
        f"requested={status_start.get('requested_mode')} "
        f"actual={status_start.get('actual_mode')} "
        f"model_hash={status_start.get('model_hash', '') or '(unset)'}")

    # ---------------------------------------------------------------- spawn streamer
    if not STREAMER.is_file():
        report["failures"].append(f"streamer script missing: {STREAMER}")
        report["verdict"] = "FAIL"
        report["pass"] = False
        return report, 2
    requested_s = args.minutes * 60.0
    startup_grace_s = min(15.0, max(args.startup_grace_s, 0.5 * requested_s))
    wall_budget_s = requested_s * args.wall_grace_factor + startup_grace_s
    first_progress_deadline = time.monotonic() + startup_grace_s

    temp_dir = Path(tempfile.mkdtemp(prefix="soak-edge2-"))
    report["evidence_dir"] = str(temp_dir)
    process, paths, streamer_log_handle = start_streamer(args, temp_dir)
    report["streamer_log"] = str(paths["streamer_log"])
    report["streamer_json"] = str(paths["streamer_json"])
    log(f"[soak] streamer pid={process.pid} log={paths['streamer_log']}")
    start = time.monotonic()
    status_samples: list[dict] = []
    thermal_samples: list[dict] = []
    last_progress: dict | None = None
    last_progress_wall: float | None = None
    mode_mismatch_streak = 0
    consecutive_status_failures = 0
    next_status_at = start
    next_thermal_at = start
    next_print_at = start
    thermal_errors_reported = False
    fatal: list[str] = []
    streamer_exit_code: int | None = None

    try:
        while True:
            now = time.monotonic()
            elapsed = now - start
            streamer_exit_code = process.poll()

            progress = read_last_progress(paths["progress_jsonl"])
            if progress is not None:
                last_progress = progress
                wall = progress.get("wall_unix")
                if isinstance(wall, (int, float)):
                    last_progress_wall = float(wall)

            if now >= next_status_at:
                next_status_at += 1.0
                try:
                    status = poll_status(args.host, args.http_port)
                    consecutive_status_failures = 0
                    sample = {
                        "t": round(elapsed, 2),
                        "requested_mode": status.get("requested_mode"),
                        "actual_mode": status.get("actual_mode"),
                        "fallbacks": status.get("fallbacks"),
                        "rejected": status.get("rejected"),
                        "packets": status.get("packets"),
                        "buffer_samples": status.get("buffer_samples"),
                        "last_rtf": status.get("last_rtf"),
                        "model_hash": status.get("model_hash"),
                        "fallback_reason": status.get("fallback_reason"),
                    }
                    status_samples.append(sample)
                    if (args.mode != "passthrough"
                            and sample["actual_mode"] != sample["requested_mode"]
                            and (sample["packets"] or 0) > 0):
                        mode_mismatch_streak += 1
                        if mode_mismatch_streak >= 3:
                            fatal.append(
                                "mode_not_engaged: requested_mode="
                                f"{sample['requested_mode']} actual_mode={sample['actual_mode']} "
                                f"fallback_reason={sample['fallback_reason']!r} "
                                "(the node is falling back instead of running the requested mode)"
                            )
                    else:
                        mode_mismatch_streak = 0
                except (OSError, ConnectionError, http.client.HTTPException) as exc:
                    consecutive_status_failures += 1
                    if consecutive_status_failures in (1, 5):
                        report["warnings"].append(
                            f"status poll failure #{consecutive_status_failures}: {exc}"
                        )
                    if consecutive_status_failures >= args.max_status_poll_failures:
                        fatal.append(
                            f"status_endpoint_unresponsive: {consecutive_status_failures} "
                            "consecutive /v1/status failures (node crashed or hung?)"
                        )

            # unrecoverable stream loss detection
            if last_progress_wall is not None:
                silence = time.time() - last_progress_wall
                if silence > args.max_unrecovered_seconds:
                    fatal.append(
                        f"unrecoverable_stream_loss: no successful chunk for {silence:.1f}s "
                        f"> --max-unrecovered-seconds {args.max_unrecovered_seconds:.1f}s"
                    )
            elif now > first_progress_deadline:
                fatal.append(
                    "streamer_produced_no_progress: no successful chunk within "
                    f"{startup_grace_s:.1f}s of start (connection or protocol broken?)"
                )

            if streamer_exit_code is not None:
                if elapsed < requested_s - 2.0:
                    fatal.append(
                        f"streamer_exited_early: exit={streamer_exit_code} after "
                        f"{elapsed:.1f}s of {requested_s:.1f}s requested"
                    )
                break

            if elapsed > wall_budget_s:
                fatal.append(
                    f"wall_budget_exceeded: {elapsed:.1f}s elapsed > "
                    f"{wall_budget_s:.1f}s budget for {requested_s:.0f}s of audio "
                    "(processing is slower than realtime)"
                )

            # thermal sampling (optional; ssh not assumed)
            if (args.ssh or args.thermal_command) and now >= next_thermal_at:
                next_thermal_at = now + args.thermal_interval_s
                sample = sample_thermal(args)
                thermal_samples.append(sample)
                if "error" in sample and not thermal_errors_reported:
                    thermal_errors_reported = True
                    report["warnings"].append(
                        f"thermal sample unavailable: {sample['error']} "
                        "(thermal behaviour NOT verified during this soak)"
                    )

            if fatal:
                break

            if now >= next_print_at:
                next_print_at += args.progress_interval_s
                chunks = (last_progress or {}).get("chunks_ok")
                target = (last_progress or {}).get("chunks_target")
                rtt = (last_progress or {}).get("last_rtt_ms")
                silence = time.time() - last_progress_wall if last_progress_wall else None
                last_status = status_samples[-1] if status_samples else {}
                log(f"[soak] t={elapsed:7.1f}s chunks={chunks}/{target} "
                    f"last_rtt={rtt if rtt is None else round(rtt, 2)}ms "
                    f"fallbacks={last_status.get('fallbacks')} "
                    f"mode={last_status.get('actual_mode')} "
                    f"silence={silence if silence is None else round(silence, 1)}s "
                    f"thermal={len(thermal_samples)}")

            time.sleep(min(0.25, max(0.05, args.status_interval_s / 2.0)))
    except KeyboardInterrupt:
        fatal.append("soak_driver_interrupted (KeyboardInterrupt)")
    finally:
        stop_streamer(process)
        try:
            streamer_log_handle.close()
        except OSError:
            pass

    duration_actual = time.monotonic() - start

    # ---------------------------------------------------------------- merge evidence
    streamer_report: dict | None = None
    try:
        streamer_report = json.loads(paths["streamer_json"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report["warnings"].append("streamer JSON report missing/unreadable")
    report["streamer_report"] = streamer_report
    report["streamer_exit_code"] = streamer_exit_code
    report["streamer_log_tail"] = tail_log(paths["streamer_log"])
    report["duration_actual_s"] = round(duration_actual, 3)

    failures: list[str] = list(fatal)
    warnings: list[str] = list(report["warnings"])

    if streamer_report is None:
        failures.append("streamer_report_missing: cannot verify continuity/timing criteria")
    else:
        if not streamer_report.get("pass", False):
            reasons = streamer_report.get("failures") or ["unspecified"]
            failures.append(f"streamer_criteria_failed: {'; '.join(map(str, reasons))}")
        if not streamer_report.get("chunks_ok"):
            failures.append("streamer_completed_no_chunks")
        if args.mode != "passthrough":
            status_end_mode = (streamer_report.get("status_end") or {}).get("actual_mode")
            if status_end_mode != args.mode:
                failures.append(
                    f"requested_mode_not_engaged_at_end: requested={args.mode} "
                    f"actual={status_end_mode}"
                )
        audio_seconds = streamer_report.get("audio_seconds_streamed") or 0.0
        if audio_seconds < 0.98 * requested_s:
            failures.append(
                f"audio_coverage_short: streamed {audio_seconds:.1f}s < 98% of "
                f"{requested_s:.1f}s requested"
            )
        realtime_ratio = streamer_report.get("realtime_ratio")
        if realtime_ratio is not None and realtime_ratio < args.min_realtime_ratio:
            failures.append(
                f"realtime_ratio {realtime_ratio:.3f} < --min-realtime-ratio "
                f"{args.min_realtime_ratio:.3f} (wall clock slower than realtime)"
            )
        fallbacks_delta = streamer_report.get("fallbacks_delta")
        if fallbacks_delta is None and len(status_samples) >= 2:
            first, last = status_samples[0], status_samples[-1]
            if isinstance(first.get("fallbacks"), int) and isinstance(last.get("fallbacks"), int):
                fallbacks_delta = last["fallbacks"] - first["fallbacks"]
        if fallbacks_delta is not None and fallbacks_delta > args.max_fallbacks:
            failures.append(
                f"fallbacks {fallbacks_delta} > --max-fallbacks {args.max_fallbacks} "
                "(packets were served by a fallback path instead of the requested mode)"
            )
        rejected_delta = streamer_report.get("rejected_delta")
        if rejected_delta is None and len(status_samples) >= 2:
            first, last = status_samples[0], status_samples[-1]
            if isinstance(first.get("rejected"), int) and isinstance(last.get("rejected"), int):
                rejected_delta = last["rejected"] - first["rejected"]
        if rejected_delta is not None and rejected_delta > 0:
            failures.append(
                f"rejected_packets {rejected_delta}: the node rejected packets during the soak"
            )
        if (streamer_report.get("reconnects") or {}).get("auto", 0) and not args.allow_reconnects:
            warnings.append(
                "the probe auto-reconnected after TCP errors during the soak; "
                "continuity was recovered but the network/board was not perfectly stable"
            )

    if args.ssh and not thermal_samples:
        warnings.append("thermal samples requested via --ssh but none were collected")
    elif not args.ssh:
        warnings.append(
            "no thermal data ( --ssh not given): board throttling during the soak is NOT verified"
        )
    thermal_summary = summarize_thermal(thermal_samples)
    max_temp = thermal_summary.get("temperature_max_c")
    if max_temp is not None and max_temp >= args.warn_temp_c:
        warnings.append(f"board temperature reached {max_temp:.1f} C (>= {args.warn_temp_c:.0f} C)")

    # status summary. Samples taken before the probe switched the requested mode describe
    # the previous experiment, so counts are restricted to polls where requested_mode
    # already equals the soak's mode.
    mode_samples = [sample for sample in status_samples
                    if sample.get("requested_mode") == args.mode]
    last_rtfs = []
    if args.mode in ("cpu", "npu"):
        # Only worker modes have a meaningful server-side RTF; last_rtf can also be stale
        # from a previous mode, so require actual_mode to match during this sample.
        last_rtfs = [float(sample["last_rtf"]) for sample in mode_samples
                     if sample.get("actual_mode") == args.mode
                     and isinstance(sample.get("last_rtf"), (int, float)) and sample["last_rtf"]]
    status_summary = {
        "samples": len(status_samples),
        "actual_mode_counts": {},
        "fallback_reasons_seen": sorted({s.get("fallback_reason") for s in status_samples
                                         if s.get("fallback_reason")}),
        "last_rtf": {
            "samples": len(last_rtfs),
            "p50": _percentile(last_rtfs, 0.50),
            "p95": _percentile(last_rtfs, 0.95),
            "max": max(last_rtfs) if last_rtfs else None,
            "note": "server-side worker RTF samples (elapsed/inference-audio-duration) "
                    "taken only while actual_mode == requested cpu/npu mode; empty for "
                    "passthrough or when a worker was never engaged",
        },
        "model_hash_start": (status_samples[0].get("model_hash") if status_samples else None)
                             or (status_start.get("model_hash")),
        "model_hash_end": status_samples[-1].get("model_hash") if status_samples else None,
    }
    status_summary["samples_with_requested_mode"] = len(mode_samples)
    for sample in mode_samples:
        key = str(sample.get("actual_mode"))
        status_summary["actual_mode_counts"][key] = status_summary["actual_mode_counts"].get(key, 0) + 1
    if args.mode != "passthrough":
        engaged = sum(1 for sample in mode_samples if sample.get("actual_mode") == args.mode)
        total = len(mode_samples)
        status_summary["mode_engagement_samples"] = total
        status_summary["requested_mode_engaged_ratio"] = (
            round(engaged / total, 4) if total else None)
        if total and engaged < 0.99 * total:
            failures.append(
                f"requested_mode_engaged_ratio {engaged}/{total} < 0.99: actual_mode was not "
                f"{args.mode} for the whole soak while it was requested"
            )

    if not status_samples:
        failures.append("no /v1/status samples were collected during the soak")
    report.update({
        "status_summary": status_summary,
        "status_samples": status_samples,
        "thermal_summary": thermal_summary,
        "thermal_samples": thermal_samples,
        "failures": failures,
        "warnings": warnings,
        "verdict": "PASS" if not failures else "FAIL",
        "pass": not failures,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": 0 if not failures else 1,
    })
    return report, 0 if not failures else 1


# ------------------------------------------------------------------- report/printing

def write_report_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def print_summary(report: dict) -> None:
    log("")
    log("=== soak_edge2 summary (engineering continuity only) ========================")
    log(f"board        : {report['host']} audio={report['audio_port']} "
        f"http={report['http_port']} reachable={report.get('board_reachable')}")
    log(f"mode         : requested={report.get('requested_mode')}")
    log(f"duration     : requested={report.get('duration_requested_s', 0) / 60.0:.2f} min, "
        f"actual={report.get('duration_actual_s', 0):.1f}s")
    streamer = report.get("streamer_report") or {}
    if streamer:
        latency = streamer.get("latency_ms") or {}
        log(f"stream       : chunks_ok={streamer.get('chunks_ok')}/"
            f"{streamer.get('chunks_target')} audio="
            f"{streamer.get('audio_seconds_streamed')}s "
            f"realtime_ratio={streamer.get('realtime_ratio')}")
        log(f"latency ms   : p50={latency.get('p50')} p95={latency.get('p95')} "
            f"p99={latency.get('p99')} max={latency.get('max')}")
        log(f"deadline     : misses={streamer.get('deadline_misses')} "
            f"miss_ratio={streamer.get('miss_ratio')} "
            f"reconnects={streamer.get('reconnects')} recoveries={streamer.get('recoveries')}")
        log(f"fallbacks    : delta={streamer.get('fallbacks_delta')} "
            f"(max allowed {report['criteria']['max_fallbacks']}) "
            f"rejected_delta={streamer.get('rejected_delta')}")
    summary = report.get("status_summary") or {}
    if summary:
        rtfs = summary.get("last_rtf") or {}
        log(f"status       : {summary.get('samples')} samples, "
            f"actual_mode_counts={summary.get('actual_mode_counts')} "
            f"fallback_reasons={summary.get('fallback_reasons_seen')}")
        if summary.get("requested_mode_engaged_ratio") is not None:
            log(f"mode engage  : ratio={summary['requested_mode_engaged_ratio']} "
                f"(samples where the requested mode was active)")
        log(f"server rtf   : p50={rtfs.get('p50')} p95={rtfs.get('p95')} max={rtfs.get('max')}")
        log(f"model hash   : {summary.get('model_hash_start') or '(unset)'}")
    thermal = report.get("thermal_summary") or {}
    if thermal.get("samples"):
        log(f"thermal      : {thermal['samples']} samples, max={thermal.get('temperature_max_c')} C, "
            f"zone_max={thermal.get('temperature_zone_max_c')}, "
            f"throttle_events_seen={thermal.get('throttle_event_count_seen')}")
    else:
        log("thermal      : NO DATA (board throttling not verified)")
    for warning in report.get("warnings") or []:
        log(f"WARNING      : {warning}")
    for failure in report.get("failures") or []:
        log(f"FAILURE      : {failure}")
    log(f"VERDICT      : {report.get('verdict', 'FAIL')}")
    log("NOTE         : this verdict covers engineering stream continuity/timing only.")
    log("               Audio QUALITY is NOT assessed; it requires human blind listening")
    log("               (tools/listening_ab.py). A soak PASS is not a listening pass.")
    if report.get("streamer_log"):
        log(f"evidence     : {report.get('evidence_dir')}")
    log("=" * 78)


def default_report_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "docs" / f"soak-report-{stamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="30-min Edge2 engineering soak driver (continuity/timing evidence only; "
                    "NOT a listening-quality judge).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--audio-port", type=int, default=9100)
    parser.add_argument("--http-port", type=int, default=8787)
    parser.add_argument("--mode", choices=("passthrough", "cpu", "npu"), default="npu",
                        help="mode requested from the node for the whole soak")
    parser.add_argument("--minutes", type=float, default=30.0, help="soak duration in minutes")
    parser.add_argument("--chunk-ms", type=float, default=100.0,
                        help="PCM chunk duration streamed by tools/run_stream.py")
    parser.add_argument("--input", type=Path, default=None,
                        help="optional source WAV (default: deterministic repo fixture)")
    parser.add_argument("--output", type=Path, default=None,
                        help="optional WAV of the received stream during the soak")
    parser.add_argument("--json-report", type=Path, default=None,
                        help="report path (default docs/soak-report-YYYYmmdd-HHMMSS.json)")
    parser.add_argument("--ssh", default=None,
                        help="optional user@board for thermal/CPU evidence over ssh")
    parser.add_argument("--ssh-opts", default="-o BatchMode=yes -o ConnectTimeout=5 "
                                             "-o StrictHostKeyChecking=accept-new",
                        help="ssh option string for thermal sampling")
    parser.add_argument("--thermal-command", default=None,
                        help="override the remote shell command (debug/testing; combined with --ssh "
                             "it runs remotely, alone it runs locally)")
    parser.add_argument("--thermal-interval", type=float, default=60.0, dest="thermal_interval_s")
    parser.add_argument("--thermal-timeout", type=float, default=20.0, dest="thermal_timeout_s")
    parser.add_argument("--warn-temp-c", type=float, default=90.0,
                        help="warn (not fail) if any thermal zone reaches this temperature")
    parser.add_argument("--max-fallbacks", type=int, default=0,
                        help="FAIL if the node fallback counter increases more than this")
    parser.add_argument("--max-unrecovered-seconds", type=float, default=5.0,
                        help="FAIL if no successful chunk progresses for longer than this")
    parser.add_argument("--max-status-poll-failures", type=int, default=5,
                        help="FAIL after this many consecutive /v1/status failures")
    parser.add_argument("--min-realtime-ratio", type=float, default=0.98,
                        help="FAIL if streamed audio seconds / wall seconds drops below this")
    parser.add_argument("--max-p95-ms", type=float, default=None,
                        help="forwarded to the probe (default: one chunk duration)")
    parser.add_argument("--max-miss-ratio", type=float, default=None,
                        help="forwarded to the probe (default: probe's own default)")
    parser.add_argument("--require-bit-exact", action="store_true",
                        help="forwarded: require passthrough replies to be bit-identical")
    parser.add_argument("--allow-reconnects", action="store_true",
                        help="do not warn when the probe auto-reconnected during the soak")
    parser.add_argument("--wall-grace-factor", type=float, default=1.5,
                        help="FAIL if wall time exceeds requested*this + startup grace")
    parser.add_argument("--startup-grace", type=float, default=10.0, dest="startup_grace_s",
                        help="seconds allowed for the probe to produce its first successful chunk")
    parser.add_argument("--preflight-timeout", type=float, default=5.0,
                        help="seconds to wait for the board/server to answer /v1/status")
    parser.add_argument("--status-interval", type=float, default=1.0, dest="status_interval_s")
    parser.add_argument("--progress-interval", type=float, default=10.0, dest="progress_interval_s")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.minutes <= 0:
        log("FAIL: --minutes must be > 0")
        return 2
    if args.chunk_ms <= 0:
        log("FAIL: --chunk-ms must be > 0")
        return 2
    report_path = args.json_report or default_report_path()
    log(f"[soak] starting {args.minutes:.2f} min soak: host={args.host} mode={args.mode} "
        f"report={report_path}")
    log("[soak] NOTE: engineering continuity only; audio quality is a human listening decision.")

    report, exit_code = run_soak(args)
    write_report_report(report, report_path)
    print_summary(report)
    log(f"report       : {report_path}")
    if exit_code == 2:
        log("RESULT: FAIL (board/server unreachable or unusable; NO soak evidence was produced)")
        return 2
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
