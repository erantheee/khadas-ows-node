#!/usr/bin/env python3
"""Real-time listening: stream a music file through the Edge2 and play the processed result
to the default output device (e.g. the Xiaomi earbuds).

The board processes fixed 4.005 s windows, so audio starts after the first window has been
processed (~5-7 s) and then plays continuously. Ctrl-C stops.

Usage:
    .venv/bin/python tools/live_play.py music.wav
    .venv/bin/python tools/live_play.py music.wav --host 172.16.20.18 --chunk 176640
"""
import argparse
import queue
import socket
import struct
import threading
import time
import wave

import numpy as np
import sounddevice as sd

HEADER = struct.Struct("<IIQIHHi")
MAGIC = 0x4B4F5753


def exact(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError
        data.extend(chunk)
    return bytes(data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9100)
    p.add_argument("--chunk", type=int, default=176640)  # 4.005 s = one model window
    a = p.parse_args()

    w = wave.open(a.input, "rb")
    assert w.getframerate() == 44100 and w.getnchannels() == 2 and w.getsampwidth() == 2
    sock = socket.create_connection((a.host, a.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    play_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    played = [0]

    def player():
        with sd.RawOutputStream(samplerate=44100, channels=2, dtype="float32") as stream:
            while not stop.is_set():
                try:
                    data = play_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                stream.write(data.astype(np.float32).tobytes())
                played[0] += len(data) // 2

    t = threading.Thread(target=player, daemon=True)
    t.start()
    print(f"[live] streaming {a.input} -> {a.host}:{a.port} (chunk {a.chunk} frames); "
          f"audio starts after the first window", flush=True)

    seq = 0
    t0 = time.monotonic()
    try:
        while raw := w.readframes(a.chunk):
            pcm = np.frombuffer(raw, dtype="<i2").astype("<f4") / 32768
            frames = len(pcm) // 2
            header = HEADER.pack(MAGIC, 1, seq, 44100, 2, 0, frames)
            sock.sendall(header + pcm.astype("<f4").tobytes())
            exact(sock, HEADER.size)
            result = np.frombuffer(exact(sock, frames * 8), dtype="<f4")
            play_q.put(result.reshape(-1, 2))
            seq += 1
            rtf = (time.monotonic() - t0) / max(seq * a.chunk / 44100, 1e-9)
            print(f"[live] chunk {seq} processed, queue {play_q.qsize()}, rtf {rtf:.2f}", flush=True)
        # Wait until the player has consumed everything before closing the stream.
        while not play_q.empty():
            time.sleep(0.2)
        time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        # Let the player drain the queue first, then stop the stream.
        while not play_q.empty():
            time.sleep(0.2)
        time.sleep(1.0)
        stop.set()
        sock.close()
        w.close()
        print(f"[live] done: {seq} chunks, played {played[0]} frames", flush=True)


if __name__ == "__main__":
    main()
