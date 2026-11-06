import json, os, socket, struct, subprocess, time, urllib.request
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1]; HEADER=struct.Struct("<IIQIHHi"); MAGIC=0x4B4F5753
def wait_status():
    for _ in range(80):
        try: return json.load(urllib.request.urlopen("http://127.0.0.1:8787/v1/status",timeout=.2))
        except Exception: time.sleep(.05)
    raise AssertionError("server not ready")
def exact(sock,n):
    data=bytearray()
    while len(data)<n: data.extend(sock.recv(n-len(data)))
    return bytes(data)
def test_network_npu_failure_is_bit_exact_passthrough():
    binary=ROOT/"build"/"khadas-ows-node"
    if not binary.exists(): return
    process=subprocess.Popen([str(binary)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        wait_status(); request=urllib.request.Request("http://127.0.0.1:8787/v1/mode",data=b"npu",method="POST"); urllib.request.urlopen(request).read()
        pcm=np.linspace(-.2,.2,400,dtype="<f4"); header=HEADER.pack(MAGIC,1,7,44100,2,0,200)
        with socket.create_connection(("127.0.0.1",9100)) as sock:
            sock.sendall(header+pcm.tobytes()); exact(sock,HEADER.size); output=np.frombuffer(exact(sock,pcm.nbytes),dtype="<f4")
        assert np.array_equal(output,pcm)
        status=wait_status(); assert status["actual_mode"]=="passthrough" and status["fallback_reason"]=="worker_unavailable"
    finally:
        process.terminate(); process.wait(timeout=3)
