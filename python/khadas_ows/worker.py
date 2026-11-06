from __future__ import annotations
import argparse, hashlib, socket, struct, time
import numpy as np
import torch
from .model import load_legacy_checkpoint

HEADER=struct.Struct("<IIQIHHi")
MAGIC=0x4B4F5753; N_FFT=2048; HOP=512
def exact(conn, size):
    data=bytearray()
    while len(data)<size:
        chunk=conn.recv(size-len(data))
        if not chunk: raise EOFError
        data.extend(chunk)
    return bytes(data)

class Runner:
    def __init__(self, mode, model_path):
        self.mode=mode; self.torch_model=None; self.rknn=None
        if mode=="cpu": self.torch_model=load_legacy_checkpoint(model_path)
        else:
            from rknnlite.api import RKNNLite
            self.rknn=RKNNLite(); assert self.rknn.load_rknn(model_path)==0; assert self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)==0
    def masks(self, magnitude):
        if self.torch_model is not None:
            with torch.no_grad(): return self.torch_model(torch.from_numpy(magnitude)).numpy()
        return np.asarray(self.rknn.inference(inputs=[magnitude])[0])
    def process(self, pcm, channels):
        x=torch.from_numpy(pcm.reshape(-1,channels).T.copy())
        window=torch.hann_window(N_FFT); spec=torch.stft(x,n_fft=N_FFT,hop_length=HOP,window=window,return_complex=True)
        magnitude=spec.abs().unsqueeze(0).numpy().astype(np.float32)
        masks=self.masks(magnitude); vocals=torch.from_numpy(masks[0,0]).to(spec.dtype)*spec
        vocal=torch.istft(vocals,n_fft=N_FFT,hop_length=HOP,window=window,length=x.shape[-1])
        # Engineering probe only: return the estimated vocal stem at matched peak.
        peak=max(float(vocal.abs().max()),1e-6); source_peak=max(float(x.abs().max()),1e-6)
        return (vocal*(source_peak/peak)).T.contiguous().numpy().astype("<f4").reshape(-1)

def serve(mode, model, port):
    runner=Runner(mode,model); server=socket.socket(); server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); server.bind(("127.0.0.1",port)); server.listen()
    print(f"{mode} worker ready port={port} sha256={hashlib.sha256(open(model,'rb').read()).hexdigest()}",flush=True)
    while True:
        conn,_=server.accept()
        with conn:
            try:
                raw=exact(conn,HEADER.size); magic,schema,seq,sr,ch,reserved,frames=HEADER.unpack(raw)
                if magic!=MAGIC or schema!=1 or sr!=44100 or ch!=2: continue
                pcm=np.frombuffer(exact(conn,frames*ch*4),dtype="<f4").copy()
                out=runner.process(pcm,ch); conn.sendall(raw+out.tobytes())
            except (EOFError, OSError, ValueError, RuntimeError) as exc: print(f"worker error: {exc}",flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("cpu","npu"),required=True); p.add_argument("--model",required=True); p.add_argument("--port",type=int)
    a=p.parse_args(); serve(a.mode,a.model,a.port or (9201 if a.mode=="cpu" else 9202))
if __name__=="__main__": main()

