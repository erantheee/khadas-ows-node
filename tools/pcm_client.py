#!/usr/bin/env python3
import argparse, socket, struct, wave
import numpy as np
HEADER=struct.Struct("<IIQIHHi"); MAGIC=0x4B4F5753
def exact(sock,n):
    data=bytearray()
    while len(data)<n:
        chunk=sock.recv(n-len(data))
        if not chunk: raise EOFError
        data.extend(chunk)
    return bytes(data)
def main():
    p=argparse.ArgumentParser(); p.add_argument("input"); p.add_argument("output"); p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=9100); p.add_argument("--chunk",type=int,default=88200)
    a=p.parse_args(); w=wave.open(a.input,"rb"); assert w.getframerate()==44100 and w.getnchannels()==2 and w.getsampwidth()==2
    out=wave.open(a.output,"wb"); out.setparams(w.getparams()); sock=socket.create_connection((a.host,a.port)); seq=0
    while raw:=w.readframes(a.chunk):
        pcm=np.frombuffer(raw,dtype="<i2").astype("<f4")/32768; frames=len(pcm)//2; header=HEADER.pack(MAGIC,1,seq,44100,2,0,frames)
        sock.sendall(header+pcm.tobytes()); exact(sock,HEADER.size); result=np.frombuffer(exact(sock,frames*8),dtype="<f4"); out.writeframes((np.clip(result,-1,1)*32767).astype("<i2").tobytes()); seq+=1
    out.close(); w.close(); sock.close()
if __name__=="__main__": main()

