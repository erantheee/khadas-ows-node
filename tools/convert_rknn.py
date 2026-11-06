#!/usr/bin/env python3
import argparse, json
from pathlib import Path
def main():
    p=argparse.ArgumentParser(); p.add_argument("--onnx",required=True); p.add_argument("--output",required=True); p.add_argument("--dataset",help="calibration dataset list; enables INT8")
    a=p.parse_args()
    from rknn.api import RKNN
    r=RKNN(verbose=True); assert r.config(target_platform="rk3588")==0; assert r.load_onnx(model=a.onnx)==0
    assert r.build(do_quantization=bool(a.dataset),dataset=a.dataset)==0; assert r.export_rknn(a.output)==0; r.release()
    print(json.dumps({"rknn":str(Path(a.output)),"quantized":bool(a.dataset)},indent=2))
if __name__=="__main__": main()

