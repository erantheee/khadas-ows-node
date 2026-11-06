#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from khadas_ows.model import load_legacy_checkpoint

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--output",required=True); p.add_argument("--frames",type=int,default=345)
    a=p.parse_args(); model=load_legacy_checkpoint(a.checkpoint); sample=torch.rand(1,2,1025,a.frames)
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,sample,output,input_names=["magnitude"],output_names=["masks"],opset_version=18,dynamic_axes=None)
    with torch.no_grad(): masks=model(sample)
    external=output.with_suffix(output.suffix+".data")
    manifest={"source_checkpoint_sha256":hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),"onnx_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"onnx_external_data_sha256":hashlib.sha256(external.read_bytes()).hexdigest() if external.exists() else None,"opset":18,"input_shape":list(sample.shape),"output_shape":list(masks.shape),"finite":bool(torch.isfinite(masks).all()),"mask_sum_max_error":float((masks.sum(1)-1).abs().max())}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()

