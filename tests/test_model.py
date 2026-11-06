import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/"python"))
from khadas_ows.model import MaskNetwork
def test_mask_contract():
    model=MaskNetwork(base=8,layers=1); x=torch.rand(1,2,33,9); y=model(x)
    assert y.shape==(1,4,33,9); assert torch.isfinite(y).all(); assert torch.allclose(y.sum(1),torch.ones(1,33,9),atol=1e-5)

