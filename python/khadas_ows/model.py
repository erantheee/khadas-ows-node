from __future__ import annotations
import torch
from torch import nn

class FreqGating(nn.Module):
    def __init__(self, dim: int, squeeze: int = 8):
        super().__init__()
        self.fc = nn.Sequential(nn.Conv1d(dim, dim // squeeze, 1), nn.ReLU(), nn.Conv1d(dim // squeeze, dim, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.fc(x.mean(dim=-1)).unsqueeze(-1))

class MaskNetwork(nn.Module):
    """NPU boundary: real magnitude [B,2,F,T] -> masks [B,4,F,T]."""
    def __init__(self, base: int = 128, layers: int = 6, stems: int = 4):
        super().__init__(); blocks=[]; channels=2
        for _ in range(layers):
            blocks += [nn.Conv2d(channels, base, (1,3), padding=(0,1)), nn.BatchNorm2d(base), nn.PReLU(),
                       nn.Conv2d(base, base, (3,1), padding=(1,0)), nn.BatchNorm2d(base), nn.PReLU()]
            channels=base
        self.blocks=nn.Sequential(*blocks); self.gate=FreqGating(base); self.mask=nn.Conv2d(base, stems, 1)
    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.mask(self.gate(self.blocks(magnitude))), dim=1)

def load_legacy_checkpoint(path: str) -> MaskNetwork:
    model=MaskNetwork(); state=torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected=model.load_state_dict(state, strict=False)
    allowed={k for k in state if k.startswith(("blocks.","gate.","mask."))}
    if unexpected or any(k not in allowed for k in state) or missing:
        raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
    return model.eval()

