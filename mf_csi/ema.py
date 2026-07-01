"""Exponential moving average of model weights.

The EMA weights are what we evaluate (and would ship): they smooth out the
step-to-step noise of the MeanFlow objective and consistently give better,
more stable NMSE than the raw training weights.
"""

from __future__ import annotations

import torch


class EMA:
    def __init__(self, module: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: torch.nn.Module, decay: float = None) -> None:
        d = self.decay if decay is None else decay
        for k, v in module.state_dict().items():
            s = self.shadow[k]
            if torch.is_floating_point(v):
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)                       # ints/bools: just track latest

    def copy_to(self, module: torch.nn.Module) -> None:
        module.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}
