import torch
from torch import nn
from typing import Optional


class GeneralReLU(nn.Module):
    """
    Jeremy Howard's general relu function: in order to have a mean of 0, we need to be able to have negative values

    Here combine leaky with small value subtracted from the activation 
    """
    def __init__(self, leak: Optional[float] = 0.1, sub: Optional[float] = 0.4, max_value: Optional[float] = 10.0) -> None:
        super().__init__()
        self.leak = leak
        self.sub = sub
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.leaky_relu(x, self.leak) if self.leak is not None else nn.functional.relu(x)
        if self.sub is not None:
            x -= self.sub
        if self.max_value is not None:
            x.clamp_max_(self.max_value)
        return x