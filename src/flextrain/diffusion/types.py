from typing import Any, Dict
import torch
from ..types import Batch


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def training_step(batch: Batch, _: Any) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()