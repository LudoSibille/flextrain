from typing import Any, Sequence, Callable, Optional
from torch import nn
import torch


class ModelBatchAdaptor(nn.Module):
    def __init__(self, base_model: nn.Module, input_names: Sequence[str], preprocessing_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        super().__init__()

        self.input_names = input_names
        self.base_model = base_model
        self.preprocessing_fn = preprocessing_fn

    def forward(self, batch) -> Any:
        i = [batch[name] for name in self.input_names]
        if self.preprocessing_fn is not None:
            i = [self.preprocessing_fn(value) for value in i]
        return self.base_model(*i)
