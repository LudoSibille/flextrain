from typing import Any, Callable, Optional, Sequence

import torch
from flexdat.types import Batch
from torch import nn


class ModelBatchAdaptor(nn.Module):
    """
    Transform a model taking a sequence of tensors to a model with input a batch

    >>> base_model = torch.Conv2d(3, 3, kernel_size=5, padding='same')
    >>> model = ModelBatchAdaptor(base_model, input_names='image')
    >>> o = model({'image': torch.zeros([10, 3, 64, 64])})
    >>> o.shape
    (10, 3, 64, 64)
    """

    def __init__(
        self,
        base_model: nn.Module,
        input_names: Sequence[str],
        preprocessing_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        self.input_names = input_names
        self.base_model = base_model
        self.preprocessing_fn = preprocessing_fn

    def forward(self, batch: Batch) -> Any:
        i = [batch[name] for name in self.input_names]
        if self.preprocessing_fn is not None:
            i = [self.preprocessing_fn(value) for value in i]
        return self.base_model(*i)


class ModelAdaptor3to2(nn.Module):
    """
    For 2.5D model, we often require NCHW (C=number slices) tensor rather than
    a NCDHW with C=1 and D=number of slices

    So convert from 3D representation to 2D and back:

    >>> base_model = torch.Conv2d(3, 3, kernel_size=5, padding='same')
    >>> model = ModelAdaptor3to2(base_model)
    >>> model(torch.zeros([10, 1, 3, 64, 64])).shape
    (10, 1, 3, 64, 64)
    """

    def __init__(self, base_model: nn.Module, allow_2d_input: bool = False) -> None:
        super().__init__()

        self.base_model = base_model
        self.allow_2d_input = allow_2d_input

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        if len(i.shape) == 4 and self.allow_2d_input:
            # already a 2D input, nothing to do
            return self.base_model(i)

        assert len(i.shape) == 5, f'expecting NCDHW with C=1. Got={i.shape}'
        assert i.shape[1] == 1, f'expecting C=1, got shape={i.shape}'
        o = self.base_model(i.squeeze(1))  # we need a copy! (cant use squeeze_)
        assert len(o.shape) == 4, f'expecting NCHW. Got={o.shape}'
        return o.unsqueeze(1)
