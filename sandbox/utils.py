import copy
from typing import Any, Callable, List, Optional

import torch

from flextrain.diffusion.types import Batch


class TransformApplyTo:
    """
    Apply a transform only on a given feature within a batch
    """

    def __init__(self, transform: Callable[[Any], Any], name: str) -> None:
        self.name = name
        self.transform = transform

    def __call__(self, batch: Batch) -> Batch:
        image_tensors = torch.from_numpy(batch[self.name])
        batch[self.name] = self.transform(image_tensors)
        return batch


class TransformLambda:
    """
    Apply a transform of a series of features iwinth a dictionary and possibly
    create new features from the result or update the feature value
    """

    def __init__(self, transform: Callable[[Any], Any], input_names: List[str], output_names: Optional[List[str]]) -> None:
        self.input_names = input_names
        self.output_names = output_names
        self.transform = transform

        if self.output_names is not None:
            assert len(self.input_names) == len(self.output_names)

    def __call__(self, batch: Batch) -> Batch:
        new_batch = copy.copy(batch)
        for input_n, input_name in enumerate(self.input_names):
            t = batch.get(input_name)
            if t is not None:
                if self.output_names is not None:
                    new_batch[self.output_names[input_n]] = self.transform(t)
                else:
                    new_batch[input_name] = self.transform(t)
        return new_batch
