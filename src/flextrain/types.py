from typing import Any, Callable, Dict

import torch

# Torch Tensor with N and any other components
TorchTensorNX = torch.Tensor

# Single component tensor
TorchTensorN = torch.Tensor

# Torch Tensor with N, C components and any other components
TorchTensorNCX = torch.Tensor

Batch = Dict[str, Any]  # {name: value}
Dataset = Dict[str, Dict[str, Any]]  # {split_name: {name: value}}
Datasets = Dict[str, Dict[str, Dataset]]  # {dataset_name: dataset}


Transform = Callable[[Batch], Batch]
