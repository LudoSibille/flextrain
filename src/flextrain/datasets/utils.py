import os
from typing import Any, Optional

from torch.utils.data import Dataset

from ..trainer.utils import default
from ..types import Batch, Transform


def get_data_root(path: Optional[str]) -> str:
    """
    Return the default data root directory
    """
    if path is not None:
        return path

    path: Optional[str] = default('DATASETS_ROOT', default_value=None)
    if path is not None:
        return path

    if os.name == 'nt':
        return 'c:/tmp'

    return '/tmp'


class DatasetDict(Dataset):
    def __init__(self, transform: Optional[Transform] = None, **kwargs: Any):
        super().__init__()
        self.kwargs = kwargs
        self.first_key = next(iter(kwargs.keys()))
        self.transform = transform
        for _, value in kwargs.items():
            assert len(kwargs[self.first_key]) == len(value)

    def __len__(self) -> int:
        return len(self.kwargs[self.first_key])

    def __getitem__(self, idx: int) -> Batch:
        b = {name: value[idx] for name, value in self.kwargs.items()}
        if self.transform is not None:
            b = self.transform(b)
        return b
