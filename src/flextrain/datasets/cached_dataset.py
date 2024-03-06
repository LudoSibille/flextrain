import logging
import os
import pickle as pkl
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..types import Batch

logger = logging.getLogger(__name__)


def read_data_pkl(path: str) -> Sequence[Dict]:
    with open(path, 'rb') as f:
        return pkl.load(f)


def write_data_pkl(data: Sequence[Dict], path: str) -> None:
    with open(path, 'wb') as f:
        pkl.dump(data, f, protocol=5)


def chunking_slice_fn(v: Any, max_dim: int = 64) -> Optional[Tuple[int, ...]]:
    if isinstance(v, (np.ndarray, torch.Tensor)):
        if len(v.shape) >= 3:
            chunks = tuple([1] + list(np.minimum(np.asarray(v.shape[1:]), max_dim)))
            return chunks

    return None


def write_data_h5(
    data: Sequence[Dict],
    path: str,
    compression='gzip',
    chunking_fn: Callable[[Any], Optional[Tuple[int, ...]]] = chunking_slice_fn,
) -> None:
    filled_indices = []
    values = []
    for i in range(len(data)):
        value = data[i]
        if value is not None:
            filled_indices.append(i)
            values.append(value)
    filled_indices = np.asarray(filled_indices)

    values_dict = torch.utils.data.default_collate(values)
    with h5py.File(path, 'w') as f:
        for name, value in values_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            chunks = chunking_fn(value) if chunking_fn is not None else None
            f.create_dataset(name, data=value, compression=compression, chunks=chunks)
        f.create_dataset('filled_indices', data=filled_indices, compression=compression)
        f.create_dataset('total_size', data=len(data))


def read_data_h5(path: str) -> Sequence[Dict]:
    with h5py.File(path, 'r') as f:
        filled_indices = f['filled_indices'][()]
        total_size = f['total_size'][()]

        data_dict = {}
        for name in f.keys():
            if name not in ('filled_indices', 'total_size'):
                value = f[name][()]
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                data_dict[name] = value

    data = [None] * total_size
    for i_n, i in enumerate(filled_indices):
        v_d = {}
        for name, value in data_dict.items():
            v_d[name] = value[i_n]
        data[i] = v_d

    return data


class DatasetCachedMemory(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        path_to_cache: Optional[str] = None,
        mode: Literal['r', 'w', 'a'] = 'r',
        new_dataset_size: Optional[None] = None,
        flush_ratio_increment: float = 0.0999,
        read_data_fn: Callable[[str], Sequence[Dict]] = read_data_h5,
        write_data_fn: Callable[[str, Sequence[Dict]], None] = write_data_h5,
        transforms: Optional[Callable[[Batch], Batch]] = None,
    ) -> None:
        """
        Dataset that keeps a cache to re-use lengthy data-processing time

        The cache is kept fully in memory.

        The cache can be exported or imported from a file by specifying `path_to_cache`.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.path_to_cache = path_to_cache
        self.mode = mode
        self.new_dataset_size = new_dataset_size
        self.cache_entries_filled = 0
        self.flush_ratio_increment = flush_ratio_increment
        self.cached_samples = None
        self.read_data_fn = read_data_fn
        self.write_data_fn = write_data_fn
        self.transforms = transforms

        if path_to_cache is not None and (mode == 'r' or mode == 'a'):
            if os.path.exists(path_to_cache):
                try:
                    self.cached_samples = self.read_data_fn(self.path_to_cache)
                    if new_dataset_size is None:
                        # by default, re-use the previous setting
                        self.new_dataset_size = len(self.cached_samples)

                    if self.new_dataset_size is not None and len(self.cached_samples) != self.new_dataset_size:
                        # resize the array as specified
                        if len(self.cached_samples) > self.new_dataset_size:
                            self.cached_samples = self.cached_samples[: self.new_dataset_size]
                        else:
                            self.cached_samples.extend([None] * (self.new_dataset_size - len(self.cached_samples)))

                    for v in self.cached_samples:
                        if v is not None:
                            self.cache_entries_filled += 1

                except KeyError as e:
                    # malformed data
                    logger.error(f'file={path_to_cache} could not be read. E={e}, discarding content!')
                    # revert to clean state
                    self.cached_samples = None
                    self.cache_entries_filled = 0

        if self.cached_samples is None:
            # default to empty cache
            self.cached_samples = [None] * len(self)

        self.last_caching_filled_ratio = self.get_filled_cache_ratio()

    def flush(self):
        if self.path_to_cache is not None:
            current_ratio = self.get_filled_cache_ratio()
            if current_ratio != self.last_caching_filled_ratio:  # don't over-write if we haven't made progress
                if self.mode == 'a' or self.mode == 'w':
                    self.write_data_fn(self.cached_samples, self.path_to_cache)

                self.last_caching_filled_ratio = self.get_filled_cache_ratio()

    def get_filled_cache_ratio(self):
        return float(self.cache_entries_filled) / len(self)

    def __getitem__(self, index) -> Batch:
        value = self.cached_samples[index]
        if value is None:
            if self.new_dataset_size is None:
                new_index = index
            else:
                new_index = np.random.randint(len(self.base_dataset))
            value = self.base_dataset[new_index]
            self.cached_samples[index] = value
            self.cache_entries_filled += 1

            # export from time to time the dataset. This should only
            # happen when a new entry is calculated
            current_filled_cache_ratio = self.get_filled_cache_ratio()
            if current_filled_cache_ratio - self.last_caching_filled_ratio >= self.flush_ratio_increment:
                self.flush()

        if self.transforms is not None:
            value = self.transforms(value)

        return value

    def __len__(self) -> int:
        if self.new_dataset_size is None:
            return len(self.base_dataset)
        else:
            return self.new_dataset_size
