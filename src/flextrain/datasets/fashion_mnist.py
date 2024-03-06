from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from ..types import Datasets
from .utils import DatasetDict, get_data_root


def fashion_mnist_dataset(
    batch_size: int = 1000,
    root: str = None,
    num_workers: int = 4,
    normalize_0_1: bool = True,
    select_classes_train: Optional[Sequence[int]] = None,
    select_classes_test: Optional[Sequence[int]] = None,
    transform_train=None,
    transform_valid=None,
    max_train_samples=None,
    shuffle_valid: bool = False,
    dataset_transformer_train: Optional[Callable[[Dataset], Dataset]] = None,
    dataset_transformer_valid: Optional[Callable[[Dataset], Dataset]] = None,
    persistent_workers: bool = True,
) -> Datasets:

    root = get_data_root(root)

    train_dataset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True)

    valid_dataset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True)

    def get_split(dataset, select_classes=None, transform=None, shuffle=False, max_samples=None, dataset_transformer=None):
        normalization_factor = 1.0 if not normalize_0_1 else 255.0
        ds = {
            'images': dataset.data.view((-1, 1, 28, 28)).float().numpy() / normalization_factor,
            'targets': dataset.targets.view(-1, 1),
        }

        if select_classes is not None:
            indices = np.where(np.in1d(dataset.targets, np.asarray(select_classes)))
            ds = {'images': ds['images'][indices], 'targets': ds['targets'][indices]}

        ds = DatasetDict(**ds, transform=transform)
        if dataset_transformer is not None:
            ds = dataset_transformer(ds)

        if max_samples is not None:
            assert not shuffle, 'incompatible!'
            # num_samples: restrict the number of samples per epoch
            sampler = torch.utils.data.RandomSampler(data_source=ds, num_samples=max_samples)
        else:
            if shuffle:
                sampler = torch.utils.data.RandomSampler(data_source=ds)
            else:
                sampler = torch.utils.data.SequentialSampler(data_source=ds)

        return DataLoader(
            dataset=ds,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            persistent_workers=persistent_workers,
        )

    splits = {
        'train': get_split(
            train_dataset,
            select_classes=select_classes_train,
            transform=transform_train,
            shuffle=True,
            max_samples=max_train_samples,
            dataset_transformer=dataset_transformer_train,
        ),
        'valid': get_split(
            valid_dataset,
            select_classes=select_classes_test,
            transform=transform_valid,
            shuffle=shuffle_valid,
            dataset_transformer=dataset_transformer_valid,
        ),
    }

    datasets = {'mnist': splits}

    return datasets
