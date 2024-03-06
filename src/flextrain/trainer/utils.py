import json
import os
import shutil
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..types import *


def default(env_name: str, default_value: Any, output_type: Any = str) -> Any:
    """
    Read the value of an environment variable or return a default value
    if not defined
    """
    value = os.environ.get(env_name)
    if value is not None:
        return output_type(value)
    if default_value is None:
        return None
    return output_type(default_value)


def is_training_split(split_name: str) -> bool:
    """
    Decide wether the split is a training or not based on the split name
    """
    return 'train' in split_name


def transfer_batch_to_device(
    batch: Union[Batch, torch.Tensor], device: torch.device, non_blocking: bool = True
) -> Union[Batch, torch.Tensor]:
    """
    Transfer the Tensors and numpy arrays to the specified device. Other types will not be moved.

    Args:
        batch: the batch of data to be transferred
        device: the device to move the tensors to
        non_blocking: non blocking memory transfer to GPU

    Returns:
        a batch of data on the specified device
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)

    device_batch = {}
    for name, value in batch.items():
        if isinstance(value, np.ndarray):
            # `torch.from_numpy` to keep the same dtype as our input
            device_batch[name] = torch.as_tensor(value).to(device, non_blocking=non_blocking)
        elif isinstance(value, torch.Tensor) and value.device != device:
            device_batch[name] = value.to(device, non_blocking=non_blocking)
        else:
            device_batch[name] = value
    return device_batch


def create_or_recreate_folder(path: str, nb_tries: bool = 3, wait_time_between_tries: float = 2.0) -> None:
    """
    Check if the path exist. If yes, remove the folder then recreate the folder, else create it

    Args:
        path: the path to create or recreate
        nb_tries: the number of tries to be performed before failure
        wait_time_between_tries: the time to wait before the next try

    Returns:
        ``True`` if successful or ``False`` if failed.
    """
    assert len(path) > 6, 'short path? just as a precaution...'
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

    def try_create():
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            print('[ignored] create_or_recreate_folder error:', str(e))
            return False

    # multiple tries (e.g., for windows if we are using explorer in the current path)
    import threading

    for _ in range(nb_tries):
        is_done = try_create()
        if is_done:
            return True
        threading.Event().wait(wait_time_between_tries)  # wait some time for the FS to delete the files
    return False


def dict_as_namespace(d: Dict) -> SimpleNamespace:
    """
    Convert a dictionary to a namespace (i.e., support for the `.` notation)
    """
    x = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(x, k, dict_as_namespace(v))
        else:
            setattr(x, k, v)
    return x


def get_device(module: torch.nn.Module, batch: Batch = None) -> Optional[torch.device]:
    """
    Return the device of a module. This may be incorrect if we have a module split across different devices
    """
    try:
        p = next(module.parameters())
        return p.device
    except StopIteration:
        # the model doesn't have parameters!
        pass

    if batch is not None:
        # try to guess the device from the batch
        for name, value in batch.items():
            if isinstance(value, torch.Tensor):
                return value.device

    # we can't make an appropriate guess, just fail!
    return None


def to_value(v: Any) -> Any:
    """
    Convert where appropriate from tensors to numpy arrays

    Args:
        v: an object. If ``torch.Tensor``, the tensor will be converted to a numpy
            array. Else returns the original ``v``

    Returns:
        ``torch.Tensor`` as numpy arrays. Any other type will be left unchanged
    """
    if isinstance(v, torch.Tensor):
        return v.cpu().data.numpy()
    return v


def is_debug_run() -> bool:
    """
    Return `True` if the script was started from VS Code.
    This is mostly used to trim down the setup time and
    have short iterations used to debug more quickly.
    """
    started_from_vs = bool(os.environ.get('STARTED_WITHIN_VSCODE'))
    return started_from_vs


def safe_lookup(dictionary: Dict, *keys, default: Any = None):
    """
    Recursively access nested dictionaries

    Args:
        dictionary: nested dictionary
        *keys: the keys to access within the nested dictionaries
        default: the default value if dictionary is ``None`` or it doesn't contain
            the keys
    Returns:
        None if we can't access to all the keys, else dictionary[key_0][key_1][...][key_n]
    """
    if dictionary is None:
        return default

    for key in keys:
        dictionary = dictionary.get(key)
        if dictionary is None:
            return default

    return dictionary


def safe_lookup_ns(ns: SimpleNamespace, *keys, default: Any = None) -> Optional[str]:
    """
    Recursively access nested namespace

    Args:
        dictionary: nested namespace
        *keys: the keys to access within the nested namespace
        default: the default value if namespace is ``None`` or it doesn't contain
            the keys
    Returns:
        `default` if we can't access to all the keys, else dictionary.key_0.key_1.{...}.key_n
    """
    assert len(keys) > 0, 'no keys!'
    for k in keys:
        if hasattr(ns, k):
            ns = ns.__getattribute__(k)
        else:
            return default
    return ns


def make_dataloaders_from_datasets(datasets: Datasets, num_workers: int, batch_size: int):
    """
    Create the data loaders based on dataset/split
    """

    # creating worker is very slow, keep them alive!
    # we don't really need reproducible seeds here
    # hence the different init from the test/valid
    datasets_loaders = {}
    data_loader_args_train = {
        # can't have this option if no worker
        'persistent_workers': num_workers
        > 0
    }

    # for reproducibility in the test/validation datasets
    # we need to set the generator seed every time the
    # dataset is iterated. it is CRITICAL to have `persistent_workers=False`
    g = torch.Generator()
    g.manual_seed(0)
    data_loader_args_other = {
        'generator': g,
    }

    def seed_worker(worker_id):
        # create a different seed for each
        # worker, else augmentations would be identical!
        worker_seed = 0 + worker_id
        import numpy

        numpy.random.seed(worker_seed)
        import random

        random.seed(worker_seed)

    for dataset_name, splits in datasets.items():
        dataset_loader = {}
        for split_name, split in splits.items():
            if is_training_split(split_name):
                data_loader = DataLoader(
                    dataset=split, num_workers=num_workers, batch_size=batch_size, shuffle=True, **data_loader_args_train
                )
            else:
                data_loader = DataLoader(
                    dataset=split,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    shuffle=False,
                    **data_loader_args_other,
                    worker_init_fn=seed_worker,
                )
            dataset_loader[split_name] = data_loader
        datasets_loaders[dataset_name] = dataset_loader
    return datasets_loaders


def len_batch(batch: Batch) -> int:
    """

    Args:
        batch: a data split or a `collections.Sequence`

    Returns:
        the number of elements within a data split
    """
    if isinstance(batch, (np.ndarray, torch.Tensor)):
        return len(batch)

    assert isinstance(batch, Dict), 'Must be a dict-like structure! got={}'.format(type(batch))

    for _, values in batch.items():
        if isinstance(values, (list, tuple)):
            return len(values)
        if isinstance(values, torch.Tensor) and len(values.shape) != 0:
            return values.shape[0]
        if isinstance(values, np.ndarray) and len(values.shape) != 0:
            return values.shape[0]
    return 0


from typing import Union


def bytes2human(n: Union[int, float]) -> str:
    """
    Format large number of bytes into readable string for a human

    Examples:
        >>> bytes2human(10000)
        '9.8K'

        >>> bytes2human(100001221)
        '95.4M'

    """
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%.2f" % n


def number2human(n: Union[int, float]) -> str:
    """
    Format large number into readable string for a human

    Examples:
        >>> number2human(1000)
        '1.0K'

        >>> number2human(1200000)
        '1.2M'

    """
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = (10**3) ** (i + 1)
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%.2f" % n


def postprocess_optimizer_scheduler_lightning(optimizer, scheduler) -> Dict:
    """
    Handle the various configuration options for the learning rate scheduler
    """
    if isinstance(scheduler, Dict):
        config = {'optimizer': optimizer, 'lr_scheduler': scheduler}
    else:
        config = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': "epoch",
            },
        }

    return config


class NumpyTorchEncoder(json.JSONEncoder):
    """
    export Numpy/Torch arrays in JSON

    >>> json.dump(obj, f, indent=3, cls=NumpyTorchEncoder)
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return to_value(obj).tolist()
        if isinstance(
            obj, (np.float32, np.float64, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)
        ):
            return float(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf8')
        return json.JSONEncoder.default(self, obj)
