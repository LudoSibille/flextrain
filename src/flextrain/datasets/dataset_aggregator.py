from torch.utils.data import Dataset
from typing import Callable, Sequence
from ..types import Batch
from numpy.random import randint


def merge_batches(batches: Sequence[Batch], keys_to_merge: Sequence[str]) -> Batch:
    """
    Merge given keys of the batches. Keys that are not given will be discarded from batches
    """
    new_batch = {}
    for batch_n, batch in enumerate(batches):
        for k in keys_to_merge:
            new_batch[f'{k}_{batch_n}'] = batch[k]
    return new_batch


def split_batch(batch: Batch, keys_to_unmerge: Sequence[str], keys_to_keep: Sequence[str]=()) -> Sequence[Batch]:
    batches = []

    while 1:
        b = {}
        for k in keys_to_unmerge:
            values = batch.get(f'{k}_{len(batches)}')
            if values is None:
                # end of sequence
                return batches
            b[k] = values
        for k in keys_to_keep:
            b[k] = batch[k]
        batches.append(b)
    
    raise ValueError('Unreachable!')


class DatasetAggregator(Dataset):
    """
    Aggregate multiple batches of data as a single batch

    This can be useful for self-supervised learning or ranking algorithms to create pairs or triplets of data
    easily
    """
    def __init__(
            self, 
            base_dataset: Dataset, 
            number_of_batches: int, 
            merge_batches_fn: Callable[[Sequence[Batch]], Batch], 
            sampling_fn: Callable[[int, Dataset], int] = lambda n, dataset: randint(len(dataset))) -> None:
        """
        Args:
            number_of_batches: the number of batches to merge into a single batch
            merge_batches_fn: specify how to merge the batches into one
            sampling_fn: specify how to sample the other batches from the dataset given a batch index
        """
        super().__init__()
        self.number_of_batches = number_of_batches
        self.merge_batches_fn = merge_batches_fn
        self.sampling_fn = sampling_fn
        self.base_dataset = base_dataset
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Batch:
        batches = [self.base_dataset[idx]]
        for _ in range(1, self.number_of_batches):
            idx_2 = self.sampling_fn(idx, self.base_dataset)
            batch = self.base_dataset[idx_2]
            batches.append(batch)

        merged_batch = self.merge_batches_fn(batches)
        return merged_batch