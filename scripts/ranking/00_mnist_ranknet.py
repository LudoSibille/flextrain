import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from flextrain.callbacks.epoch_summary import CallbackLogMetrics
from flextrain.classification.model_mnist import create_model
from flextrain.contrastive.lightning import MultiBatchLightning
from flextrain.datasets.dataset_aggregator import (
    DatasetAggregator,
    merge_batches,
    split_batch,
)
from flextrain.datasets.mnist import mnist_dataset
from flextrain.layers.utils import ModelBatchAdaptor
from flextrain.losses import LossRanking
from flextrain.trainer.options import Options
from flextrain.trainer.start_training import start_training

options = Options()
options.training.nb_epochs = 50
options.training.precision = 16
options.training.devices = '0'
device = torch.device('cuda:0')


merge_batches_fn = partial(merge_batches, keys_to_merge=('images', 'targets'))
split_batch_fn = partial(split_batch, keys_to_unmerge=('images', 'targets'))


Aggregator = partial(DatasetAggregator, number_of_batches=2, merge_batches_fn=merge_batches_fn)

batch_size = 1000
datasets = mnist_dataset(
    batch_size=batch_size,
    max_train_samples=None,
    num_workers=0,
    persistent_workers=False,
    dataset_transformer_train=Aggregator,
    dataset_transformer_valid=Aggregator,
)

model = create_model(nb_classes=1)
model = nn.Sequential(model, nn.Sigmoid())  # bound the outputs to [0..1]


loss_fn = LossRanking(true_ordering_fn=lambda batche_0, batche_1: batche_0['targets'] < batche_1['targets'])


model = ModelBatchAdaptor(model, ['images'])
model = torch.compile(model)
model_pl = MultiBatchLightning(
    model, split_batch_fn=partial(split_batch, keys_to_unmerge=('images', 'targets')), loss_fn=loss_fn
)

callbacks = [
    CallbackLogMetrics(),
]
start_training(options, datasets, callbacks, model_pl)


def compute_embeddings(model_pl, device, dataset):
    model_pl = model_pl.to(device)
    embeddings_all = []
    targets_all = []
    images_all = []
    for super_batch in dataset:
        batches = split_batch_fn(super_batch)
        for batch in batches:
            images = batch['images'].to(device)
            targets = batch['targets']
            embeddings = model_pl({'images': images})

            embeddings_all.append(embeddings.detach().cpu())
            targets_all.append(targets.detach().cpu())
            images_all.append(images.cpu())

    embeddings_all = torch.concatenate(embeddings_all)
    targets_all = torch.concatenate(targets_all).squeeze()
    return embeddings_all, targets_all, images_all


embeddings_all_v, targets_all_v, images_all_v = compute_embeddings(model_pl, device, datasets['mnist']['valid'])

# here we expect mapping 0->9 => [0..1]
plt.close()
plt.scatter(
    x=embeddings_all_v.squeeze().numpy(),
    y=np.random.uniform(-1, 1, size=[len(embeddings_all_v)]),
    c=targets_all_v.numpy(),
    cmap=plt.cm.rainbow,
)
plt.colorbar()
plt.savefig(os.path.join(options.data.root_current_experiment, 'embeddings_valid.png'))
