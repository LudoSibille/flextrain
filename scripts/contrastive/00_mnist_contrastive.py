import torch

torch.set_num_threads(8)

import os
from functools import partial
from typing import Sequence

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

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
from flextrain.losses import LossContrastive, contrastive_loss
from flextrain.trainer.options import Options
from flextrain.trainer.start_training import start_training
from flextrain.types import Batch, TorchTensorN, TorchTensorNX

options = Options()
options.training.nb_epochs = 101
options.training.precision = 16
options.training.devices = '0'
device = torch.device('cuda:0')


merge_batches_fn = partial(merge_batches, keys_to_merge=('images', 'targets'))
split_batch_fn = partial(split_batch, keys_to_unmerge=('images', 'targets'))

Aggregator = partial(DatasetAggregator, number_of_batches=2, merge_batches_fn=merge_batches_fn)

datasets = mnist_dataset(
    batch_size=500,
    max_train_samples=None,
    num_workers=10,
    dataset_transformer_train=Aggregator,
    dataset_transformer_valid=Aggregator,
)


loss_fn = LossContrastive(same_class_fn=lambda batches: (batches[0]['targets'] == batches[1]['targets']).float().squeeze())

model = create_model(nb_classes=2)
model = ModelBatchAdaptor(model, ['images'])
model = torch.compile(model)
model_pl = MultiBatchLightning(model, split_batch_fn=split_batch_fn, loss_fn=loss_fn)

callbacks = [
    CallbackLogMetrics(),
]
start_training(options, datasets, callbacks, model_pl)


def compute_embeddings(model_pl, device, dataset):
    model_pl = model_pl.to(device)
    embeddings_all = []
    targets_all = []
    for super_batch in dataset:
        batches = split_batch_fn(super_batch)
        for batch in batches:
            images = batch['images'].to(device)
            targets = batch['targets']
            embeddings = model_pl({'images': images})

            embeddings_all.append(embeddings.detach().cpu())
            targets_all.append(targets.detach().cpu())

    embeddings_all = torch.concatenate(embeddings_all)
    targets_all = torch.concatenate(targets_all).squeeze()
    return embeddings_all, targets_all


plt.close()
embeddings_all_v, targets_all_v = compute_embeddings(model_pl, device, datasets['mnist']['valid'])
plt.scatter(x=embeddings_all_v[:, 0].numpy(), y=embeddings_all_v[:, 1].numpy(), c=targets_all_v.numpy(), cmap=plt.cm.rainbow)
plt.colorbar()
plt.savefig(os.path.join(options.data.root_current_experiment, 'embeddings_valid.png'))


plt.close()
embeddings_all_t, targets_all_t = compute_embeddings(model_pl, device, datasets['mnist']['train'])
plt.scatter(x=embeddings_all_t[:, 0].numpy(), y=embeddings_all_t[:, 1].numpy(), c=targets_all_t.numpy(), cmap=plt.cm.rainbow)
plt.colorbar()
plt.savefig(os.path.join(options.data.root_current_experiment, 'embeddings_train.png'))


classifier = KNeighborsClassifier()
classifier.fit(X=embeddings_all_t.numpy(), y=targets_all_t.numpy())

targets_all_v_predicted = classifier.predict(embeddings_all_v)
accuracy_v = (targets_all_v_predicted == targets_all_v.numpy()).sum() / float(len(targets_all_v))

# expected ~98.5% accuracy
print(f'Accuracy Validation={accuracy_v}')
print('Training done!')
