"""
Train a classifier on MNist, 28x28 original dataset.

LSUV initialization for complex activation functions.
"""

from functools import partial

import torch
from torch import nn

from flextrain.callbacks.activation_stats import CallbackLogLayerStatistics
from flextrain.callbacks.epoch_summary import CallbackLogMetrics
from flextrain.classification.lightning import ClassifierLightning
from flextrain.datasets.fashion_mnist import fashion_mnist_dataset
from flextrain.layers.activations import GeneralReLU
from flextrain.layers.initialization_lsuv import lsuv_
from flextrain.layers.utils import ModelBatchAdaptor
from flextrain.losses import LossCrossEntropy
from flextrain.trainer.options import Options
from flextrain.trainer.start_training import start_training

options = Options()
options.training.nb_epochs = 4
options.training.precision = 16
device = torch.device('cuda:0')
options.training.devices = str(device.index)

batch_size = 1000
datasets = fashion_mnist_dataset(batch_size=batch_size, max_train_samples=None, num_workers=10, shuffle_valid=False)

leaky_factor = 0.1
sub_factor = 0.5 - leaky_factor


act_fn = partial(GeneralReLU, leak=leaky_factor, sub=sub_factor)


def conv(ni, nf, ks=3, act=act_fn):
    res = nn.Conv2d(ni, nf, kernel_size=ks, stride=2, padding=ks // 2)
    if act:
        res = nn.Sequential(res, act())
    return res


def baseline():
    layers = [conv(1, 8), conv(8, 16), conv(16, 32), conv(32, 64), conv(64, 10, act=False), nn.Flatten()]
    return nn.Sequential(*layers)


model = baseline()
model = ModelBatchAdaptor(model, ['images'], preprocessing_fn=lambda x: (x - x.mean()) / x.std())
lsuv_(model, datasets['mnist']['train'], device=device, nb_batches=10)


ddpm_pl = ClassifierLightning(
    model,
    loss_fn=LossCrossEntropy(target_name='targets'),
    optimizer_fn=partial(torch.optim.SGD, lr=0.2, nesterov=False, momentum=0.8),
)


callbacks = [
    CallbackLogMetrics(),
    CallbackLogLayerStatistics(collect_layers_fn=lambda m: type(m) in [GeneralReLU]),
]
start_training(options, datasets, callbacks, ddpm_pl)
