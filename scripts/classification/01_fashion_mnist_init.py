"""
Train a classifier on MNist, 28x28 original dataset.

Using the activations statistics to understand (how bad)
this model actually is.

Very large standard deviation of the activation: throughout the
model, activation should be close to 0-mean, 1-std with activation
histogram that is stable and well spread on y-axis 


- input normalization (0-mean, 1-std)
- glorot init
"""

from functools import partial

import torch
from torch import nn

from flextrain.callbacks.activation_stats import CallbackLogLayerStatistics
from flextrain.callbacks.epoch_summary import CallbackLogMetrics
from flextrain.classification.lightning import ClassifierLightning
from flextrain.datasets.fashion_mnist import fashion_mnist_dataset
from flextrain.layers.activations import GeneralReLU
from flextrain.layers.utils import ModelBatchAdaptor
from flextrain.losses import LossCrossEntropy
from flextrain.trainer.options import Options
from flextrain.trainer.start_training import start_training

options = Options()
options.training.nb_epochs = 40
options.training.precision = 32
options.training.devices = '0'

batch_size = 10000
datasets = fashion_mnist_dataset(batch_size=batch_size, max_train_samples=None, num_workers=10, shuffle_valid=False)

leaky_factor = 0.1
sub_factor = 0.5 - leaky_factor


act_fn = partial(GeneralReLU, leak=leaky_factor, sub=sub_factor)


def init_weight_(m, leaky: float = leaky_factor):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight, a=leaky)


def conv(ni, nf, ks=3, act=act_fn):
    res = nn.Conv2d(ni, nf, kernel_size=ks, stride=2, padding=ks // 2)
    if act:
        res = nn.Sequential(res, act())
    return res


def baseline():
    layers = [conv(1, 8), conv(8, 16), conv(16, 32), conv(32, 64), conv(64, 10, act=False), nn.Flatten()]
    return nn.Sequential(*layers)


model = baseline()
model.apply(partial(init_weight_, leaky=0.1))
model = ModelBatchAdaptor(model, ['images'], preprocessing_fn=lambda x: (x - x.mean()) / x.std())
ddpm_pl = ClassifierLightning(
    model,
    loss_fn=LossCrossEntropy(target_name='targets'),
    optimizer_fn=partial(torch.optim.SGD, lr=0.2, nesterov=False, weight_decay=0, momentum=0.85),
    scheduler_steps_fn=None,
)


callbacks = [
    CallbackLogMetrics(),
    CallbackLogLayerStatistics(collect_layers_fn=lambda m: type(m) in [GeneralReLU]),
]
start_training(options, datasets, callbacks, ddpm_pl)
