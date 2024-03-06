"""
Train a classifier on MNist, 28x28 original dataset.

This time use normalization layers
"""

from functools import partial

import torch
from torch import nn

from flextrain.callbacks.activation_stats import CallbackLogLayerStatistics
from flextrain.callbacks.epoch_summary import CallbackLogMetrics
from flextrain.callbacks.model_summary import CallbackModelSummary
from flextrain.callbacks.record_lr_mom import CallbackRecordLearninRateMomentum
from flextrain.classification.lightning import ClassifierLightning
from flextrain.datasets.fashion_mnist import fashion_mnist_dataset
from flextrain.layers.activations import GeneralReLU
from flextrain.layers.utils import ModelBatchAdaptor
from flextrain.losses import LossCrossEntropy
from flextrain.trainer.optimization import scheduler_one_cycle_cosine_fn
from flextrain.trainer.options import Options
from flextrain.trainer.start_training import start_training

options = Options()
options.training.nb_epochs = 25
options.training.precision = 32
device = torch.device('cuda:0')
options.training.devices = str(device.index)

batch_size = 100
datasets = fashion_mnist_dataset(batch_size=batch_size, max_train_samples=None, num_workers=10, shuffle_valid=False)

leaky_factor = 0.1
sub_factor = 0.5 - leaky_factor


act_fn = partial(GeneralReLU, leak=leaky_factor, sub=sub_factor)


def init_weight_(m, leaky: float = leaky_factor):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight, a=leaky)


def conv(ni, nf, ks=3, act=act_fn, norm=nn.BatchNorm2d):
    layers = [nn.Conv2d(ni, nf, kernel_size=ks, stride=2, padding=ks // 2)]
    if norm:
        layers.append(norm(nf))
    if act:
        layers.append(act())
    return nn.Sequential(*layers)


def baseline():
    layers = [conv(1, 8), conv(8, 16), conv(16, 32), conv(32, 64), conv(64, 10, act=False), nn.Flatten()]
    return nn.Sequential(*layers)


model = baseline()
model = ModelBatchAdaptor(model, ['images'], preprocessing_fn=lambda x: (x - x.mean()) / x.std())
model.apply(partial(init_weight_, leaky=leaky_factor))


ddpm_pl = ClassifierLightning(
    model,
    loss_fn=LossCrossEntropy(target_name='targets'),
    optimizer_fn=torch.optim.AdamW,
    scheduler_steps_fn=partial(
        scheduler_one_cycle_cosine_fn, max_lr=1e-1, total_steps=options.training.nb_epochs * 60000 // batch_size
    ),
)

lr_recorder = CallbackRecordLearninRateMomentum()
callbacks = [
    CallbackModelSummary(),
    CallbackLogMetrics(),
    CallbackLogLayerStatistics(collect_layers_fn=lambda m: type(m) in [GeneralReLU]),
    lr_recorder,
]
start_training(options, datasets, callbacks, ddpm_pl)
