"""
Train a classifier on MNist, 28x28 original dataset.

This time use activation normalization
"""

from functools import partial

import torch
from torch import nn

from flextrain.callbacks.activation_stats import CallbackLogLayerStatistics
from flextrain.callbacks.checkpoint import CallbackCheckpoint
from flextrain.callbacks.epoch_summary import CallbackLogMetrics
from flextrain.callbacks.model_summary import CallbackModelSummary
from flextrain.callbacks.record_lr_mom import CallbackRecordLearninRateMomentum
from flextrain.callbacks.samples import CallbackRecordSamples
from flextrain.callbacks.tracing import CallbackExperimentTracing
from flextrain.classification.lightning import ClassifierLightning
from flextrain.datasets.fashion_mnist import fashion_mnist_dataset
from flextrain.layers.activations import GeneralReLU
from flextrain.layers.initialization_lsuv import lsuv_
from flextrain.layers.utils import ModelBatchAdaptor
from flextrain.losses import LossCrossEntropy
from flextrain.trainer.optimization import scheduler_one_cycle_cosine_fn
from flextrain.trainer.options import Options
from flextrain.trainer.start_training import start_training

options = Options()
# options.training.nb_epochs = 25
options.training.nb_epochs = 5
options.training.precision = 32
device = torch.device('cuda:0')
options.training.devices = str(device.index)

batch_size = 100
datasets = fashion_mnist_dataset(batch_size=batch_size, max_train_samples=None, num_workers=10, shuffle_valid=False)

leaky_factor = 0.1
sub_factor = 0.5 - leaky_factor
act_fn = partial(GeneralReLU, leak=leaky_factor, sub=sub_factor)


def conv(ni, nf, ks=3, act=act_fn, norm=nn.BatchNorm2d, stride=2):
    layers = [nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks // 2)]
    if norm:
        layers.append(norm(nf))
    if act:
        layers.append(act())
    return nn.Sequential(*layers)


def _conv_block(ni, nf, stride, act=act_fn, norm=None, ks=3):
    return nn.Sequential(
        conv(ni, nf, stride=1, act=act, norm=norm, ks=ks), conv(nf, nf, stride=stride, act=None, norm=norm, ks=ks)
    )


class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1, ks=3, act=act_fn, norm=None):
        super().__init__()
        self.convs = _conv_block(ni, nf, stride, act=act, ks=ks, norm=norm)
        self.idconv = torch.nn.Identity() if ni == nf else conv(ni, nf, ks=1, stride=1, act=None)
        self.pool = torch.nn.Identity() if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)
        self.act = act()

    def forward(self, x):
        return self.act(self.convs(x) + self.idconv(self.pool(x)))


class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return x.mean((-2, -1))


def baseline(act=act_fn, nfs=(16, 32, 64, 128, 256), norm=nn.BatchNorm2d):
    layers = [conv(1, 16, ks=5, stride=1, act=act, norm=norm)]
    layers += [ResBlock(nfs[i], nfs[i + 1], act=act, norm=norm, stride=2) for i in range(len(nfs) - 1)]
    layers += [GlobalAvgPool(), nn.Linear(256, 10, bias=False), nn.BatchNorm1d(10)]
    return nn.Sequential(*layers)


def init_weight_(m, leaky: float = leaky_factor):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight, a=leaky)


model = baseline()
model = ModelBatchAdaptor(model, ['images'], preprocessing_fn=lambda x: (x - x.mean()) / x.std())
model.apply(partial(init_weight_, leaky=leaky_factor))
# lsuv_(model, datasets['mnist']['train'], device=device, nb_batches=10, norm_fn=None, last_layer_no_activation=False)

ddpm_pl = ClassifierLightning(
    model,
    loss_fn=LossCrossEntropy(target_name='targets'),
    optimizer_fn=torch.optim.AdamW,
    scheduler_steps_fn=partial(
        scheduler_one_cycle_cosine_fn, max_lr=2e-2, total_steps=options.training.nb_epochs * 60000 // batch_size
    ),
)

lr_recorder = CallbackRecordLearninRateMomentum()
callbacks = [
    CallbackExperimentTracing(),
    CallbackRecordSamples(nb_samples=3, nb_repeat=5),
    CallbackModelSummary(),
    CallbackLogMetrics(),
    CallbackLogLayerStatistics(collect_layers_fn=lambda m: type(m) in [GeneralReLU, nn.ReLU]),
    lr_recorder,
    CallbackCheckpoint(metric_name='metric_1-accuracy_valid', metric_to_maximize=False),
]
start_training(options, datasets, callbacks, ddpm_pl)
