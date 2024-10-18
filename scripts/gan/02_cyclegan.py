import os
from glob import glob
from typing import Any, Optional, Sequence, Union

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from monai.transforms import Compose, Lambdad, RandSpatialCropd, Resized
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_and_extract_archive

from flextrain.callbacks.epoch_summary import CallbackLogMetrics

# from flextrain.callbacks.autoencoder import unnorm_fn
from flextrain.callbacks.gan import CallbackGanRecon
from flextrain.callbacks.gan import unnorm_m1p1_255_fn as unnorm_fn
from flextrain.callbacks.record_lr_mom import CallbackRecordLearninRateMomentum
from flextrain.callbacks.samples import CallbackRecordSamples
from flextrain.callbacks.skip_epochs import CallbackSkipEpoch
from flextrain.datasets.dataset_cyclegan import dataset_facades
from flextrain.gan.gan_cycle import GanCycle
from flextrain.trainer.options import Options
from flextrain.trainer.start_training import start_training

options = Options()
options.training.nb_epochs = 200
options.training.precision = 32
options.training.devices = '0'
# options.training.pretraining = (
#    '/mnt/hdd/data_tmp/experiments/gan/02_cyclegan/lightning_logs/version_0/checkpoints/epoch=199-step=240000.ckpt'
# )

batch_size = 1
input_shape = (3, 256, 256)
paired = False
misalignment = True

size_x = int(input_shape[1] * 1.12)


if misalignment:
    transform_train = Compose(
        [
            Resized(keys=('A', 'B'), mode='bicubic', spatial_size=(size_x, size_x)),
            # independent cropping
            RandSpatialCropd(keys=('A'), roi_size=(input_shape[1], input_shape[2])),
            RandSpatialCropd(keys=('B'), roi_size=(input_shape[1], input_shape[2])),
        ]
    )
else:
    transform_train = Compose(
        [
            Resized(keys=('A', 'B'), mode='bicubic', spatial_size=(size_x, size_x)),
            # joint cropping
            RandSpatialCropd(keys=('A', 'B'), roi_size=(input_shape[1], input_shape[2])),
        ]
    )


datasets = dataset_facades(
    batch_size=batch_size,
    normalize_minus1_1=True,
    normalize_0_1=False,
    paired=paired,
    transform_train=transform_train,
)

from flextrain.models.cyclegan import GeneratorResNet, Discriminator, weights_init_normal

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, 9)
G_BA = GeneratorResNet(input_shape, 9)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

model_pl = GanCycle(
    generator_AB=G_AB,
    generator_BA=G_BA,
    discriminator_A=D_A,
    discriminator_B=D_B,
    image_name_A='A',
    image_name_B='B',
)

callbacks = [
    CallbackLogMetrics(),
    CallbackRecordSamples(nb_repeat=3, nb_samples=3),
    CallbackRecordLearninRateMomentum(),
    CallbackSkipEpoch(
        [
            CallbackGanRecon(
                input_name='A',
                save_numpy=False,
                model_sample_g_fn=model_pl.sample_A,
                generator_conditional_feature_names=('B',),
                conditioning_image_names=('A', 'B'),
                unnorm_cond_fn=unnorm_fn,
                unnorm_output_fn=unnorm_fn,
                unnorm_truth_fn=unnorm_fn,
            ),
        ],
        nb_epochs=5,
        include_epoch_zero=True,
    ),
]

start_training(options, datasets, callbacks, model_pl)
