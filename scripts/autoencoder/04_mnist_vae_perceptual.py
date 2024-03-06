import os
from functools import partial

import torch
from layers import make_decoder, make_encoder
from monai.transforms import Lambdad
from torch import nn

from flextrain.autoencoder.lightning import AutoencoderLightning
from flextrain.autoencoder.vae import AutoencoderConvolutionalVariational
from flextrain.callbacks.autoencoder import CallbackAutoenderRecon
from flextrain.callbacks.epoch_summary import CallbackLogMetrics
from flextrain.callbacks.skip_epochs import CallbackSkipEpoch
from flextrain.datasets.mnist import mnist_dataset
from flextrain.layers.utils import ModelBatchAdaptor
from flextrain.losses import (
    LossBceLogitsSigmoid,
    LossCombine,
    LossL1,
    LossPerceptual,
    LossSupervised,
)
from flextrain.metrics.fid_mnist import create_mnist_pretrained
from flextrain.trainer.options import Options
from flextrain.trainer.start_training import start_training
from flextrain.trainer.utils import default

encoder = make_encoder(1, [8, 8, 16, 16, 16, 24, 8], pool=[1, 3, 4])
# decoder = make_decoder(1, [8, 16, 16, 8, 8], unpool=[0, 1, 2])
decoder = nn.Sequential(make_decoder(1, [8, 32, 64, 128, 8], unpool=[0, 1, 2]), nn.Sigmoid())

options = Options()
options.training.nb_epochs = 128
options.training.precision = 32
options.training.devices = '0'
# options.training.pretraining = '/tmp/experiments/autoencoder/04_mnist_vae_perceptual_vs_good/lightning_logs/version_0/checkpoints/epoch=99-step=6000.ckpt'

batch_size = 1000
datasets = mnist_dataset(
    batch_size=batch_size,
    max_train_samples=None,
    num_workers=4,
    shuffle_valid=False,
    # pad to power of 2 to make our life simpler for the upsampling
    transform_train=Lambdad(keys='images', func=lambda x: torch.nn.functional.pad(torch.from_numpy(x), (2, 2, 2, 2))),
    transform_valid=Lambdad(keys='images', func=lambda x: torch.nn.functional.pad(torch.from_numpy(x), (2, 2, 2, 2))),
)

batch = next(iter(datasets['mnist']['train']))

ddpm_pl = AutoencoderLightning(
    AutoencoderConvolutionalVariational(
        batch, z_size=128, cnn_dim=4, encoder=ModelBatchAdaptor(encoder, ['images']), decoder=decoder
    ),
    # loss_fn=partial(AutoencoderConvolutionalVariational.loss_function, x_input_name='images', recon_loss_name='L1'),
    # loss_fn=partial(AutoencoderConvolutionalVariational.loss_function_v2, recon_loss_fn=LossBceLogitsSigmoid(batch_key='images'), kullback_leibler_weight=0.1),
    loss_fn=partial(
        AutoencoderConvolutionalVariational.loss_function_v2,
        recon_loss_fn=LossL1(batch_key='images'),
        kullback_leibler_weight=0.1,
    ),
    optimizer_fn=partial(torch.optim.Adam, lr=1e-3),
)

callbacks = [
    CallbackLogMetrics(),
    CallbackSkipEpoch(
        [
            CallbackAutoenderRecon(
                input_name='images',
                save_numpy=True,
                #            unnorm_output_fn=nn.Sigmoid(),
                unnorm_output_fn=None,
                unnorm_truth_fn=None,
            )
        ],
        nb_epochs=10,
        include_epoch_zero=True,
    ),
]
start_training(options, datasets, callbacks, ddpm_pl)


import numpy as np
from torchvision.utils import make_grid

samples = ddpm_pl.ae.sample(batch_size)
# samples = nn.Sigmoid()(samples)
np.save(
    os.path.join(options.data.root_current_experiment, 'random_samples.npy'),
    make_grid(samples, nrow=int(np.sqrt(batch_size))).numpy(),
)
