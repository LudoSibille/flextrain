import os

import torch
from layers import make_decoder, make_encoder
from monai.transforms import Lambdad
from torch import nn

from flextrain.autoencoder.ae import AutoEncoder
from flextrain.autoencoder.lightning import AutoencoderLightning
from flextrain.callbacks.autoencoder import CallbackAutoenderRecon
from flextrain.callbacks.epoch_summary import CallbackLogMetrics
from flextrain.callbacks.skip_epochs import CallbackSkipEpoch
from flextrain.datasets.mnist import mnist_dataset
from flextrain.layers.utils import ModelBatchAdaptor
from flextrain.losses import LossBceLogitsSigmoid
from flextrain.trainer.options import Options
from flextrain.trainer.start_training import start_training
from flextrain.trainer.utils import default

encoder = make_encoder(1, [8, 8, 16, 16, 16, 24, 8], pool=[1, 3, 4])
decoder = make_decoder(1, [8, 16, 16, 8, 8], unpool=[0, 1, 2])

options = Options()
options.training.nb_epochs = 100
options.training.precision = 16
options.training.devices = '0'

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


ddpm_pl = AutoencoderLightning(
    AutoEncoder(encoder=ModelBatchAdaptor(encoder, ['images']), decoder=decoder),
    loss_fn=LossBceLogitsSigmoid(batch_key='images'),
)

callbacks = [
    CallbackLogMetrics(),
    CallbackSkipEpoch(
        [CallbackAutoenderRecon(input_name='images', save_numpy=False)], nb_epochs=10, include_epoch_zero=True
    ),
]

root_output = default('OUTPUT_ARTEFACT_ROOT', default_value='/tmp/')
os.makedirs(root_output, exist_ok=True)
start_training(options, datasets, callbacks, ddpm_pl)
