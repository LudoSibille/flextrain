import torch
from monai.transforms import Lambdad
from torch import nn

from flextrain.callbacks.epoch_summary import CallbackLogMetrics
from flextrain.callbacks.gan import CallbackGanRecon
from flextrain.callbacks.samples import CallbackRecordSamples
from flextrain.callbacks.skip_epochs import CallbackSkipEpoch
from flextrain.datasets.mnist import mnist_dataset
from flextrain.gan.gan_dc import GanDC
from flextrain.models.cyclegan import (
    Discriminator,
    GeneratorResNet,
    weights_init_normal,
)
from flextrain.trainer.options import Options
from flextrain.trainer.start_training import start_training

options = Options()
options.training.nb_epochs = 100
options.training.precision = 32
options.training.devices = '0'

batch_size = 64
datasets = mnist_dataset(
    batch_size=batch_size,
    max_train_samples=None,
    num_workers=4,
    shuffle_valid=False,
    # pad to power of 2 to make our life simpler for the upsampling
    # range = (-1, 1)
    transform_train=Lambdad(
        keys='images', func=lambda x: torch.nn.functional.pad(torch.from_numpy(x), (2, 2, 2, 2)) * 2 - 1
    ),
    transform_valid=Lambdad(
        keys='images', func=lambda x: torch.nn.functional.pad(torch.from_numpy(x), (2, 2, 2, 2)) * 2 - 1
    ),
)


generator = GeneratorResNet(input_shape=(1, 32, 32), num_residual_blocks=6)
generator.apply(weights_init_normal)
discriminator = Discriminator(input_shape=(1, 32, 32))
discriminator.apply(weights_init_normal)

adversarial_loss = torch.nn.MSELoss()
z_sampler = lambda batch_size: torch.randn((batch_size, 1, 32, 32), dtype=torch.float32)
model_pl = GanDC(
    generator=generator,
    adversarial_loss=adversarial_loss,
    discriminator=discriminator,
    image_name='images',
    z_sampler=z_sampler,
)

callbacks = [
    CallbackLogMetrics(),
    CallbackRecordSamples(nb_repeat=1, nb_samples=20),
    CallbackSkipEpoch(
        [
            CallbackGanRecon(input_name='images', save_numpy=False),
        ],
        nb_epochs=5,
        include_epoch_zero=True,
    ),
]

start_training(options, datasets, callbacks, model_pl)
