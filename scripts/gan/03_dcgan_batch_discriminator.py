import torch
from monai.transforms import Lambdad
from torch import nn

from flextrain.callbacks.epoch_summary import CallbackLogMetrics
from flextrain.callbacks.gan import CallbackGanRecon
from flextrain.callbacks.record_lr_mom import CallbackRecordLearninRateMomentum
from flextrain.callbacks.skip_epochs import CallbackSkipEpoch
from flextrain.datasets.mnist import mnist_dataset
from flextrain.gan.gan_dc import GanDC
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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, img_size: int, latent_dim: int, channels: int):
        super().__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# The original paper by Salimans et. al. discusses only 1D minibatch discrimination
class MinibatchDiscrimination1d(nn.Module):
    r"""1D Minibatch Discrimination Module as proposed in the paper `"Improved Techniques for
    Training GANs by Salimans et. al." <https://arxiv.org/abs/1805.08318>`_

    Allows the Discriminator to easily detect mode collapse by augmenting the activations to the succeeding
    layer with side information that allows it to determine the 'closeness' of the minibatch examples
    with each other

    .. math :: M_i = T * f(x_{i})
    .. math :: c_b(x_{i}, x_{j}) = \exp(-||M_{i, b} - M_{j, b}||_1) \in \mathbb{R}.
    .. math :: o(x_{i})_b &= \sum_{j=1}^{n} c_b(x_{i},x_{j}) \in \mathbb{R} \\
    .. math :: o(x_{i}) &= \Big[ o(x_{i})_1, o(x_{i})_2, \dots, o(x_{i})_B \Big] \in \mathbb{R}^B \\
    .. math :: o(X) \in \mathbb{R}^{n \times B}

    This is followed by concatenating :math:`o(x_{i})` and :math:`f(x_{i})`

    where

    - :math:`f(x_{i}) \in \mathbb{R}^A` : Activations from an intermediate layer
    - :math:`f(x_{i}) \in \mathbb{R}^A` : Parameter Tensor for generating minibatch discrimination matrix


    Args:
        in_features (int): Features input corresponding to dimension :math:`A`
        out_features (int): Number of output features that are to be concatenated corresponding to dimension :math:`B`
        intermediate_features (int): Intermediate number of features corresponding to dimension :math:`C`

    Returns:
        A Tensor of size :math:`(N, in_features + out_features)` where :math:`N` is the batch size
    """

    def __init__(self, in_features, out_features, intermediate_features=16):
        super(MinibatchDiscrimination1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.intermediate_features = intermediate_features

        self.T = nn.Parameter(torch.Tensor(in_features, out_features, intermediate_features))
        nn.init.normal_(self.T)

    def forward(self, x):
        r"""Computes the output of the Minibatch Discrimination Layer

        Args:
            x (torch.Tensor): A Torch Tensor of dimensions :math: `(N, infeatures)`

        Returns:
            3D Torch Tensor of size :math: `(N,infeatures + outfeatures)` after applying Minibatch Discrimination
        """
        assert len(x.shape) == 2
        assert x.shape[1] == self.in_features

        M = torch.mm(x, self.T.view(self.in_features, -1))
        M = M.view(-1, self.out_features, self.intermediate_features).unsqueeze(0)
        M_t = M.permute(1, 0, 2, 3)
        out = torch.sum(torch.exp(-(torch.abs(M - M_t).sum(3))), dim=0) - 1
        return torch.cat([x, out], 1)


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int,
        channels: int,
    ):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True, kernel_size=3, padding=1):
            # block = [
            #    nn.Conv2d(
            #        in_filters, out_filters, kernel_size=kernel_size, padding=padding, stride=1, padding_mode='reflect'
            #    ),
            #    nn.LeakyReLU(0.2, inplace=True),
            #    nn.Dropout2d(0.25),
            #    nn.MaxPool2d(2),
            # ]
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        f = 128
        ds_size = img_size // 2**4
        features = f * ds_size**2
        features_md_out = 8
        self.mini_batch_discriminator = MinibatchDiscrimination1d(features, features_md_out)
        self.adv_layer = nn.Sequential(nn.Linear(features_md_out + features, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        out_d = self.mini_batch_discriminator(out)
        validity = self.adv_layer(out_d)
        return validity


generator = Generator(img_size=32, latent_dim=100, channels=1)
generator.apply(weights_init_normal)
discriminator = Discriminator(img_size=32, channels=1)
discriminator.apply(weights_init_normal)
model_pl = GanDC(generator=generator, discriminator=discriminator, image_name='images', label_smoothing=0.2)

callbacks = [
    CallbackLogMetrics(),
    CallbackRecordLearninRateMomentum(),
    CallbackSkipEpoch(
        [
            CallbackGanRecon(input_name='images', save_numpy=False, nb_samples=128),
        ],
        nb_epochs=5,
        include_epoch_zero=True,
    ),
]

start_training(options, datasets, callbacks, model_pl)
