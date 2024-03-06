import torch
from generative.networks.nets import DiffusionModelUNet
from monai import transforms

from flextrain.callbacks.epoch_summary import CallbackLogMetrics
from flextrain.callbacks.sample_diffusion import CallbackSample2dDiffusionModel
from flextrain.callbacks.skip_epochs import CallbackSkipEpoch
from flextrain.datasets.mnist import mnist_dataset
from flextrain.diffusion.discrete_ddpm_simple import SimpleGaussianDiffusion
from flextrain.diffusion.lightning import GaussianDiffusionLightning
from flextrain.layers.dummy import UNet
from flextrain.metrics.fid_mnist import create_fid_mnist
from flextrain.trainer.options import Options
from flextrain.trainer.start_training import start_training


def unnorm_fn(images: torch.Tensor) -> torch.Tensor:
    images = images.clamp(-0.5, 0.5) + 0.5
    images = (images * 255).type(torch.uint8)
    return images


def unnorm_fid_fn(images: torch.Tensor) -> torch.Tensor:
    images = images.clamp(-0.5, 0.5) + 0.5
    return images


def batch_images_adapator_0_1_05(dataloader):
    for batch in dataloader:
        yield batch['images'] + 0.5


if __name__ == '__main__':
    options = Options()
    options.training.nb_epochs = 501
    options.training.precision = 16
    options.training.devices = '0'
    options.workflow.enable_progress_bar = False
    image_size = 28
    batch_size = 1000

    transform_train = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(keys=["images"], a_min=0.0, a_max=1.0, b_min=-0.5, b_max=0.5, clip=True),
        ]
    )

    datasets = mnist_dataset(
        batch_size=batch_size,
        transform_train=transform_train,
        transform_valid=transform_train,
        max_train_samples=None,
        shuffle_valid=True,  # show more samples for better comparison & FID real
    )

    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 64),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=64,
    )
    ddpm = SimpleGaussianDiffusion(model)
    ddpm_pl = GaussianDiffusionLightning(
        ddpm,
        input_name='images',
    )

    fid = create_fid_mnist()
    fid_real = [fid(batch_images_adapator_0_1_05(datasets['mnist']['valid'])) for i in range(10)]
    print('FID REAL data mean=', float(torch.asarray(fid_real).mean()), 'FID STD=', float(torch.asarray(fid_real).std()))

    callbacks = [
        CallbackLogMetrics(),
        CallbackSkipEpoch(
            [
                CallbackSample2dDiffusionModel(
                    sample_kwargs={
                        'batch_shape': (batch_size, 1, image_size, image_size),
                    },
                    input_name='images',
                    nb_samples=4000,
                    unnorm_fn=unnorm_fn,
                    unnorm_fid_fn=unnorm_fid_fn,
                    fid=fid,
                )
            ],
            nb_epochs=25,
            include_epoch_zero=True,
        ),
    ]
    start_training(options, datasets, callbacks, ddpm_pl)
    print('Training done!')
