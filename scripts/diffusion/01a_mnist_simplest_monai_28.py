import torch
from generative.networks.nets import DiffusionModelUNet
from monai import transforms
from utils import batch_images_adapator_0_1

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

if __name__ == '__main__':
    options = Options()
    options.training.nb_epochs = 101
    options.training.precision = 16
    options.training.devices = '0'
    # options.training.check_val_every_n_epoch = 5
    options.workflow.enable_progress_bar = False

    image_size = 28
    batch_size = 1000

    transform_train = transforms.Compose(
        [
            # transforms.RandAffined(
            #    keys=["images"],
            #    rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
            #    translate_range=[(-1, 1), (-1, 1)],
            #    scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
            #    spatial_size=[28, 28],
            #    padding_mode="zeros",
            #    prob=0.5,
            # ),
            # avoid padding issues by scaling as the last step
            transforms.ScaleIntensityRanged(keys=["images"], a_min=0.0, a_max=1.0, b_min=-1.0, b_max=1.0, clip=True),
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

    # calculate the FID for real data. That will give us the target range and variation of FID
    # FID ~0.3
    fid_real = [fid(batch_images_adapator_0_1(datasets['mnist']['valid'])) for i in range(10)]
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
                    nb_samples=1000,
                    fid=fid,
                )
            ],
            nb_epochs=options.training.check_val_every_n_epoch * 10,
            include_epoch_zero=True,
        ),
    ]
    start_training(options, datasets, callbacks, ddpm_pl)
    # FID ~1.5
    print('Training done!')
