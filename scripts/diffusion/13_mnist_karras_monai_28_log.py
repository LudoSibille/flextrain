import logging
from functools import partial

import numpy as np
import torch
import torchvision
from monai import transforms
from utils import TransformApplyTo, batch_images_adapator_0_1

from flextrain.callbacks.epoch_summary import CallbackLogMetrics
from flextrain.callbacks.sample_diffusion import CallbackSample2dDiffusionModel
from flextrain.callbacks.skip_epochs import CallbackSkipEpoch
from flextrain.datasets.mnist import mnist_dataset
from flextrain.diffusion.karras import (
    KarrasDiffusion,
    rand_cosine_interpolated,
    rand_log_normal,
    rand_uniform,
    sample,
    sample_euler,
    sigmas_karras,
)
from flextrain.diffusion.lightning import GaussianDiffusionLightning
from flextrain.layers.dummy import UNet
from flextrain.metrics.fid_mnist import create_fid_mnist
from flextrain.trainer.options import Options
from flextrain.trainer.start_training import start_training

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

    options = Options()
    options.training.nb_epochs = 300
    options.training.precision = 16
    options.training.devices = '0'
    options.training.gradient_clip_val = 1.0
    options.training.check_val_every_n_epoch = 1
    options.workflow.enable_progress_bar = True
    # options.training.pretraining = '/home/ludovic/work/data/data_tmp/diffusion/BACKUP_13_mnist_karras_monai_28_log_vs/lightning_logs/version_0/checkpoints/epoch=135-step=127432.ckpt'

    image_size = 28
    batch_size = 64

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

    from flextrain.diffusion.models.kdiff_unet_v1 import ImageDenoiserModelV1

    model = ImageDenoiserModelV1(
        1,
        256,
        (1, 1, 1),
        (32, 32, 64),
        (False, False, True),
        None,
        patch_size=1,
        dropout_rate=0.05,
        mapping_cond_dim=0,
        unet_cond_dim=0,
        cross_cond_dim=0,
        skip_stages=0,
        has_variance=False,
    )

    # params = torch.load('/home/ludovic/work/projects/github_diffusions/k-diffusion/MNIST_00320000.pth', map_location='cpu')
    # model.load_state_dict(params['model'])

    sigma_data = 0.6162
    ddpm = KarrasDiffusion(
        model,
        sigma_data=sigma_data,
        # sigmas_fn=partial(sigmas_karras, sigma_min=1e-2, sigma_max=80, nb_steps=200),
        sigmas_fn=partial(sigmas_karras, sigma_min=1e-1, sigma_max=80, nb_steps=200),
        # sigmas_train_sampling_fn=rand_log_normal,
        sigmas_train_sampling_fn=partial(rand_uniform, min_value=1e-2, max_value=1.0),
        # sigmas_train_sampling_fn=partial(rand_cosine_interpolated, image_d=28, noise_d_low=32, noise_d_high=28, sigma_data=sigma_data),
        sample_fn=partial(sample, sampler=sample_euler),
    )

    ddpm_pl = GaussianDiffusionLightning(
        ddpm,
        input_name='images',
        optimizer_fn=partial(torch.optim.AdamW, lr=2e-4, eps=1e-6, betas=(0.95, 0.999), weight_decay=1e-3),
        scheduler_steps_fn=None,
        # scheduler_steps_fn=partial(scheduler_one_cycle_cosine_fn, max_lr=1e-3, total_steps=len(datasets['mnist']['train'].dataset) // batch_size * options.training.nb_epochs)
        # scheduler_steps_fn=partial(scheduler_steps_fn, nb_steps=4)
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
