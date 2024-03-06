import os

import numpy as np
import torch
import torchvision
from generative.networks.nets import DiffusionModelUNet
from PIL import Image
from torchvision.transforms import Resize
from torchvision.utils import make_grid
from utils import to_image

from flextrain.diffusion.discrete_ddpm_simple import SimpleGaussianDiffusion
from flextrain.diffusion.discrete_sampling import sample_ddpm, sample_ddpm_ilvr
from flextrain.diffusion.lightning import GaussianDiffusionLightning
from flextrain.types import TorchTensorNX


def lowpass_updown_fn(images: TorchTensorNX, factor=4) -> TorchTensorNX:
    assert len(images.shape) == 4, 'NCHW'
    target_w = 28 // factor
    downsampler = Resize((target_w, target_w), antialias=True)
    upsampler = Resize(images.shape[2:], antialias=False)
    return upsampler(downsampler(images))


def lowpass_gaussian_fn(images: TorchTensorNX, kernel_size=9) -> TorchTensorNX:
    assert len(images.shape) == 4, 'NCHW'
    return torchvision.transforms.functional.gaussian_blur(images, kernel_size=kernel_size)


def lowpass_id_fn(images: TorchTensorNX, factor=4) -> TorchTensorNX:
    return images / 5


checkpoint_path = '/home/ludovic/work/data/data_tmp/dm/01b_mnist_simplest_monai_28_longer_vs/lightning_logs/version_0/checkpoints/epoch=459-step=27600.ckpt'
device = torch.device('cuda:0')
output_path = '/home/ludovic/work/data/data_tmp/dm/tmp/'

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
).to(device)
with open(checkpoint_path, 'rb') as f:
    data = torch.load(f, map_location=device)
ddpm_pl.load_state_dict(data['state_dict'], strict=True)

print('Sampling...')
torch.random.manual_seed(0)
x_T = torch.randn([100, 1, 28, 28], dtype=torch.float32, device=device)
data = sample_ddpm(ddpm.α, ddpm.ᾱ, ddpm.σ, device=device, model=model, batch_shape=x_T.shape, x_T=x_T)
to_image(data, os.path.join(output_path, 'samples.png'))
print('Sampling DONE!')


print('Sampling ILVR')
# lowpass_fn = lowpass_gaussian_fn
lowpass_fn = lowpass_updown_fn
data_low_orig = lowpass_fn(data)
to_image(data_low_orig, os.path.join(output_path, 'ilvr_samples_lowpass.png'))

data = torch.clamp(data, -1, 1)

# stop_t = 100
stop_t = 200
# stop_t = 300
for i in range(10):
    x_T = torch.randn([100, 1, 28, 28], dtype=torch.float32, device=device)
    data_ilvr = sample_ddpm_ilvr(
        ddpm.α,
        ddpm.ᾱ,
        ddpm.σ,
        device=device,
        model=model,
        batch_shape=x_T.shape,
        x_T=x_T,
        y=data,
        lowpass_filter_fn=lowpass_fn,
        stop_t=stop_t,
    )
    to_image(data_ilvr, os.path.join(output_path, f'ilvr_samples_{i}.png'))

    data_ilvr_error = (lowpass_fn(data_ilvr) - lowpass_fn(data)).abs()
    to_image(data_ilvr_error, os.path.join(output_path, f'error_ilvr_samples_{i}.png'))
    print(f'ILVR DONE_{i}')
