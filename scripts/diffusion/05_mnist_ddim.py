import os

import torch
from generative.networks.nets import DiffusionModelUNet
from utils import to_image

from flextrain.diffusion.discrete_ddpm_simple import SimpleGaussianDiffusion
from flextrain.diffusion.discrete_sampling import sample_ddim, sample_ddpm
from flextrain.diffusion.lightning import GaussianDiffusionLightning

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
eta = 0.0  # deterministic sampling
x_T = torch.randn([100, 1, 28, 28], dtype=torch.float32, device=device)
data = sample_ddim(ddpm.α, ddpm.ᾱ, ddpm.σ, device=device, model=model, batch_shape=x_T.shape, x_T=x_T, nb_steps=150, eta=eta)
to_image(data, os.path.join(output_path, 'samples_ddim_150_eta_0.png'))

data = sample_ddim(ddpm.α, ddpm.ᾱ, ddpm.σ, device=device, model=model, batch_shape=x_T.shape, x_T=x_T, nb_steps=250, eta=eta)
to_image(data, os.path.join(output_path, 'samples_ddim_250_eta_0.png'))

data = sample_ddim(ddpm.α, ddpm.ᾱ, ddpm.σ, device=device, model=model, batch_shape=x_T.shape, x_T=x_T, nb_steps=50, eta=eta)
to_image(data, os.path.join(output_path, 'samples_ddim_50_eta_0.png'))

data = sample_ddpm(ddpm.α, ddpm.ᾱ, ddpm.σ, device=device, model=model, batch_shape=x_T.shape, x_T=x_T)
to_image(data, os.path.join(output_path, 'samples_ddpm.png'))
print('Sampling DONE!')
