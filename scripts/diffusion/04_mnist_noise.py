import os

import torch
from generative.networks.nets import DiffusionModelUNet
from utils import to_image

from flextrain.diffusion.discrete_ddpm_simple import SimpleGaussianDiffusion
from flextrain.diffusion.discrete_sampling import sample_ddpm, sample_ddpm_kg
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
x_T = torch.randn([100, 1, 28, 28], dtype=torch.float32, device=device)
data = sample_ddpm(ddpm.α, ddpm.ᾱ, ddpm.σ, device=device, model=model, batch_shape=x_T.shape, x_T=x_T)
to_image(data, os.path.join(output_path, 'samples.png'))
print('Sampling DONE!')


print('Sampling ILVR')
lowpass_fn = lambda t: t + torch.randn_like(t)
data_noisy = lowpass_fn(data)
to_image(data_noisy, os.path.join(output_path, 'samples_noisy.png'))


data = torch.clamp(data, -1, 1)
# stop_t = 0  # less "generation"
stop_t = 100  # "beautify" the data with more generation
for i in range(10):
    x_T = torch.randn([100, 1, 28, 28], dtype=torch.float32, device=device)
    σ_d = torch.full([x_T.shape[0]], fill_value=0.75, dtype=torch.float32, device=device)
    data_clean = sample_ddpm_kg(
        ddpm.α,
        ddpm.ᾱ,
        ddpm.σ,
        σ_d=σ_d,
        device=device,
        model=model,
        batch_shape=x_T.shape,
        x_T=x_T,
        noisy=data_noisy,
        stop_t=stop_t,
    )
    to_image(data_clean, os.path.join(output_path, f'clean_samples_{i}.png'))
    print(f'ILVR DONE_{i}')
