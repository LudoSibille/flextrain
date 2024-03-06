from functools import partial
import os
import matplotlib.pyplot as plt
import numpy as np
from diffusion_models.diffusion.discrete_beta_schedulers import *
from diffusion_models.datasets.utils import get_data_root

export_path = get_data_root('/tmp/experiments')
os.makedirs(export_path, exist_ok=True)

nb_steps = 1000
t = np.arange(nb_steps)
schedulers = {
    #'Linear 1000': linear(steps=nb_steps),
    #'Linear 1000_1e-6, 2e-2': linear(steps=nb_steps, β_start=1e-6, β_end=2e-2),
    #'Linear 10000 / 10': linear(steps=nb_steps * 10),
    #'Linear_zero_snr 1000': linear_zero_snr(steps=nb_steps),
    'Cosine 1000': cosine_beta_schedule(steps=nb_steps),
    'Cosine 10000': cosine_beta_schedule(steps=nb_steps * 10),
    #'Sigmoid 1000': sigmoid_beta_schedule(steps=nb_steps),
    #'Sigmoid 10000': sigmoid_beta_schedule(steps=nb_steps * 10),
    #'scaled_linear^2': scaled_linear_beta(steps=nb_steps),
    #'scaled_linear^5': scaled_linear_beta(steps=nb_steps, power=5, β_start=1e-6, β_end=4e-2),
}


plt.close()
for name, values in schedulers.items():
    ᾱ = values.ᾱ
    if len(ᾱ) != nb_steps:
        ᾱ = ᾱ[::10]
    plt.plot(t, ᾱ.numpy(), label=name)    
plt.title('Scheduler ᾱ')
plt.legend()
plt.savefig(os.path.join(export_path, 'scheduler_ᾱ_epoch_invariance.png'))
plt.show()


plt.close()
for name, values in schedulers.items():
    if len(values.α) != nb_steps:
        β = 1 - values.α[::10]
    else:
        β = 1 - values.α
    plt.plot(t, β, label=name)    
plt.title('Scheduler β')
plt.legend()
plt.savefig(os.path.join(export_path, 'scheduler_β_epoch_invariance.png'))
plt.show()


from diffusion_models.datasets.mnist import mnist_dataset
from diffusion_models.diffusion.discrete_ddpm_simple import SimpleGaussianDiffusion
import numpy as np
from PIL import Image
batch = next(iter(mnist_dataset(batch_size=1)['mnist']['valid']))
image = batch['images']
scheduler_steps = 10


# Min/Max value of the dataset
min_value = 0
max_value = 1
for name, values in schedulers.items():
    steps_delta = len(values.α) // scheduler_steps
    ddpm = SimpleGaussianDiffusion(model=None, noise_scheduler_fn=lambda: values)

    images_noised = []
    images_noised.append(image)
    for t in range(0, len(values.α), steps_delta):
        image_noised, _ = ddpm.forward_diffusion(image, torch.asarray([t]))
        images_noised.append(image_noised)
    images_noised = torch.concat(images_noised, dim=3)
    images_noised = 255 * (images_noised - min_value) / (max_value - min_value)
    images_noised = torch.clamp(images_noised, 0, 255).squeeze().numpy().astype(np.uint8)
    Image.fromarray(images_noised).save(os.path.join(export_path, f'noising_{name}.png'))
    
print('DONE!')