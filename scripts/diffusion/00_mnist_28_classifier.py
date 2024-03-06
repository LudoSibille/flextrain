"""
Train a classifier on MNist, 28x28 original dataset.

The purpose of this classifier is to be used with the
`Frechet Inception Score` (FID) metric. Here we trained
specifically on MNIST rather than relying on a generic
classifier trained on large images (224x224) where
resizing will introduce visual artifacts.
"""

import os
from functools import partial

import torch

from flextrain.callbacks.epoch_summary import CallbackLogMetrics
from flextrain.classification.lightning import ClassifierLightning
from flextrain.classification.model_mnist import create_model
from flextrain.datasets.mnist import mnist_dataset
from flextrain.layers.utils import ModelBatchAdaptor
from flextrain.losses import LossCrossEntropy
from flextrain.metrics.fid_mnist import create_fid_mnist
from flextrain.trainer.options import Options
from flextrain.trainer.start_training import start_training
from flextrain.trainer.utils import default

options = Options()
options.training.nb_epochs = 101
options.training.precision = 16
options.training.devices = '0'

batch_size = 10000
datasets = mnist_dataset(batch_size=batch_size, max_train_samples=None, num_workers=10, shuffle_valid=False)

model = create_model()
model = ModelBatchAdaptor(model, ['images'])
model = torch.compile(model)
ddpm_pl = ClassifierLightning(
    model,
    loss_fn=LossCrossEntropy(target_name='targets'),
    optimizer_fn=partial(torch.optim.SGD, lr=0.1, nesterov=True, weight_decay=5e-5, momentum=0.99),
)

callbacks = [
    CallbackLogMetrics(),
]
start_training(options, datasets, callbacks, ddpm_pl)

root_output = default('OUTPUT_ARTEFACT_ROOT', default_value='/tmp/')

os.makedirs(root_output, exist_ok=True)
with open(os.path.join(root_output, 'mnist_28_classifier.pth'), 'wb') as f:
    # save the weights of the original model
    # https://discuss.pytorch.org/t/how-to-save-load-a-model-with-torch-compile/179739
    torch.save(model._orig_mod.state_dict(), f)

fid = create_fid_mnist(default_model_root=root_output, force_create=True)
print('Training done!')
