import os
from torch import nn
import torch

from ..datasets.mnist import mnist_dataset
from ..classification.model_mnist import create_model
from .fid import FID
from ..trainer.utils import default
from ..classification.lightning import ClassifierLightning


def mnist_image_adapator(dataloader):
    for batch in dataloader:
        yield batch['images']


def create_fid_mnist(default_model_root='/tmp/', batch_size: int = 1000, force_create: bool = False, target_nb_samples: int = 4000, del_layers: int = 1, remove_prefix: str='base_model.') -> nn.Module:
    """
    Helper to create a FID metric

    force_create: force the creation of the FID features
    """
    model = create_model()
    root_output = default('OUTPUT_ARTEFACT_ROOT', default_value=default_model_root)
    pretrained_path = os.path.join(root_output, 'mnist_28_classifier.pth')
    fid_path = os.path.join(root_output, 'mnist_28_fid.pth')
    if not force_create and os.path.exists(fid_path):
        with open(fid_path, 'rb') as f:
            fid = torch.load(f)
            return fid
        
    with open(os.path.join(pretrained_path), 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device('cpu'))
    
    if len(remove_prefix) > 0:
        state_dict = {n.replace(remove_prefix, ''): v for n, v in state_dict.items()}
    model.load_state_dict(state_dict)
    for _ in range(del_layers):
        del model[-1]  # remove the classification layer
    model_pl = ClassifierLightning(model, 'images', 'targets')
    
    datasets = mnist_dataset(
        batch_size=batch_size, 
        max_train_samples=None,
        num_workers=4,
        shuffle_valid=False
    )

    fid = FID(model=model_pl)
    fid.fit(mnist_image_adapator(datasets['mnist']['valid']), target_nb_samples=target_nb_samples)  # use never seen data (valid)

    # make sure we cache the calculated features
    with open(fid_path, 'wb') as f:
        torch.save(fid, f)
    
    return fid


def create_mnist_pretrained(default_model_root: str = '/tmp/') -> nn.Module:
    model = create_model()
    root_output = default('OUTPUT_ARTEFACT_ROOT', default_value=default_model_root)
    pretrained_path = os.path.join(root_output, 'mnist_28_classifier.pth')        
    with open(os.path.join(pretrained_path), 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device('cpu'))
        
    state_dict = {name.replace('base_model.', ''): value for name, value in state_dict.items()} 
    model.load_state_dict(state_dict)
    return model
