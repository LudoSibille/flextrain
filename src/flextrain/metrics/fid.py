from torch import nn
import torch
from ..types import Batch
from..trainer.utils import transfer_batch_to_device, len_batch
from typing import Optional, Sequence
from .sqrtm import sqrtm_newton_schulz
from scipy.linalg import sqrtm
import warnings


class FID(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        """
        The model used to extract features
        """
        super().__init__()
        self.model = model
        self.reference_features = None
    
    def fit(self, dataloader: Sequence[Batch], device: torch.device = torch.device('cpu'), target_nb_samples: int = 1000) -> None:
        """
        Pre-calculate the features on the true samples
        """
        self.reference_features = self.calculate_features(dataloader=dataloader, device=device, target_nb_samples=target_nb_samples)

    def calculate_features(self, dataloader: Sequence[Batch], device: torch.device, target_nb_samples: Optional[int] = None):
        total_samples = 0
        self.model = self.model.to(device)
        self.model.eval()
        all_features = []
        for batch in dataloader:
            batch = transfer_batch_to_device(batch, device)
            nb_samples = len_batch(batch)
            with torch.no_grad():  # no need for gradient
                features = self.model(batch)

            all_features.append(features.detach().cpu())
            total_samples += nb_samples
            if target_nb_samples is not None and total_samples >= target_nb_samples:
                # we have enough samples to calculate the statistics
                break

        features = torch.cat(all_features, dim=0)
        if target_nb_samples is not None:
            features = features[:target_nb_samples]
        return features
    
    def calculate_fid_from_features(
            self, 
            current_features: torch.Tensor, 
            device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Calculate FID for pre-computed model features against reference features
        """
        assert current_features.shape[1:] == self.reference_features.shape[1:], \
            f'Feature shape is different. Got={current_features.shape}, expected={self.reference_features.shape}'
        
        if len(current_features) != len(self.reference_features):
            warnings.warn('FID are only comparable when the same number of samples in the reference and current are the same!')

        current_features = current_features.view([current_features.shape[0], -1]).to(device)
        m1 = current_features.mean(0)
        c1 = current_features.T.cov()

        reference_features = self.reference_features.view([self.reference_features.shape[0], -1]).to(device)
        m2 = reference_features.mean(0)
        c2 = reference_features.T.cov()

        # TODO REVISIT THIS
        #csr = sqrtm_newton_schulz(c1 @ c2)
        #print('ERROR=', (csr2.cpu() - csr).abs().max())

        csr = torch.tensor(sqrtm(c1.cpu().numpy() @ c2.cpu().numpy())).real.to(device)
        fid = (((m1 - m2)**2).sum() + c1.trace() + c2.trace() - 2 * csr.trace()).item()
        return fid

    def forward(self, dataloader: Sequence[Batch], device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Calculate the FID for the given samples
        """
        current_features = self.calculate_features(dataloader, device, target_nb_samples=len(self.reference_features))
        return self.calculate_fid_from_features(current_features, device=device)
    
    def get_target_nb_samples(self):
        assert self.reference_features is not None, 'features must be precomputed!'
        return len(self.reference_features)