from dataclasses import dataclass
import logging
from typing import Any, Callable, Optional, Tuple
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import os
from functools import partial
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from torch import nn
import numpy as np
import seaborn as sns
from ..diffusion.utils import catch_all_and_log
from .callback import Callback


logger = logging.getLogger(__name__)


@dataclass
class ActivationStats:
    mean: float
    std: float
    histogram: np.ndarray
    dead_unit_ratio: float
    

class HookActivationStats:
    def __init__(self, m: nn.Module, nb_histogram_bins: int, histogram_clamp_min_max: Tuple[float, float]) -> None:
        self.hook = m.register_forward_hook(self.hook_fn)
        self.nb_histogram_bins = nb_histogram_bins
        self.histogram_clamp_min_max = histogram_clamp_min_max
        assert len(histogram_clamp_min_max) == 2
        assert histogram_clamp_min_max[0] < histogram_clamp_min_max[1]
        
        range_frac_0 = (0 - (self.histogram_clamp_min_max[0])) / (self.histogram_clamp_min_max[1] - self.histogram_clamp_min_max[0])
        self.histogram_zero_bin = int(round(range_frac_0 * self.nb_histogram_bins))

        self.means = None
        self.stds = None
        self.histograms = None
        self.clear()

    @torch.no_grad()
    def hook_fn(self, module: nn.Module, inputs: Any, output: torch.Tensor):
        std, mean = torch.std_mean(output)
        self.means.append(float(mean))
        self.stds.append(float(std))

        # keep the histogramming on GPU, else this is very slow
        histogram = torch.histc(
            output.detach().float(), 
            bins=self.nb_histogram_bins, 
            min=self.histogram_clamp_min_max[0], 
            max=self.histogram_clamp_min_max[1]
        )

        # normalize the histogram to avoid very large numbers
        histogram_sum = histogram.sum() + 1e-5
        histogram_normalized = histogram / histogram_sum
        self.histograms += histogram_normalized.cpu().numpy()

    def clear(self) -> None:
        self.means = []
        self.stds = []
        self.histograms = np.zeros(self.nb_histogram_bins, dtype=np.float64)

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def get_stats(self) -> Optional[ActivationStats]:
        if self.means is None or len(self.means) == 0:
            return None
        
        histogram_mean = self.histograms / len(self.means)
        sum_hist = sum(histogram_mean)
        if abs(sum_hist - 1.0) >= 0.2:
            logger.warning(f'problem with histogramming. At this point the histogram should be normalized to be a probability. Maybe due to float16? Got={sum_hist} instead of 1.0')

        dead_unit_ratio = np.sum(histogram_mean[self.histogram_zero_bin - 1:self.histogram_zero_bin + 2])
        
        return ActivationStats(
            mean=np.mean(self.means),
            std=np.mean(self.stds),
            histogram = histogram_mean,
            dead_unit_ratio=dead_unit_ratio
        )

    def __del__(self):
        self.remove()



def collect_conv_and_dense(m: nn.Module) -> bool:
    return type(m) in (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)


class CallbackLogLayerStatistics(Callback):
    """
    Log model statistics for debugging.

    In particular, we want to make sure the model layer outputs are close to 0-mean 
    and 1-standard deviation with fraction of dead unit as close to 0 as possible

    Args:
        collect_layers_fn: function that indicates what layer to calculate the statistics 
        histogram_transform: a transform for better display for histogram range
        layer_stride: analyze only a portion of the collected layers, with `layer_stride` skipped
        collect_batches_per_epoch: if not None, collect statistics per several batches, else
            use all the batches to get the statistics
    """
    def __init__(
            self, 
            collect_layers_fn: Callable[[nn.Module], bool] = collect_conv_and_dense, 
            layer_stride: int = 1,
            nb_histogram_bins: int = 64, 
            histogram_clamp_min_max: Tuple[float, float]=(-2, 10),  # asymmetric: show have low number of negative values
            nb_histogram_columns: int = 1,
            histogram_transform: Callable[[np.ndarray], np.ndarray] = np.log1p,
            histogram_display_factor: float = 1e6,
            histogram_short_excerpt: int = 300,
            collect_batches_per_epoch: Optional[int] = 1,
            export_name: str = 'activations') -> None:
        super().__init__()
        self.collect_layers_fn = collect_layers_fn
        self.layer_stride = layer_stride
        self.nb_histogram_bins = nb_histogram_bins
        self.histogram_clamp_min_max = histogram_clamp_min_max
        self.export_name = export_name
        self.nb_histogram_columns = nb_histogram_columns
        self.histogram_transform = histogram_transform
        self.histogram_display_factor = histogram_display_factor
        self.collect_batches_per_epoch = collect_batches_per_epoch
        self.histogram_short_excerpt = histogram_short_excerpt
        
        self.hooks = None
        self.stats_by_layer_by_epoch = None


    @rank_zero_only
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        layers = [m for m in pl_module.modules() if self.collect_layers_fn(m)][::self.layer_stride]
        logger.info(f'#layers tracked={len(layers)}')
        for layer_n, layer in enumerate(layers):
            logger.info(f' Layer[{layer_n}]={layer}')
        
        assert len(layers) > 0, 'no layer found!'
        self.stats_by_layer_by_epoch = [[] for n in range(len(layers))]
        self.layers = layers

        self.hooks = [HookActivationStats(l, nb_histogram_bins=self.nb_histogram_bins, histogram_clamp_min_max=self.histogram_clamp_min_max) for l in layers]

    @catch_all_and_log
    @rank_zero_only
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # be sure the hooks were removed
        for h in self.hooks:
            h.remove()

        self.hooks = None

        # export the stats
        def collect_value(stats, attribute_name):
            values = np.asarray([getattr(s, attribute_name) for s in stats])
            return values
        
        def collect_value_by_layer(collect_fn):
            values = [collect_fn(stats) for stats in self.stats_by_layer_by_epoch]
            return values
        

        output_dir = os.path.join(trainer.options.data.root_current_experiment, self.export_name)
        os.makedirs(output_dir, exist_ok=True)
        xlabel_name = 'epochs' if self.collect_batches_per_epoch is None else 'batches'

        for attribute_name in ('mean', 'std', 'dead_unit_ratio'):
            plt.close()
            mean_by_layers = collect_value_by_layer(partial(collect_value, attribute_name=attribute_name))
            ax = sns.lineplot(mean_by_layers)
            ax.set(xlabel=xlabel_name, ylabel='values')
            ax.set_title(f'Activation {attribute_name} by epoch of the model layers')
            plt.savefig(os.path.join(output_dir, f'activation_{attribute_name}.png'))

        nb_cols = self.nb_histogram_columns
        nb_rows = int(np.ceil(len(self.stats_by_layer_by_epoch) / float(nb_cols)))
        
        def show_histograms(excerpt):
            plt.close('all')
            h_by_layers = collect_value_by_layer(partial(collect_value, attribute_name='histogram'))
            fig, axes = plt.subplots(nrows=nb_rows, ncols=nb_cols, dpi=200, figsize=(16, 2 * len(h_by_layers)))

            fig.suptitle(f'Activation histogram by layer. Transform={str(self.histogram_transform)}')
            for layer_n, ax in enumerate(axes.flatten()):
                ax.set_xticks([])
                ax.set_yticks([])
                last_row = layer_n >= (nb_cols * (nb_rows - 1))
                xlabel = xlabel_name if last_row else ''

                ylabel = 'fraction' if layer_n % nb_cols == 0 else ''
                ax.set(xlabel=xlabel, ylabel=ylabel)

            for layer_n, (ax, histograms) in enumerate(zip(axes.flatten(), h_by_layers)):
                # `log` the histogram to decrease the importance bins with the most activations
                # else we cannot see the activations that are less frequent.
                # Although `histograms` is normalized to have bin in [0..1], so artificially
                # increase the values but a large factor then apply the normalization
                image = self.histogram_transform(self.histogram_display_factor * np.stack(histograms).T)
                if excerpt:
                    image = image[:, :self.histogram_short_excerpt]

                # revert the image y-axis (-1: bottom, 1: top)
                ax.imshow(image, origin='lower', interpolation='nearest', aspect='auto')
                ax.set_title(f'layer: {layer_n}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'activation_histograms_excerpt_{excerpt}.png'))

        show_histograms(excerpt=False)
        show_histograms(excerpt=True)

    def _collect_stat(self):
        assert len(self.hooks) == len(self.stats_by_layer_by_epoch)
        for hook_n, hook in enumerate(self.hooks):
            stats = hook.get_stats()
            hook.clear()
        
            if stats is None:
                logger.warning(f'hook={hook_n} had no activation collected!')
            else:
                self.stats_by_layer_by_epoch[hook_n].append(stats)

    @rank_zero_only
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # collect statistics per few batches
        if self.collect_batches_per_epoch is not None and batch_idx % self.collect_batches_per_epoch == 0: 
            self._collect_stat()
    

    @rank_zero_only
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # collect a single statistics per epoch
        if self.collect_batches_per_epoch is None:
            self._collect_stat()
