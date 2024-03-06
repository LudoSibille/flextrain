from collections import defaultdict
import lightning as L
import os
import logging
from pytorch_lightning.utilities import rank_zero_only
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class CallbackRecordLearninRateMomentum(L.Callback):
    """
    Simple call back to record momentum and learning rate
    of an experiment
    """
    def __init__(self, output_dir_name: str = 'lr', per_batch: bool = True):
        super().__init__()
        self.output_dir_name = output_dir_name
        self.output_path = None
        self.per_batch = per_batch

        self.learning_rates = defaultdict(list)
        self.momentums = defaultdict(list)

    def _record(self, trainer: L.Trainer):
        optimizers = trainer.optimizers
        for optimizer_n, optimizer in enumerate(optimizers):
            for pg_n, pg in enumerate(optimizer.param_groups):
                name = f'opt_{optimizer_n}_pg_{pg_n}'
                self.learning_rates[name].append(pg['lr'])
                if 'betas' in pg:
                    self.momentums[name].append(pg['betas'][0])

    @rank_zero_only
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not self.per_batch:
            self._record(trainer)

    @rank_zero_only
    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, *args, **kwargs):
        if self.per_batch:
            self._record(trainer)

    @rank_zero_only
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.per_batch:
            x_name = 'batches'
        else:
            x_name = 'epochs'


        def plot_dict(values, x_name, output_path, title):
            lrs = pd.DataFrame(values)
            plt.close('all')
            ax = sns.lineplot(lrs)

            ax.set(xlabel=x_name, ylabel='values')
            ax.set_title(title)
            
            output_dir = os.path.join(trainer.options.data.root_current_experiment, self.output_dir_name)
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_path)


        output_dir = os.path.join(trainer.options.data.root_current_experiment, self.output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        plot_dict(self.learning_rates, x_name, os.path.join(output_dir, 'learning_rates.png'), f'Learning rates')
        plot_dict(self.momentums, x_name, os.path.join(output_dir, 'momentums.png'), f'Momentums')
        logger.info(f'plots exported={output_dir}')