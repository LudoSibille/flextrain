from typing import Any, Optional

import lightning as L
import torch.optim


def scheduler_steps_fn(
    optimizer: torch.optim.Optimizer, m: L.LightningModule, nb_steps: int = 5, gamma: float = 0.5
) -> torch.optim.lr_scheduler.LRScheduler:
    step_size = max(int(m.trainer.options.training.nb_epochs) // nb_steps, 1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return scheduler


def scheduler_one_cycle_cosine_fn(
    optimizer: torch.optim.Optimizer, m: L.LightningModule, **kwargs: Any
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
    scheduler_conf = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

    return scheduler_conf


class _PolyLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
        current_step: Optional[int] = None,
    ) -> None:
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step: Optional[int] = None) -> None:
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


def scheduler_poly_fn(
    optimizer: torch.optim.Optimizer, m: L.LightningModule, initial_lr: float, max_epochs: int, exponent: float = 0.9
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler = _PolyLRScheduler(optimizer, initial_lr=initial_lr, max_steps=max_epochs, exponent=exponent)
    return scheduler
