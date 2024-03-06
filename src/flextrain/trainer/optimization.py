import lightning as L
import torch.optim


def scheduler_steps_fn(
    optimizer: torch.optim.Optimizer, m: L.LightningModule, nb_steps: int = 5, gamma: float = 0.5
) -> torch.optim.lr_scheduler.LRScheduler:
    step_size = max(int(m.trainer.options.training.nb_epochs) // nb_steps, 1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return scheduler


def scheduler_one_cycle_cosine_fn(
    optimizer: torch.optim.Optimizer, m: L.LightningModule, **kwargs
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
    scheduler_conf = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

    return scheduler_conf
