from copy import copy
from typing import Dict, Any, Set, Optional, Iterable

import numpy as np
import wandb
from scipy.stats import gaussian_kde

from logger import GANLogger, LoggerConfig


class _WandbLogger(GANLogger):
    """
    Should be used only from WandbCM
    """
    def __init__(self, config: Optional[LoggerConfig] = None):
        super().__init__(config=config)

    def _log_metrics(self, data: Dict[str, Any], period: str, period_index: int) -> None:
        # if period not in self._periods:
        wandb.define_metric(period)
        for metric in data:
            wandb.define_metric(metric, step_metric=period)
        logged_dict = copy(data)
        logged_dict[period] = period_index
        wandb.log(logged_dict)

    def log_critic_values_distribution(self, critic_values_true: Iterable[float],
                                       critic_values_gen: Iterable[float],
                                       period: str, period_index: int) -> None:
        kernel_true = gaussian_kde(critic_values_true)
        kernel_gen = gaussian_kde(critic_values_gen)

        min_val = min(min(critic_values_true), min(critic_values_gen))
        max_val = max(max(critic_values_true), max(critic_values_gen))
        min_val -= 0.1
        max_val += 1

        xs = np.linspace(min_val, max_val, num=25)
        ys_true = kernel_true(xs)
        ys_gen = kernel_gen(xs)
        wandb.log({f'critic_values_plot': wandb.plot.line_series(
                  xs=xs,
                  ys=[ys_true, ys_gen],
                  keys=['true', 'generated'],
                  title=f'Critic values distributions {period}#{period_index}',
                  xname='critic value')})


class WandbCM:
    """
    Wandb logger context manager

    calls wandb.login(), wandb.init() and wandb.finish()
    """
    def __init__(self, project_name: str, experiment_id: str, token: str, config: Optional[LoggerConfig] = None) -> None:
        self.project_name = project_name
        self.experiment_id = experiment_id
        self.token = token
        self.config = config

    def __enter__(self) -> _WandbLogger:
        wandb.login(key=self.token)
        wandb.init(
            project=self.project_name,
            name=self.experiment_id
        )
        return _WandbLogger()

    def __exit__(self, exc_type, exc_val, exc_tb):
        wandb.finish()
