import tempfile
from copy import copy
from typing import Dict, Any, Set, Optional, Iterable

import numpy as np
import wandb
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from logger import GANLogger, LoggerConfig


class _WandbLogger(GANLogger):
    """
    Should be used only from WandbCM
    """
    PYPLOT_FORMAT = 'png'

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

    def log_distribution(self, values: Dict[str, Iterable[float]], name: str,
                         period: str, period_index: int) -> None:
        keys = []
        ys = []

        min_x = min(min(vals) for vals in values.values())
        max_x = max(max(vals) for vals in values.values())

        xs = np.linspace(min_x-0.1, max_x+0.1, num=25)

        for key, vals in values.items():
            keys.append(key)
            kernel = gaussian_kde(vals)
            y = kernel(xs)
            ys.append(y)

        wandb.log({name: wandb.plot.line_series(
            xs=xs,
            ys=ys,
            keys=keys,
            title=f'{name} {period}#{period_index}',
            xname='value')})

    def log_critic_values_distribution(self, critic_values_true: Iterable[float],
                                       critic_values_gen: Iterable[float],
                                       period: str, period_index: int) -> None:
        self.log_distribution(
            values={
                'gen': critic_values_gen,
                'true': critic_values_true,
                },
            name='Critic values',
            period=period, period_index=period_index)

    def log_pyplot(self, name: str, period: str, period_index: int) -> None:
        file = tempfile.NamedTemporaryFile()
        plt.savefig(file, format=self.PYPLOT_FORMAT)
        plt.close()

        wandb.log({name: wandb.Image(file.name)})

        file.close()


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
