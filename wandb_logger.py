from copy import copy
from typing import Dict, Any, Set, Optional

import wandb

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
