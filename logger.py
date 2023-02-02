from abc import abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Set, Optional, Tuple

"""
Currently there's no convenient support for several loggers at one time.
It is expected to be provided if the existing logging interface shows its convenience.
"""


@dataclass
class LoggerConfig:
    ignored_periods: Optional[Set[str]] = None
    ignored_metrics: Optional[Set[str]] = None


def get_default_config() -> LoggerConfig:
    return LoggerConfig(
        ignored_periods={'batch'}
    )


class GANLogger:
    """An abstract class for GAN loggers"""
    def __init__(self, config: Optional[LoggerConfig] = None) -> None:
        self.config = get_default_config() if config is None else config
        self.accumulated_data: Dict[str, Tuple[int, Dict[str, Any]]] = {}   # (period: data with {period: period_index})

    def log_metrics(self, data: Dict[str, Any], period: str, period_index: Optional[int] = None, commit: bool = True) -> None:
        """
        Log values of metrics after some period

        :param data: a dict of metrics values {metric_name: value}
        :param period: the name of a period (e.g., "batch", "epoch")
        :param period_index: the index of a period, if the call is not the first for this period, it may be omitted
        :param commit: if False, data will be accumulated but not logged
        use commit=True only for the last call for the pair (period, period_index)
        """
        if self.config.ignored_periods and period in self.config.ignored_periods:
            return

        if self.config.ignored_metrics is not None:
            data = copy(data)
            for metric in copy(data):
                if metric in self.config.ignored_metrics:
                    data.pop(metric)

        data = copy(data)
        if period in self.accumulated_data:
            prev_period_index, prev_data = self.accumulated_data[period]
            if period_index is not None and prev_period_index != period_index:
                raise RuntimeError(f'Trying to log data for the {period} #{period_index} while the data for the {period} #{prev_period_index} was not logged')
            period_index = prev_period_index
            data.update(prev_data)

        assert period_index is not None, 'Period index is not specified'
        self.accumulated_data[period] = (period_index, data)

        if commit:
            self._log_metrics(data, period, period_index)
            self.accumulated_data.pop(period)

    def commit(self, period: str):
        if period in self.accumulated_data:
            self.log_metrics(data={}, period=period, commit=True)

    @abstractmethod
    def _log_metrics(self, data: Dict[str, Any], period: str, period_index: int) -> None:
        pass

    # Optional for implementation
    # it may plot the histogram, density approximation, etc.
    # or print some statistics of the distribution (mean, variance, etc.)
    def log_distribution(self, values: Dict[str, Iterable[float]], name: str,
                         period: str, period_index: int) -> None:
        pass

    # Optional for implementation
    def log_critic_values_distribution(self, critic_values_true: Iterable[float],
                                       critic_values_gen: Iterable[float],
                                       period: str, period_index: int) -> None:
        """
        Log the distributions of critic values

        :param critic_values_gen: critic values for the generated data
        :param critic_values_true: critic values for the true data, must be the same length as
        `critic_values_gen`
        """
        pass
