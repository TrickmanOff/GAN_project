"""A bridge between metrics and logger"""
from typing import Any, Union

import numpy as np

from logger import GANLogger
from metrics import Metric, MetricsSequence, CriticValuesDistributionMetric


def log_metric(metric: Metric, results: Any, logger: GANLogger, period: str, period_index: int) -> None:
    """
    :param metric:
    :param logger:
    """
    if isinstance(metric, MetricsSequence):
        for metric, result in zip(metric.metrics, results):
            log_metric(metric, result, logger, period=period, period_index=period_index)
    elif isinstance(metric, CriticValuesDistributionMetric):
        critic_vals_true: np.ndarray
        critic_vals_gen: np.ndarray
        critic_vals_true, critic_vals_gen = results
        logger.log_critic_values_distribution(critic_vals_true, critic_vals_gen, period=period, period_index=period_index)
    else:
        raise NotImplementedError('This metric is not supported for logging')
