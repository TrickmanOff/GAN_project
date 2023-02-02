"""A bridge between metrics and logger"""
from dataclasses import dataclass
from typing import Any, Union, Tuple, Optional, Dict, Type

import numpy as np

from logger import GANLogger
from metrics import Metric, MetricsSequence, CriticValuesDistributionMetric, \
                    LongitudualClusterAsymmetryMetric, DataStatistic, DataStatistics, \
                    TransverseClusterAsymmetryMetric, ClusterLongitudualWidthMetric, \
                    ClusterTransverseWidthMetric, \
                    PhysicsPRDMetric
from physical_metrics.calogan_prd import plot_pr_aucs

"""
Agreements:
- All functions that use matplotlib.pyplot to plot a chart should not call `plt.show()`.
They are expected to create new figure and use it. When logging two options are possible:
a) plt.show()
b) plt.savefig() + plt.close()
"""


def _reduce_range(x: np.ndarray, quantile_removed: float, values_range: Optional[Tuple[float, float]] = None):
    """
    Removes at least (quantile_removed/2)*100% minimal and maximal values.
    Then only values from the range `values_range` (exclusively) are left, if this range is not None.
    """
    lower_bound = np.quantile(x, quantile_removed / 2)
    upper_bound = np.quantile(x, 1 - quantile_removed / 2)
    if values_range is not None:
        lower_bound = max(lower_bound, values_range[0])
        upper_bound = min(upper_bound, values_range[1])
    return x[(x > lower_bound) & (x < upper_bound)]


# for metrics that return samples
@dataclass
class DistributionLogInfo:
    name: str
    values_range: Optional[Tuple[float, float]] = None


DISTRIBUTIONS_LOG_INFO: Dict[Type, DistributionLogInfo] = {
    LongitudualClusterAsymmetryMetric: DistributionLogInfo(name='Longitudual Cluster Asymmetry'),
    TransverseClusterAsymmetryMetric: DistributionLogInfo(name='Transverse Cluster Asymmetry'),
    ClusterLongitudualWidthMetric: DistributionLogInfo(name='Cluster Longitudual Width',
                                                       values_range=(0, 15)),
    ClusterTransverseWidthMetric: DistributionLogInfo(name='Cluster Transverse Width',
                                                      values_range=(0, 15)),
}


def log_metric(metric: Metric, results: Any, logger: GANLogger, period: str, period_index: int,
               quantile_removed: float = 0.01) -> None:
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
    elif isinstance(metric, DataStatistics):
        for statistic, res in zip(metric.statistics, results):
            log_metric(statistic, res, logger, period=period, period_index=period_index)
    elif isinstance(metric, DataStatistic):
        if type(metric) == PhysicsPRDMetric:
            precisions, recalls = results
            pr_aucs = plot_pr_aucs(precisions=precisions, recalls=recalls)
            logger.log_pyplot('PRD', period=period, period_index=period_index)
            logger.log_metrics(data={'PRD PR-AUC': np.mean(pr_aucs)}, period=period, period_index=period_index, commit=False)
        elif type(metric) in DISTRIBUTIONS_LOG_INFO:
            dist_log_info = DISTRIBUTIONS_LOG_INFO[type(metric)]

            gen_values, true_values = results
            gen_values = _reduce_range(gen_values, quantile_removed, values_range=dist_log_info.values_range)
            values = {'gen': gen_values}
            if true_values is not None:
                true_values = _reduce_range(true_values, quantile_removed, values_range=dist_log_info.values_range)
                values['true'] = true_values
            logger.log_distribution(values=values, name=dist_log_info.name, period=period,
                                    period_index=period_index)
        else:
            raise NotImplementedError  # currently, all DataStatistic-s return samples from a distribution
    else:
        raise NotImplementedError(f'Metric "{type(metric)}" is not supported for logging')
