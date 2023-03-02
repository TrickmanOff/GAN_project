import torch

from metrics import *
from physical_metrics.calogan_prd import get_energy_embedding


def create_prd_energy_embed():
    calculated_metric = PhysicsPRDMetric()
    calculated_metric.NAME = 'Energy ' + calculated_metric.NAME

    metric = TransformData(
        calculated_metric,
        transform_fn=get_energy_embedding,
    )
    return metric


def create_conditional_prd_energy_embed():
    calculated_metric = AveragePRDAUCMetric()
    calculated_metric.NAME = 'Energy embed. ' + calculated_metric.NAME

    metric = TransformData(
        ConditionBinsMetric(
            calculated_metric,
            dim_bins=torch.Tensor([3, 3]),
            condition_index=0
        ),
        transform_fn=get_energy_embedding,
    )
    return metric


def create_prd_physics_statistics():
    calculated_metric = PhysicsPRDMetric()
    calculated_metric.NAME = 'PhysStats ' + calculated_metric.NAME

    metric = TransformData(
        calculated_metric,
        transform_fn=DataStatisticsCombiner(
            *[statistic.evaluate_statistic for statistic in PHYS_STATISTICS]
        )
    )
    return metric


def create_conditional_prd_physics_statistics():
    calculated_metric = AveragePRDAUCMetric()
    calculated_metric.NAME = 'PhysStats ' + calculated_metric.NAME

    metric = TransformData(
        ConditionBinsMetric(
            calculated_metric,
            dim_bins=torch.Tensor([3, 3]),
            condition_index=0
        ),
        transform_fn=DataStatisticsCombiner(
            *[statistic.evaluate_statistic for statistic in PHYS_STATISTICS]
        )
    )
    return metric


__all__ = ['create_prd_energy_embed', 'create_conditional_prd_energy_embed',
           'create_prd_physics_statistics', 'create_conditional_prd_physics_statistics']
