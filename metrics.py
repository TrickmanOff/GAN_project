from abc import abstractmethod
from typing import Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from data import collate_fn, get_random_infinite_dataloader, move_batch_to, stack_batches
from device import get_local_device
from gan import GAN
from physical_metrics import calogan_metrics, calogan_prd
from physical_metrics.calogan_prd import plot_pr_aucs


class Metric:
    def evaluate(self, *args, **kwargs):
        pass

    def __call__(self, gan_model: Optional[GAN] = None,
                 dataloader: Optional[torch.utils.data.DataLoader] = None,
                 train_dataset: Optional[torch.utils.data.Dataset] = None,
                 val_dataset: Optional[torch.utils.data.Dataset] = None,
                 gen_data: Optional[Any] = None,
                 val_data: Optional[Any] = None):

        return self.evaluate(gan_model=gan_model, dataloader=dataloader,
                             train_dataset=train_dataset, val_dataset=val_dataset,
                             gen_data=gen_data, val_data=val_data)


class CriticValuesDistributionMetric(Metric):
    def __init__(self, values_cnt: int = 100):
        """
        :param values_cnt: the number of critic values to calculate (separately for generator and real data)
        """
        self.values_cnt = values_cnt

    def evaluate(self, gan_model: Optional[GAN] = None, train_dataset: Optional[torch.utils.data.Dataset] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: critic_vals_gen, critic_vals_true
        """
        if gan_model is None or train_dataset is None:
            raise RuntimeError('Required arguments not given')

        batch_size = 64

        # random batches
        dataloader = get_random_infinite_dataloader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
        dataloader_iter = iter(dataloader)

        critic_vals_true = []
        critic_vals_gen = []

        batches_cnt = (self.values_cnt + batch_size - 1) // batch_size
        for _ in range(batches_cnt):
            real_batch_x, real_batch_y = move_batch_to(next(dataloader_iter), get_local_device())
            with torch.no_grad():
                # (batch_size, )
                true_vals = gan_model.discriminator(real_batch_x, real_batch_y)
                critic_vals_true.append(true_vals)

                gen_batch_y = real_batch_y
                noise_batch_z = gan_model.gen_noise(batch_size).to(get_local_device())
                gen_batch_x = gan_model.generator(noise_batch_z, gen_batch_y).to(get_local_device())
                gen_vals = gan_model.discriminator(gen_batch_x, gen_batch_y)
                critic_vals_gen.append(gen_vals)

        return torch.cat(critic_vals_gen).flatten().cpu().numpy(), torch.cat(critic_vals_true).flatten().cpu().numpy()


def generate_data(gan_model: GAN, dataloader: torch.utils.data.DataLoader):
    """Генерирует данные GAN-ом батчами"""
    gan_model = gan_model.to(get_local_device())

    gen_data_batches = []
    for batch in dataloader:
        batch_x, batch_y = batch
        batch_y = move_batch_to(batch_y, get_local_device())
        noise_batch_z = gan_model.gen_noise(len(batch_x)).to(get_local_device())
        gen_batch_x = gan_model.generator(noise_batch_z, batch_y)
        gen_data_batches.append((gen_batch_x.cpu(), move_batch_to(batch_y, torch.device('cpu'))))

    return stack_batches(gen_data_batches)


class DataStatistic(Metric):
    def __init__(self):
        self.cached_val_value = None

    @abstractmethod
    def evaluate_statistic(self, data):
        pass

    # data format: (X, (Y1, ..., Yk)) or (X, None), where X and Yi are Torch.tensor's
    def evaluate(self, gen_data: Any,
                 val_data: Optional[Any] = None,
                 **kwargs) -> Tuple[Any, Any]:
        gen_value = self.evaluate_statistic(gen_data)

        if self.cached_val_value is None:
            if val_data is not None:
                self.cached_val_value = self.evaluate_statistic(val_data)

        return gen_value, self.cached_val_value


class DataStatistics(Metric):
    """Uses same generation results for all statistics"""
    def __init__(self, *statistics: DataStatistic):
        self.statistics = statistics

    def evaluate(self, val_data: Optional[Any] = None,
                 val_dataset: Optional[torch.utils.data.Dataset] = None,
                 val_dataloader: Optional[torch.utils.data.DataLoader] = None,
                 gan_model: Optional[GAN] = None,
                 gen_data: Optional[Any] = None,
                 **kwargs):
        if gen_data is None:
            assert gan_model is not None
            if val_dataloader is None:
                assert val_dataset is not None
                val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
            gen_data = generate_data(gan_model=gan_model, dataloader=val_dataloader)
        return [statistic(gen_data=gen_data, val_data=val_data) for statistic in self.statistics]


def split_prep_physics_data(data):
    EnergyDeposit, (ParticlePoint, ParticleMomentum) = data
    EnergyDeposit = torch.squeeze(EnergyDeposit)
    return EnergyDeposit.detach().numpy(), ParticlePoint.detach().numpy(), ParticleMomentum.detach().numpy()


class LongitudualClusterAsymmetryMetric(DataStatistic):
    def evaluate_statistic(self, data):
        EnergyDeposit, ParticlePoint, ParticleMomentum = split_prep_physics_data(data)
        return calogan_metrics.get_assymetry(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=False)


class TransverseClusterAsymmetryMetric(DataStatistic):
    def evaluate_statistic(self, data):
        EnergyDeposit, ParticlePoint, ParticleMomentum = split_prep_physics_data(data)
        return calogan_metrics.get_assymetry(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=True)


class ClusterLongitudualWidthMetric(DataStatistic):
    def evaluate_statistic(self, data):
        EnergyDeposit, ParticlePoint, ParticleMomentum = split_prep_physics_data(data)
        return calogan_metrics.get_shower_width(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=False)


class ClusterTransverseWidthMetric(DataStatistic):
    def evaluate_statistic(self, data):
        EnergyDeposit, ParticlePoint, ParticleMomentum = split_prep_physics_data(data)
        return calogan_metrics.get_shower_width(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=True)


class PhysicsPRDMetric(DataStatistic):
    def evaluate(self, gen_data: Any,
                 val_data: Optional[Any] = None,
                 **kwargs) -> Tuple[Any, Any]:
        precisions, recalls = calogan_prd.calc_pr_rec(data_real=val_data[0], data_fake=gen_data[0])
        return precisions, recalls


class AveragePRDAUCMetric(DataStatistic):
    def evaluate(self, gen_data: Any,
                 val_data: Optional[Any] = None,
                 **kwargs):
        precisions, recalls = PhysicsPRDMetric().evaluate(gen_data=gen_data, val_data=val_data)
        pr_aucs = plot_pr_aucs(precisions=precisions, recalls=recalls)
        return np.mean(pr_aucs)


def _split_into_bins(bins, vals):
    """
    return densities of shape (len(bins) + 1,)
    """
    bin_indices = np.searchsorted(bins, vals)
    unique_vals, cnts = np.unique(bin_indices, return_counts=True)
    all_cnts = np.zeros(len(bins) + 1)
    all_cnts[unique_vals] = cnts

    return all_cnts / len(vals)


def _kl_div(true_probs, fake_probs):
    """
    true_probs, fake_probs must be of the same size.
    They are assumed to be probabilities of some discrete random variables
    return KL(true || fake)
    """
    calc_indices = true_probs != 0
    if (fake_probs[calc_indices] == 0.).any():
        return np.inf
    else:
        return (true_probs[calc_indices] * np.log(true_probs[calc_indices] / fake_probs[calc_indices])).mean()


class KLDivergence(DataStatistic):
    def __init__(self, statistic: DataStatistic, bins_cnt: int = 10):
        super().__init__()
        self.statistic = statistic
        self.bins_cnt = bins_cnt

    def evaluate(self, gen_data: Any,
                 val_data: Optional[Any] = None,
                 **kwargs):
        """
        делим val_samples на bin-ы по квантилям и считаем, что влево и вправо на бесконечности уходят по ещё одному bin-у
        затем по дискретизированным согласно этим bin-ам величинам считаем дивергенцию
        """
        gen_samples, val_samples = self.statistic.evaluate(gen_data=gen_data, val_data=val_data)
        _, bins = pd.qcut(np.hstack(gen_samples), q=self.bins_cnt, retbins=True)

        val_probs = _split_into_bins(bins, val_samples)
        gen_probs = _split_into_bins(bins, gen_samples)
        return _kl_div(true_probs=val_probs, fake_probs=gen_probs)


class MetricsSequence(Metric):
    def __init__(self, *metrics):
        self.metrics = metrics

    def evaluate(self, *args, **kwargs):
        return [metric(*args, **kwargs) for metric in self.metrics]


__all__ = ['Metric', 'CriticValuesDistributionMetric',
           'DataStatistic', 'DataStatistics',
           'LongitudualClusterAsymmetryMetric', 'TransverseClusterAsymmetryMetric',
           'ClusterLongitudualWidthMetric', 'ClusterTransverseWidthMetric',
           'PhysicsPRDMetric',
           'KLDivergence',
           'MetricsSequence',
           'AveragePRDAUCMetric']
