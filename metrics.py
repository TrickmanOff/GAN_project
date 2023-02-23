from abc import abstractmethod
from typing import Optional, Tuple, List, Any, Generator, Iterable, Union, Dict

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm

from data import collate_fn, get_random_infinite_dataloader, move_batch_to, stack_batches
from device import get_local_device
from gan import GAN
from physical_metrics import calogan_metrics, calogan_prd
from physical_metrics.calogan_prd import plot_pr_aucs


class Metric:
    NAME = None

    def evaluate(self, *args, **kwargs):
        pass

    def prepare_args(self, **kwargs):
        return kwargs

    def __call__(self, gan_model: Optional[GAN] = None,
                 dataloader: Optional[torch.utils.data.DataLoader] = None,
                 train_dataset: Optional[torch.utils.data.Dataset] = None,
                 val_dataset: Optional[torch.utils.data.Dataset] = None,
                 gen_data: Optional[Any] = None,
                 val_data: Optional[Any] = None,
                 inverse_to_initial_domain_fn: Optional[Any] = None):
        kwargs = {
            'gan_model': gan_model,
            'dataloader': dataloader,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'gen_data': gen_data,
            'val_data': val_data,
            'inverse_to_initial_domain_fn': inverse_to_initial_domain_fn
        }
        kwargs = self.prepare_args(**kwargs)
        return self.evaluate(**kwargs)


# метрика, которая анализирует GAN, и не анализирует данные
class ModelMetric(Metric):
    def prepare_args(self, **kwargs):
        kwargs = super().prepare_args(**kwargs)

        return {
            'gan_model': kwargs['gan_model']
        }


def generate_data(gan_model: GAN, dataloader: torch.utils.data.DataLoader,
                  gen_size: Optional[int] = None) -> Generator:
    """
    Генерирует данные GAN-ом батчами

    если gen_size None, то генерируются по всему dataloader, иначе генерируется хотя бы gen_size значений
    """
    gan_model = gan_model.to(get_local_device())

    gen_data_batches = []
    current_gen_size = 0
    for batch in dataloader:
        batch_x, batch_y = batch
        batch_y = move_batch_to(batch_y, get_local_device())
        noise_batch_z = gan_model.gen_noise(len(batch_x)).to(get_local_device())
        gen_batch_x = gan_model.generator(noise_batch_z, batch_y)
        gen_data_batches.append((gen_batch_x.cpu(), move_batch_to(batch_y, torch.device('cpu'))))
        yield gen_batch_x.cpu(), move_batch_to(batch_y, torch.device('cpu'))

        current_gen_size += len(gen_batch_x)
        if gen_size is not None and current_gen_size >= gen_size:
            return


def limited_batch_iterator(dataloader: Iterable, limit_size: Optional[int] = None) -> Generator:
    current_size = 0
    for batch in dataloader:
        yield batch
        current_size += len(batch[0])
        if limit_size is not None and current_size >= limit_size:
            return


def apply_function_to_x(dataloader, func=None) -> Generator:
    for batch in dataloader:
        batch_x, batch_y = batch
        if func is not None:
            batch_x = func(batch_x)
        yield batch_x, batch_y


# метрика, которая использует сгенерированные и валидационные данные
# плохо сейчас то, что данные возвращаются как один тензор; надо будет заменить на работу
# с dataloader-ами
class DataMetric(Metric):
    def __init__(self, initial_domain_data: bool = False,
                 val_data_size: Optional[int] = None,
                 gen_data_size: Optional[int] = None,
                 cache_val_data: bool = False,
                 dataloader_batch_size: int = 64,
                 shuffle_val_dataset: bool = False,
                 return_as_batches: bool = True):
        """
        :param initial_domain_data:
        :param val_data_size: если None, то передаём все
        :param gen_data_size: если None, то генерируем по val_data_size
        :param dataloader_batch_size: если не передан val_dataloader, то будет использован такой
            размер batch'а
        :param return_as_batches: если False, то объединяет batch'и в тензор
        """
        self.initial_domain_data = initial_domain_data
        self.val_data_size = val_data_size
        self.gen_data_size = gen_data_size
        self.cache_val_data = cache_val_data
        self.cached_val_data = None
        self.dataloader_batch_size = dataloader_batch_size
        self.shuffle_val_dataset = shuffle_val_dataset
        self.return_as_batches = return_as_batches

    def prepare_args(self, **kwargs):
        kwargs = super().prepare_args(**kwargs)
        gan_model = kwargs['gan_model']

        # переданные gen_data и val_data имеют приоритет
        gen_data = kwargs.get('gen_data', None)
        val_data = kwargs.get('val_data', None)
        if val_data is None and self.cached_val_data:
            val_data = self.cached_val_data

        if gen_data is None or val_data is None:
            val_dataloader = kwargs.get('val_dataloader', None)
            if val_dataloader is None or self.shuffle_val_dataset:
                val_dataset = kwargs['val_dataset']
                if self.shuffle_val_dataset:  # shuffling
                    random_indices = np.random.permutation(len(val_dataset))
                    val_dataset = torch.utils.data.Subset(val_dataset, random_indices)
                val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                             batch_size=self.dataloader_batch_size,
                                                             collate_fn=collate_fn)

        if gen_data is None:
            gen_data = generate_data(gan_model=gan_model, dataloader=val_dataloader,
                                     gen_size=self.gen_data_size)
        if val_data is None:
            val_data = limited_batch_iterator(val_dataloader, limit_size=self.val_data_size)

        if self.initial_domain_data:  # преобразуем, если обратная функция дана
            inverse_to_initial_domain_fn = kwargs.get('inverse_to_initial_domain_fn', None)
            if inverse_to_initial_domain_fn is not None:
                gen_data = apply_function_to_x(gen_data, inverse_to_initial_domain_fn)
                if not self.cached_val_data:
                    val_data = apply_function_to_x(val_data, inverse_to_initial_domain_fn)

        if not self.return_as_batches:
            gen_data = stack_batches(list(gen_data))
            if not self.cached_val_data:
                val_data = stack_batches(list(val_data))

        if self.cache_val_data:
            self.cached_val_data = val_data
        # генераторы, выдающие батчи
        return {
            'gan_model': gan_model,
            'gen_data': gen_data,
            'val_data': val_data,
        }


class CriticValuesDistributionMetric(DataMetric):
    NAME = 'Critic values distribution'

    def __init__(self, values_cnt: int = 1000):
        super().__init__(initial_domain_data=False,
                         val_data_size=values_cnt,
                         gen_data_size=None,
                         cache_val_data=False,
                         shuffle_val_dataset=True,
                         return_as_batches=True)

    def evaluate(self, gan_model, gen_data, val_data, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: critic_vals_gen, critic_vals_true
        """

        critic_vals_true = []
        critic_vals_gen = []
        for gen_batch, real_batch in zip(gen_data, val_data):
            with torch.no_grad():
                gen_batch_x, gen_batch_y = move_batch_to(gen_batch, get_local_device())
                real_batch_x, real_batch_y = move_batch_to(real_batch, get_local_device())

                true_vals = gan_model.discriminator(real_batch_x, real_batch_y)
                critic_vals_true.append(true_vals)

                gen_vals = gan_model.discriminator(gen_batch_x, gen_batch_y)
                critic_vals_gen.append(gen_vals)

        return torch.cat(critic_vals_gen).flatten().cpu().numpy(), torch.cat(critic_vals_true).flatten().cpu().numpy()


class DataStatistic(DataMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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


class MetricsSequence(Metric):
    def __init__(self, *metrics):
        self.metrics = metrics

    def evaluate(self, *args, **kwargs):
        return [metric(*args, **kwargs) for metric in self.metrics]

    def __iter__(self):
        return iter(self.metrics)


class DataStatistics(DataMetric):
    """Uses the same generation results for all statistics"""
    def __init__(self, *statistics: DataStatistic, **kwargs):
        DataMetric.__init__(self, **kwargs)
        self.statistics = statistics

    def evaluate(self, *args, **kwargs):
        return [statistic.evaluate(*args, **kwargs) for statistic in self.statistics]


class PhysicsDataStatistics(DataStatistics):
    def __init__(self, *statistics: DataStatistic):
        data_metric_kwargs = {
            'initial_domain_data': True,
            'cache_val_data': True,
            'return_as_batches': False,
        }
        super().__init__(*statistics, **data_metric_kwargs)


def split_prep_physics_data(data):
    EnergyDeposit, (ParticlePoint, ParticleMomentum) = data
    EnergyDeposit = torch.squeeze(EnergyDeposit)
    return EnergyDeposit.detach().numpy(), ParticlePoint.detach().numpy(), ParticleMomentum.detach().numpy()


class PhysicsDataStatistic(DataStatistic):
    def __init__(self):
        super().__init__(
            initial_domain_data=True,
            cache_val_data=True,
            return_as_batches=False,
        )


class LongitudualClusterAsymmetryMetric(PhysicsDataStatistic):
    NAME = 'Longitudual Cluster Asymmetry'

    def evaluate_statistic(self, data):
        EnergyDeposit, ParticlePoint, ParticleMomentum = split_prep_physics_data(data)
        return calogan_metrics.get_assymetry(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=False)


class TransverseClusterAsymmetryMetric(PhysicsDataStatistic):
    NAME = 'Transverse Cluster Asymmetry'

    def evaluate_statistic(self, data):
        EnergyDeposit, ParticlePoint, ParticleMomentum = split_prep_physics_data(data)
        return calogan_metrics.get_assymetry(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=True)


class ClusterLongitudualWidthMetric(PhysicsDataStatistic):
    NAME = 'Cluster Longitudual Width'

    def evaluate_statistic(self, data):
        EnergyDeposit, ParticlePoint, ParticleMomentum = split_prep_physics_data(data)
        return calogan_metrics.get_shower_width(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=False)


class ClusterTransverseWidthMetric(PhysicsDataStatistic):
    NAME = 'Cluster Transverse Width'

    def evaluate_statistic(self, data):
        EnergyDeposit, ParticlePoint, ParticleMomentum = split_prep_physics_data(data)
        return calogan_metrics.get_shower_width(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=True)


class PhysicsPRDMetric(PhysicsDataStatistic):
    def evaluate(self, gen_data: Any, val_data, **kwargs) -> Tuple[Any, Any]:
        precisions, recalls = calogan_prd.calc_pr_rec(data_real=val_data[0], data_fake=gen_data[0])
        return precisions, recalls


class AveragePRDAUCMetric(PhysicsDataStatistic):
    NAME = 'Average PRD-AUC'

    def evaluate(self, gen_data: Any,
                 val_data: Optional[Any] = None,
                 **kwargs):
        precisions, recalls = PhysicsPRDMetric().evaluate(gen_data=gen_data, val_data=val_data)
        pr_aucs = plot_pr_aucs(precisions=precisions, recalls=recalls)
        return np.mean(pr_aucs)


def select_indices(data: Union[torch.Tensor, Tuple[torch.Tensor]], indices):
    if isinstance(data, tuple):
        return tuple(select_indices(t, indices) for t in data)
    else:
        return data[indices]


def split_into_bins(data: Union[torch.Tensor, Tuple[torch.Tensor]], condition_data: torch.Tensor,
                    dim_bins: torch.Tensor,
                    ret_bins: bool = False):
    """
    если ret_bins == True, то возвращается (bins_codes, bins), где bins - массив ограничений каждого bin'а
    TODO

    dim_bins - кол-во bin-ов для каждой размерности
    разделяет data по bin-ам

    dim_bins.shape == condition_data.shape[1:]

    возвращает разбиение data
    """

    mins = condition_data.min(dim=0)[0]
    maxs = condition_data.max(dim=0)[0]
    steps = (maxs - mins) / dim_bins

    # я не знаю, как это можно сделать лучше
    bins_mul = [1]
    for el in dim_bins.flatten()[1:]:
        bins_mul.append(int(bins_mul[-1] * el))
    bins_mul = torch.LongTensor(bins_mul, ).reshape(dim_bins.shape)
    # ------

    dims_codes = (condition_data - mins) // steps
    dims_codes = torch.maximum(dims_codes, torch.zeros(dims_codes.shape))
    dims_codes = torch.minimum(dims_codes, dim_bins - 1)
    dims_codes = dims_codes.long()

    condition_dims = tuple(range(1, len(condition_data.shape)))
    bins_codes = (dims_codes * bins_mul).sum(
        dim=condition_dims)  # номера bin-ов, в которых лежат данные

    # разбиваем data на bin'ы
    data_bins = []
    max_bin_index = int(dim_bins.prod())
    all_indices = torch.arange(len(condition_data))
    for bin_index in range(max_bin_index):
        cur_indices = all_indices[bins_codes == bin_index]
        if len(cur_indices) == 0:
            cur_bin = None
        else:
            cur_bin = select_indices(data, cur_indices)
        data_bins.append(cur_bin)

    # сами бины
    # TODO
    if ret_bins:
        raise NotImplemented

    return data_bins


class PhysicsPRDBinsMetric(PhysicsDataStatistic):
    def __init__(self, dim_bins=torch.Tensor([3, 3])):
        """
        делит по points
        """
        super().__init__()
        self.dim_bins = dim_bins

    def evaluate(self, gen_data, val_data, **kwargs):
        points = val_data[1][0]
        gen_x, val_x = gen_data[0], val_data[0]

        splitted_data = split_into_bins((gen_x, val_x), condition_data=points,
                                        dim_bins=self.dim_bins)

        prd_aucs = []
        # TODO: add PRD curve to log
        for gen_bin, val_bin in tqdm(splitted_data):
            if gen_bin is None or val_bin is None:
                prd_auc = 0.  # zero recall or zero precision respectively
            else:
                precisions, recalls = calogan_prd.calc_pr_rec(data_real=val_bin, data_fake=gen_bin)
                cur_prd_aucs = plot_pr_aucs(precisions=precisions, recalls=recalls)
                plt.close()
                prd_auc = np.mean(cur_prd_aucs)
            prd_aucs.append(prd_auc)

        return prd_aucs


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
    NAME = 'KL Divergence'

    def __init__(self, statistic: DataStatistic, bins_cnt: int = 10):
        super().__init__()
        self.statistic = statistic
        self.bins_cnt = bins_cnt
        self.NAME = self.NAME + ' of ' + statistic.NAME

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


def _unravel_metric_results(unraveled: Dict[str, Any], metric: Metric, results) -> None:
    if isinstance(metric, MetricsSequence):
        for metric, res in zip(metric, results):
            _unravel_metric_results(unraveled, metric, res)
    else:
        unraveled[metric.NAME] = results


def unravel_metric_results(metric: Metric, results) -> Dict[str, Any]:
    """
    преобразует результаты вычисленной metric в словарь
    {<имя метрики>: значение}
    нужно из-за MetricsSequence
    """
    unraveled: Dict[str, Any] = {}
    _unravel_metric_results(unraveled, metric, results)
    return unraveled


__all__ = ['Metric', 'CriticValuesDistributionMetric',
           'DataStatistic', 'DataStatistics', 'DataMetric',
           'LongitudualClusterAsymmetryMetric', 'TransverseClusterAsymmetryMetric',
           'ClusterLongitudualWidthMetric', 'ClusterTransverseWidthMetric',
           'PhysicsPRDMetric', 'PhysicsDataStatistics', 'PhysicsDataStatistic',
           'KLDivergence',
           'MetricsSequence',
           'AveragePRDAUCMetric',
           'unravel_metric_results',
           'PhysicsPRDBinsMetric']
