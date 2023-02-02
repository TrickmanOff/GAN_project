from abc import abstractmethod
from typing import Optional, Tuple, List, Any

import numpy as np
import torch
import torch.utils.data

from data import collate_fn, get_random_infinite_dataloader, move_batch_to, stack_batches
from device import get_local_device
from gan import GAN
from physical_metrics import calogan_metrics


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


class MetricsSequence(Metric):
    def __init__(self, *metrics):
        self.metrics = metrics

    def evaluate(self, *args, **kwargs):
        return [metric(*args, **kwargs) for metric in self.metrics]
