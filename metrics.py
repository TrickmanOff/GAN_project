from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.utils.data

from data import collate_fn
from device import get_local_device
from gan import GAN


class Metric:
    def evaluate(self, *args, **kwargs):
        pass

    def __call__(self, gan_model: Optional[GAN] = None,
                 dataloader: Optional[torch.utils.data.DataLoader] = None,
                 dataset: Optional[torch.utils.data.Dataset] = None):
        return self.evaluate(gan_model=gan_model, dataloader=dataloader, dataset=dataset)


class CriticValuesDistributionMetric(Metric):
    def __init__(self, values_cnt: int = 100):
        """
        :param values_cnt: the number of critic values to calculate (separately for generator and real data)
        """
        self.values_cnt = values_cnt

    def evaluate(self, gan_model: Optional[GAN] = None, dataset: Optional[torch.utils.data.Dataset] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: critic_vals_true, critic_vals_gen
        """
        if gan_model is None or dataset is None:
            raise RuntimeError('Required arguments not given')

        batch_size = 64

        # random batches
        sampler = torch.utils.data.sampler.RandomSampler(dataset, replacement=True)
        random_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch_size,
                                                               drop_last=False)

        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=random_sampler, collate_fn=collate_fn)
        dataloader_iter = iter(dataloader)

        critic_vals_true = []
        critic_vals_gen = []

        batches_cnt = (self.values_cnt + batch_size - 1) // batch_size
        for _ in range(batches_cnt):
            real_batch_x, real_batch_y = next(dataloader_iter)
            with torch.no_grad():
                # (batch_size, )
                true_vals = gan_model.discriminator(real_batch_x, real_batch_y)
                critic_vals_true.append(true_vals)

                gen_batch_y = real_batch_y
                noise_batch_z = gan_model.gen_noise(batch_size).to(get_local_device())
                gen_batch_x = gan_model.generator(noise_batch_z, gen_batch_y).to(get_local_device())
                gen_vals = gan_model.discriminator(gen_batch_x, gen_batch_y)
                critic_vals_gen.append(gen_vals)

        return torch.cat(critic_vals_true).flatten().numpy(), torch.cat(critic_vals_gen).flatten().numpy()


class MetricsSequence(Metric):
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    def evaluate(self, *args, **kwargs):
        return [metric(*args, **kwargs) for metric in self.metrics]
