from typing import Tuple, Generator, Dict, Any, Optional
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import optim

from device import get_local_device
from gan import GAN
from logger import Logger
from normalization import update_normalizers_stats
from storage import ModelDir


class Stepper:
    """обёртка над всем необходимым для шага градиентного спуска"""
    def __init__(self, optimizer: optim.Optimizer,
                 scheduler=None, scheduler_mode: str = 'epoch') -> None:
        """
        TODO: над scheduler хочу сделать обёртку, чтобы передавать в него вообще всё во время обучение

        scheduler_mode - mode of the scheduler
           'epoch' - step after each epoch
           'batch' - step after each batch
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_mode = scheduler_mode

    def step(self, *args, **kwargs) -> None:
        self.optimizer.step()
        if self.scheduler is not None and self.scheduler_mode == 'batch':
            self.scheduler.step(*args, **kwargs)

    def epoch_finished(self, *args, **kwargs) -> None:
        if self.scheduler is not None and self.scheduler_mode == 'epoch':
            self.scheduler.step(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['scheduler'])


# знает, как обучать GAN
class GanEpochTrainer(ABC):
    @abstractmethod
    def train_epoch(self, gan_model: GAN, dataset: torch.utils.data.Dataset,
                    generator_stepper: Stepper, critic_stepper: Stepper,
                    logger: Optional[Logger] = None) -> None:
        pass


class WganEpochTrainer(GanEpochTrainer):
    def __init__(self, n_critic: int = 5, batch_size: int = 64) -> None:
        self.n_critic = n_critic
        self.batch_size = batch_size

    def train_epoch(self, gan_model: GAN, dataset: torch.utils.data.Dataset,
                    generator_stepper: Stepper, critic_stepper: Stepper,
                    logger: Optional[Logger] = None) -> None:
        sampler = torch.utils.data.sampler.RandomSampler(dataset, replacement=True)
        random_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=self.batch_size,
                                                               drop_last=False)
        def collate_fn(els_list: list):
            if isinstance(els_list[0], tuple):  # x, y
                return torch.stack([el[0] for el in els_list]), torch.Tensor([el[1] for el in els_list])
            else:
                return torch.stack(els_list)
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=random_sampler,
                                                 collate_fn=collate_fn)

        def get_batches() -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
            """Look at the return statement"""
            real_batch = next(iter(dataloader))
            if isinstance(real_batch, tuple):
                assert len(real_batch) == 2
                real_batch_x, real_batch_y = real_batch
                real_batch_x = real_batch_x.to(get_local_device())
                real_batch_y = real_batch_y.to(get_local_device())
            else:
                real_batch_x, real_batch_y = real_batch.to(get_local_device()), None
            gen_batch_y = real_batch_y
            noise_batch_z = gan_model.gen_noise(self.batch_size).to(get_local_device())
            gen_batch_x = gan_model.generator(noise_batch_z, gen_batch_y).to(get_local_device())

            return gen_batch_x, real_batch_x, gen_batch_y, real_batch_y

        # critic training
        for t in range(self.n_critic):
            gen_batch_x, real_batch_x, gen_batch_y, real_batch_y = get_batches()

            loss = - (gan_model.discriminator(real_batch_x, real_batch_y) -
                      gan_model.discriminator(gen_batch_x, gen_batch_y)).mean()
            if logger is not None:
                logger.log(level='batch', module='train/discriminator',
                           msg={'batch_num': t + 1, 'wasserstein dist estimation': -loss.item()})
                logger.flush(level='batch')
            loss.backward()
            critic_stepper.step()
            critic_stepper.optimizer.zero_grad()
            generator_stepper.optimizer.zero_grad()
            update_normalizers_stats(gan_model.discriminator)

        # generator training
        gen_batch_x, real_batch_x, gen_batch_y, real_batch_y = get_batches()

        was_loss_approx = (gan_model.discriminator(real_batch_x, real_batch_y) -
                           gan_model.discriminator(gen_batch_x, gen_batch_y)).mean()
        was_loss_approx.backward()
        generator_stepper.step()
        generator_stepper.optimizer.zero_grad()
        critic_stepper.optimizer.zero_grad()

        if logger is not None:
            logger.log(level='epoch', module='train/generator',
                       msg={'wasserstein dist': was_loss_approx.item()})


# знает, что нужно для обучения GAN
class GanTrainer:
    def __init__(self, model_dir: ModelDir, save_checkpoint: bool = True,
                 use_saved_checkpoint: bool = True) -> None:
        self.model_dir = model_dir
        self.save_checkpoint = save_checkpoint
        self.use_saved_checkpoint = use_saved_checkpoint

    def train(self, dataset: torch.utils.data.Dataset, gan_model: GAN,
              generator_stepper: Stepper, critic_stepper: Stepper,
              epoch_trainer: GanEpochTrainer,
              n_epochs: int = 100,
              logger: Optional[Logger] = None) -> Generator[Tuple[int, GAN], None, GAN]:

        gan_model.to(get_local_device())

        epoch = 1

        if self.use_saved_checkpoint:
            checkpoint = self.model_dir.get_checkpoint_state()
            if checkpoint is not None:
                epoch = checkpoint['epoch']
                print(f"Checkpoint was loaded. Current epoch: {epoch}")
                gan_model.load_state_dict(checkpoint['gan'])
                generator_stepper.load_state_dict(checkpoint['generator_stepper'])
                critic_stepper.load_state_dict(checkpoint['critic_stepper'])

        while epoch <= n_epochs:
            if logger is not None:
                logger.log(level='epoch', module='train', msg={'epoch_num': epoch})

            epoch_trainer.train_epoch(gan_model=gan_model, dataset=dataset,
                                      generator_stepper=generator_stepper, critic_stepper=critic_stepper,
                                      logger=logger)

            logger.flush(level='epoch')
            epoch += 1

            if self.save_checkpoint:
                checkpoint = {
                    'epoch': epoch,
                    'gan': gan_model.state_dict(),
                    'generator_stepper': generator_stepper.state_dict(),
                    'critic_stepper': critic_stepper.state_dict()
                }
                self.model_dir.save_checkpoint_state(checkpoint)

            yield epoch, gan_model

        return gan_model
