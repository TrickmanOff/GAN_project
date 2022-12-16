import typing as tp
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import optim

from device import get_local_device
from gan import GAN
from logger import Logger
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

    def state_dict(self) -> dict[str]:
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str]) -> None:
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['scheduler'])


# знает, как обучать GAN
class GanEpochTrainer(ABC):
    @abstractmethod
    def train_epoch(self, gan_model: GAN, dataset: torch.utils.data.Dataset,
                    generator_stepper: Stepper, critic_stepper: Stepper,
                    logger: Logger | None = None) -> None:
        pass


class WganEpochTrainer(GanEpochTrainer):
    def __init__(self, n_critic: int = 5, batch_size: int = 64, clip_c: float = 0.01) -> None:
        self.n_critic = n_critic
        self.batch_size = batch_size
        self.clip_c = clip_c

    def _clip_model(self, model: nn.Module) -> None:
        with torch.no_grad():
            for param in model.parameters():
                param.clip_(min=-self.clip_c, max=self.clip_c)

    def train_epoch(self, gan_model: GAN, dataset: torch.utils.data.Dataset,
                    generator_stepper: Stepper, critic_stepper: Stepper,
                    logger: Logger | None = None) -> None:
        sampler = torch.utils.data.sampler.RandomSampler(dataset, replacement=True)
        random_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=self.batch_size,
                                                               drop_last=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=random_sampler,
                                                 collate_fn=lambda b_list: torch.stack([el[0] for el in b_list]))  # можно, наверное, переписать проще

        # weight-clipping
        self._clip_model(gan_model.discriminator)

        # critic training
        for t, real_batch in enumerate(dataloader):
            if t == self.n_critic:
                break
            real_batch.to(get_local_device())
            noise_batch = gan_model.gen_noise(self.batch_size).to(get_local_device())

            gen_batch = gan_model.generator(noise_batch)

            loss = - (gan_model.discriminator(real_batch) - gan_model.discriminator(
                gen_batch)).mean()
            if logger is not None:
                logger.log(level='batch', module='train/discriminator',
                           msg={'batch_num': t + 1, 'wasserstein dist estimation': -loss.item()})
                logger.flush(level='batch')
            loss.backward()
            critic_stepper.step()
            critic_stepper.optimizer.zero_grad()
            generator_stepper.optimizer.zero_grad()

            # weight-clipping
            self._clip_model(gan_model.discriminator)

        # generator training
        noise_batch = gan_model.gen_noise(self.batch_size)
        gen_batch = gan_model.generator(noise_batch)
        real_batch = next(iter(dataloader))

        was_loss_approx = (
                    gan_model.discriminator(real_batch) - gan_model.discriminator(gen_batch)).mean()
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
              logger: Logger | None = None) -> tp.Generator[tuple[int, GAN], None, GAN]:

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
