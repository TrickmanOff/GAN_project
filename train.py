import contextlib
from typing import Tuple, Generator, Dict, Any, Optional, ContextManager, Callable, List
from abc import ABC, abstractmethod

import torch
import torch.utils.data
from torch import optim
from tqdm import tqdm

from data import collate_fn, move_batch_to, get_random_infinite_dataloader
from device import get_local_device
from gan import GAN
from logger import GANLogger
from metrics import Metric, MetricsSequence
from metrics_logging import log_metric
from normalization import update_normalizers_stats
from predicates import TrainPredicate
from regularizer import Regularizer
from results_storage import ExperimentInfo, Result
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
    def train_epoch(self, gan_model: GAN,
                    train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset,
                    generator_stepper: Stepper, critic_stepper: Stepper,
                    logger: Optional[GANLogger] = None,
                    regularizer: Optional[Regularizer] = None) -> None:
        pass


class WganEpochTrainer(GanEpochTrainer):
    def __init__(self, n_critic: int = 5, batch_size: int = 64) -> None:
        self.n_critic = n_critic
        self.batch_size = batch_size

    def train_epoch(self, gan_model: GAN,
                    train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset,
                    generator_stepper: Stepper, critic_stepper: Stepper,
                    logger: Optional[GANLogger] = None,
                    regularizer: Optional[Regularizer] = None) -> None:
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)
        random_dataloader = get_random_infinite_dataloader(train_dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        random_dataloader_iter = iter(random_dataloader)

        def get_batches(real_batch) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
            """Look at the return statement"""
            real_batch_x, real_batch_y = move_batch_to(real_batch, get_local_device())
            gen_batch_y = real_batch_y
            noise_batch_z = gan_model.gen_noise(len(real_batch_x)).to(get_local_device())
            gen_batch_x = gan_model.generator(noise_batch_z, gen_batch_y).to(get_local_device())

            return gen_batch_x, real_batch_x, gen_batch_y, real_batch_y

        critic_loss_total = 0.
        gen_loss_total = 0.

        # generator_batch = next(random_dataloader_iter)  # for local testing
        # for batch_index in range(1):                    # -----------------
        for batch_index, generator_batch in enumerate(tqdm(dataloader)):
            # critic training
            gan_model.generator.requires_grad_(False)
            for t in range(self.n_critic):
                real_batch = next(random_dataloader_iter)
                gen_batch_x, real_batch_x, gen_batch_y, real_batch_y = get_batches(real_batch)
                disc_real_vals = gan_model.discriminator(real_batch_x, real_batch_y)
                disc_gen_vals = gan_model.discriminator(gen_batch_x, gen_batch_y)
                loss = - (disc_real_vals - disc_gen_vals).mean()
                if regularizer is not None:
                    loss += regularizer()
                if logger is not None:
                    logger.log_metrics(data={'train/discriminator/loss': -loss.item()},
                                       period='batch', period_index=t+1, commit=True)
                loss.backward()
                critic_stepper.step()
                critic_stepper.optimizer.zero_grad()
                update_normalizers_stats(gan_model.discriminator, disc_real_vals=disc_real_vals,
                                         disc_gen_vals=disc_gen_vals)

            critic_loss_total += loss.item() * len(gen_batch_x)

            gan_model.generator.requires_grad_(True)

            # generator training
            gan_model.discriminator.requires_grad_(False)
            gen_batch_x, real_batch_x, gen_batch_y, real_batch_y = get_batches(generator_batch)

            observations = (gan_model.discriminator(real_batch_x, real_batch_y) -
                            gan_model.discriminator(gen_batch_x, gen_batch_y))
            gen_loss = observations.mean()
            if regularizer is not None:
                loss += regularizer()
                regularizer.step()
            gen_loss_total += gen_loss * len(gen_batch_x)
            gen_loss.backward()
            generator_stepper.step()
            generator_stepper.optimizer.zero_grad()
            gan_model.discriminator.requires_grad_(True)

            if logger is not None:
                logger.log_metrics(data={'train/generator/loss': gen_loss.item()},
                                   period='batch', period_index=batch_index+1, commit=False)

        generator_stepper.epoch_finished()
        critic_stepper.epoch_finished()

        if logger is not None:
            if generator_stepper.scheduler is not None:
                generator_lr = generator_stepper.scheduler.get_last_lr()
                logger.log_metrics(
                    data={'generator/lr': generator_lr},
                    period='epoch',
                    commit=False)
            if critic_stepper.scheduler is not None:
                critic_lr = critic_stepper.scheduler.get_last_lr()
                logger.log_metrics(
                    data={'critic/lr': critic_lr},
                    period='epoch',
                    commit=False)

            logger.log_metrics(data={'train/critic/loss': critic_loss_total / len(train_dataset)},
                               period='epoch',
                               commit=False)
            logger.log_metrics(data={'train/generator/loss': gen_loss_total / len(train_dataset)},
                               period='epoch',
                               commit=False)  # period_index was specified by the caller


def fill_result(result: Result, metric_names: List, metric_values: List):
    for metric_name, metric_value in zip(metric_names, metric_values):
        if isinstance(metric_name, List):
            fill_result(result, metric_name, metric_value)
        else:
            result.add_metric(metric_name, value=metric_value)


# знает, что нужно для обучения GAN
class GanTrainer:
    def __init__(self, model_dir: ModelDir, save_checkpoint_once_in_epoch: int = 1,
                 use_saved_checkpoint: bool = True) -> None:
        self.model_dir = model_dir
        self.save_checkpoint_once_in_epoch = save_checkpoint_once_in_epoch
        self.use_saved_checkpoint = use_saved_checkpoint

    def train(self, gan_model: GAN,
              train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset,
              generator_stepper: Stepper, critic_stepper: Stepper,
              epoch_trainer: GanEpochTrainer,
              n_epochs: int = 100,
              metric: Optional[Metric] = None,
              metric_predicate: Optional[TrainPredicate] = None,
              logger_cm_fn: Optional[Callable[[], ContextManager[GANLogger]]] = None,
              regularizer: Optional[Regularizer] = None,
              normalization_loss: Optional[Callable] = None,
              result_metrics: Optional[Tuple[List, MetricsSequence]] = None,
              results_info: Optional[ExperimentInfo] = None) -> Generator[Tuple[int, GAN], None, GAN]:
        """
        :param train_dataset: is expected to return tuples (x, y)
        :param val_dataset: is expected to return tuples (x, y)
        :param metric: metric that will be calculated and logged (only if logger is given) after each epoch
        :param metric_predicate: `metric` is calculated if and only if `metric_predicate` returns True
        :return:
        """
        gan_model.to(get_local_device())
        inverse_to_initial_domain_fn = getattr(train_dataset, 'inverse_transform', None)
        epoch = 1

        if self.use_saved_checkpoint:
            checkpoint = self.model_dir.get_checkpoint_state()
            if checkpoint is not None:
                epoch = checkpoint['epoch']
                print(f"Checkpoint was loaded. Current epoch: {epoch}")
                gan_model.load_state_dict(checkpoint['gan'])
                generator_stepper.load_state_dict(checkpoint['generator_stepper'])
                critic_stepper.load_state_dict(checkpoint['critic_stepper'])

        if logger_cm_fn is None:
            logger_cm = contextlib.nullcontext(None)
        else:
            logger_cm = logger_cm_fn() or contextlib.nullcontext(None)

        with logger_cm as logger:
            while epoch <= n_epochs:
                if logger is not None:
                    logger.log_metrics(data={}, period='epoch', period_index=epoch, commit=False)

                epoch_trainer.train_epoch(gan_model=gan_model, train_dataset=train_dataset, val_dataset=val_dataset,
                                          generator_stepper=generator_stepper, critic_stepper=critic_stepper,
                                          logger=logger, regularizer=regularizer)

                if logger is not None:
                    if metric is not None and (metric_predicate is None or metric_predicate(epoch=epoch)):
                        metrics_results = metric(gan_model=gan_model, train_dataset=train_dataset, val_dataset=val_dataset,
                                                 inverse_to_initial_domain_fn=inverse_to_initial_domain_fn)
                        log_metric(metric, results=metrics_results, logger=logger, period='epoch', period_index=epoch)
                    logger.commit(period='epoch')
                epoch += 1

                if self.save_checkpoint_once_in_epoch != 0 and epoch % self.save_checkpoint_once_in_epoch == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'gan': gan_model.state_dict(),
                        'generator_stepper': generator_stepper.state_dict(),
                        'critic_stepper': critic_stepper.state_dict()
                    }
                    self.model_dir.save_checkpoint_state(checkpoint)

                yield epoch, gan_model

        # evaluating results
        # TODO: probably this logic should be placed somewhere else
        if result_metrics is not None and results_info is not None:
            result = results_info.get_result()
            # metrics
            metrics_values = result_metrics[1](gan_model=gan_model, train_dataset=train_dataset,
                                               val_dataset=val_dataset,
                                               inverse_to_initial_domain_fn=inverse_to_initial_domain_fn)
            print(metrics_values)
            fill_result(result, result_metrics[0], metrics_values)
            results_info.save_result(result)

        return gan_model
