import contextlib
import os
from enum import Enum, auto
from typing import Tuple, Generator, Optional, Dict, List

import numpy as np
import torch
import torch.utils.data

import data
import logger
from discriminators import SimplePhysicsDiscriminator, CaloganPhysicsDiscriminator
from evaluation import evaluate_model
from gan import GAN
from generators import SimplePhysicsGenerator, CaloganPhysicsGenerator
from metrics import *
from custom_metrics import *
from normalization import apply_normalization, SpectralNormalizer
from results_storage import ResultsStorage
from storage import ExperimentsStorage
from train import Stepper, WganEpochTrainer, GanTrainer
from wandb_logger import WandbCM


def init_storage() -> ExperimentsStorage:
    # === config variables ===
    experiments_dir = './experiments'
    checkpoint_filename = './training_checkpoint'
    model_state_filename = './model_state'
    # ========================
    return ExperimentsStorage(experiments_dir=experiments_dir, checkpoint_filename=checkpoint_filename,
                              model_state_filename=model_state_filename)


experiments_storage = init_storage()


def init_results_storage() -> ResultsStorage:
    # === config variables ===
    results_dir = './results'
    results_filename = './results.json'
    # ========================
    return ResultsStorage(storage_dir=results_dir, results_filename=results_filename)


results_storage = init_results_storage()


class Environment(Enum):
    LOCAL = auto()
    KAGGLE = auto()


ENV = Environment.LOCAL


def get_wandb_token() -> str:
    if ENV is Environment.LOCAL:
        return os.getenv('WANDB_TOKEN')
    elif ENV is Environment.KAGGLE:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret('WANDB_TOKEN')


def init_logger(model_name: str = '', project_name='GANs'):
    config = logger.get_default_config()
    @contextlib.contextmanager
    def logger_cm():
        try:
            with WandbCM(project_name=project_name, experiment_id=model_name, token=get_wandb_token(), config=config) as wandb_logger:
                yield wandb_logger
        finally:
            pass
    return logger_cm


def form_metric() -> Metric:
    return MetricsSequence(
        CriticValuesDistributionMetric(values_cnt=1000),
        PhysicsDataStatistics(
            *[statistic_cls() for statistic_cls in PHYS_STATISTICS],
            create_prd_energy_embed(),
            create_conditional_prd_energy_embed(),
            create_prd_physics_statistics(),
            create_conditional_prd_physics_statistics(),
        ),
    )


def form_result_metrics() -> Metric:
    return MetricsSequence(
                # DataStatistics(
                    # KLDivergence(LongitudualClusterAsymmetryMetric()),
                    # KLDivergence(TransverseClusterAsymmetryMetric()),
                    # KLDivergence(ClusterLongitudualWidthMetric()),
                    # KLDivergence(ClusterTransverseWidthMetric()),
                    # AveragePRDAUCMetric(),
                # ),
            )


def form_dataset(train: bool = False) -> torch.utils.data.Dataset:
    data_filepath = '../caloGAN_case11_5D_120K.npz'
    return data.UnifiedDatasetWrapper(data.get_physics_dataset(data_filepath, train=train))


def form_gan_trainer(model_name: str, gan_model: Optional[GAN] = None, n_epochs: int = 100) -> Generator[Tuple[int, GAN], None, GAN]:
    """
    :return: a generator that yields (epoch number, gan_model after this epoch)
    """
    logger_cm_fn = init_logger(model_name)
    metric = form_metric()

    train_dataset = form_dataset(train=True)
    val_dataset = form_dataset(train=False)

    # for local testing
    val_size = int(0.1 * len(val_dataset))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(val_size))
    # -------
    noise_dimension = 50

    def uniform_noise_generator(n: int) -> torch.Tensor:
        return 2*torch.rand(size=(n, noise_dimension)) - 1  # [-1, 1]

    generator = CaloganPhysicsGenerator(noise_dim=noise_dimension)
    discriminator = CaloganPhysicsDiscriminator()
    discriminator = apply_normalization(discriminator, SpectralNormalizer)

    if gan_model is None:
        gan_model = GAN(generator, discriminator, uniform_noise_generator)

    generator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(generator.parameters(), lr=1e-4)
    )

    discriminator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)
    )

    epoch_trainer = WganEpochTrainer(n_critic=5, batch_size=100)

    model_dir = experiments_storage.get_model_dir(model_name)
    trainer = GanTrainer(model_dir=model_dir, use_saved_checkpoint=True, save_checkpoint_once_in_epoch=500)
    train_gan_generator = trainer.train(gan_model=gan_model,
                                        train_dataset=train_dataset, val_dataset=val_dataset,
                                        generator_stepper=generator_stepper,
                                        critic_stepper=discriminator_stepper,
                                        epoch_trainer=epoch_trainer,
                                        n_epochs=n_epochs,
                                        metric=metric,
                                        logger_cm_fn=logger_cm_fn)
    return train_gan_generator


def call_generator(generator):
    """Like 'yield from' but doesn't make the current function a generator"""
    while True:
        try:
            generator.send(None)
        except StopIteration as exc:
            return exc.value


def main():
    model_name = 'physics_test'
    gan_trainer = form_gan_trainer(model_name=model_name, n_epochs=100)
    gan_model = call_generator(gan_trainer)
    # evaluate the trained model
    result_metrics = form_result_metrics()
    val_dataset = form_dataset(train=False)
    evaluate_model(model_name, gan_model, val_dataset, result_metrics, storage=results_storage)
    # do sth with gan model


if __name__ == '__main__':
    main()
