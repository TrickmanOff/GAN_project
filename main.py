import contextlib
import os
from enum import Enum, auto
from typing import Tuple, Generator, Optional

import torch

import data
import logger
from discriminators import SimpleImageDiscriminator, MNISTDiscriminator, SimplePhysicsDiscriminator
from gan import GAN
from generators import SimpleImageGenerator, MNISTGenerator, SimplePhysicsGenerator
from metrics import CriticValuesDistributionMetric, Metric
from normalization import apply_normalization, ClippingNormalizer, SpectralNormalizer
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


def init_logger(model_name: str = ''):
    config = logger.get_default_config()
    @contextlib.contextmanager
    def logger_cm():
        try:
            with WandbCM(project_name='GANs', experiment_id=model_name, token=get_wandb_token(), config=config) as wandb_logger:
                yield wandb_logger
        finally:
            pass
    return logger_cm


def form_metric() -> Metric:
    return CriticValuesDistributionMetric(values_cnt=1000)


def form_gan_trainer(model_name: str, gan_model: Optional[GAN] = None, n_epochs: int = 100) -> Generator[Tuple[int, GAN], None, GAN]:
    """
    :return: a generator that yields (epoch number, gan_model after this epoch)
    """
    logger_cm_fn = init_logger(model_name)
    metric = form_metric()
    # classes_cnt = 10
    # dataset = data.get_physics_dataset('/kaggle/input/physics-gan/caloGAN_case11_5D_120K.npz')
    dataset = data.get_physics_dataset('../caloGAN_case11_5D_120K.npz')

    noise_dimension = 50

    def uniform_noise_generator(n: int) -> torch.Tensor:
        return 2*torch.rand(size=(n, noise_dimension)) - 1  # [-1, 1]

    generator = SimplePhysicsGenerator(noise_dim=noise_dimension)
    discriminator = SimplePhysicsDiscriminator()
    discriminator = apply_normalization(discriminator, SpectralNormalizer)

    if gan_model is None:
        gan_model = GAN(generator, discriminator, uniform_noise_generator)

    generator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(generator.parameters(), lr=1e-5)
    )

    discriminator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(discriminator.parameters(), lr=1e-5)
    )

    epoch_trainer = WganEpochTrainer(n_critic=20, batch_size=64)

    model_dir = experiments_storage.get_model_dir(model_name)
    trainer = GanTrainer(model_dir=model_dir, use_saved_checkpoint=True, save_checkpoint_once_in_epoch=500)
    train_gan_generator = trainer.train(dataset=dataset, gan_model=gan_model,
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
    model_name = 'mnist_test'
    gan_trainer = form_gan_trainer(model_name=model_name, n_epochs=100)
    gan_model = call_generator(gan_trainer)
    # do sth with gan model


if __name__ == '__main__':
    main()
