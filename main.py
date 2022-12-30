from typing import Tuple, Generator, Optional

import torch

import data
from discriminators import SimpleImageDiscriminator, MNISTDiscriminator, SimplePhysicsDiscriminator
from gan import GAN
from generators import SimpleImageGenerator, MNISTGenerator, SimplePhysicsGenerator
from logger import Logger, StreamHandler
from normalization import apply_normalization, ClippingNormalizer, SpectralNormalizer
from storage import ExperimentsStorage
from train import Stepper, WganEpochTrainer, GanTrainer


def init_storage() -> ExperimentsStorage:
    # === config variables ===
    experiments_dir = './experiments'
    checkpoint_filename = './training_checkpoint'
    model_state_filename = './model_state'
    # ========================
    return ExperimentsStorage(experiments_dir=experiments_dir, checkpoint_filename=checkpoint_filename,
                              model_state_filename=model_state_filename)


experiments_storage = init_storage()


def init_logger() -> Logger:
    handlers = {
        'config': StreamHandler(),
        'epoch': StreamHandler(),
        # 'batch': StreamHandler(),
    }
    logger = Logger()
    for level, handler in handlers.items():
        logger.add_handler(level, handler)
    return logger


def form_gan_trainer(model_name: str, gan_model: Optional[GAN] = None, n_epochs: int = 100) -> Generator[Tuple[int, GAN], None, GAN]:
    """
    :return: a generator that yields (epoch number, gan_model after this epoch)
    """
    # classes_cnt = 10

    logger = init_logger()
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

    epoch_trainer = WganEpochTrainer(n_critic=10, batch_size=64)

    model_dir = experiments_storage.get_model_dir(model_name)
    trainer = GanTrainer(model_dir=model_dir, use_saved_checkpoint=True, save_checkpoint_once_in_epoch=500)
    train_gan_generator = trainer.train(dataset=dataset, gan_model=gan_model,
                                        generator_stepper=generator_stepper,
                                        critic_stepper=discriminator_stepper,
                                        epoch_trainer=epoch_trainer,
                                        n_epochs=n_epochs,
                                        logger=logger)
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
