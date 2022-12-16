import typing as tp

import torch

import data
from discriminators import SimpleImageDiscriminator, MNISTDiscriminator
from gan import GAN
from generators import SimpleImageGenerator, MNISTGenerator
from logger import StreamLogger
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


def form_gan_trainer(model_name: str, gan_model: GAN | None = None, n_epochs: int = 100) -> tp.Generator[tuple[int, GAN], None, GAN]:
    logger = StreamLogger()
    dataset = data.get_mnist_dataset()

    noise_dimension = 50

    def uniform_noise_generator(n: int) -> torch.Tensor:
        return 2*torch.rand(size=(n, noise_dimension)) - 1  # [-1, 1]

    generator = MNISTGenerator(noise_dim=noise_dimension)
    discriminator = MNISTDiscriminator()

    if gan_model is None:
        gan_model = GAN(generator, discriminator, uniform_noise_generator)

    generator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(generator.parameters(), lr=1e-3)
    )

    discriminator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(discriminator.parameters(), lr=1e-5)
    )

    epoch_trainer = WganEpochTrainer(n_critic=5, batch_size=64, clip_c=0.01)

    model_dir = experiments_storage.get_model_dir(model_name)
    trainer = GanTrainer(model_dir=model_dir, use_saved_checkpoint=True)
    train_gan_generator = trainer.train(dataset=dataset, gan_model=gan_model,
                                        generator_stepper=generator_stepper,
                                        critic_stepper=discriminator_stepper,
                                        epoch_trainer=epoch_trainer,
                                        n_epochs=n_epochs,
                                        logger=logger)
    return train_gan_generator


def main():
    model_name = 'mnist_test'
    gan_trainer = form_gan_trainer(model_name=model_name, n_epochs=100)
    gan_model = yield from gan_trainer
    # do sth with gan model


if __name__ == '__main__':
    main()
