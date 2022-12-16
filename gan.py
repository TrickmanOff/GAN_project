from typing import Callable

import torch
from torch import nn


class GAN(nn.Module):
    def __init__(self, generator: nn.Module, discriminator: nn.Module,
                 noise_generator: Callable[[int], torch.Tensor]) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.noise_generator = noise_generator

    def gen_noise(self, n: int) -> torch.Tensor:
        return self.noise_generator(n)

    def forward(self, noise=None):
        noise = noise or self.gen_noise(1)
        return self.generator(noise)

    def state_dict(self, **kwargs) -> dict[str]:
        return {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }

    def load_state_dict(self, state_dict: dict[str], strict: bool = True) -> None:
        self.generator.load_state_dict(state_dict['generator'])
        self.discriminator.load_state_dict(state_dict['discriminator'])

    def to(self, device) -> 'GAN':
        self.generator.to(device)
        self.discriminator.to(device)
        return self