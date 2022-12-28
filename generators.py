from typing import Tuple

import numpy as np
import torch
from torch import nn


# взят из репозитория https://github.com/LucaAmbrogioni/Wasserstein-GAN-on-MNIST/blob/master/Wasserstein%20GAN%20playground.ipynb
class MNISTGenerator(nn.Module):
    def __init__(self, noise_dim: int):
        super().__init__()

        conv_channels = 512
        base_width = 3

        self.model = nn.Sequential(
            nn.Linear(in_features=noise_dim, out_features=base_width * base_width * conv_channels),
            nn.Unflatten(dim=1, unflattened_size=(conv_channels, base_width, base_width)),
            nn.BatchNorm2d(num_features=conv_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=conv_channels, out_channels=conv_channels // 2,
                               kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=conv_channels // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=conv_channels // 2, out_channels=conv_channels // 4,
                               kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=conv_channels // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=conv_channels // 4, out_channels=conv_channels // 8,
                               kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=conv_channels // 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=conv_channels // 8, out_channels=1, kernel_size=3,
                               stride=3, padding=1),
            nn.Tanh(),
            # nn.Sigmoid()
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y = self.model(X)
        return y


class SimpleImageGenerator(nn.Module):
    def __init__(self, noise_dim: int, output_shape: Tuple[int, ...]):
        super().__init__()
        output_len = int(np.prod(output_shape))
        hidden_neurons = (noise_dim + output_len) // 2
        self.model = nn.Sequential(
            nn.Linear(in_features=noise_dim, out_features=hidden_neurons),
            nn.ReLU(),
            nn.Linear(in_features=hidden_neurons, out_features=output_len),
        )
        self.output_shape = output_shape

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y = self.model(X)

        batch_size = X.shape[0]

        return y.reshape(batch_size, *self.output_shape)
