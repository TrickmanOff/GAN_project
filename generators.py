from abc import abstractmethod
from typing import Tuple, Any, Optional

import numpy as np
import torch
from torch import nn

import aux


class Generator(nn.Module):
    @abstractmethod
    def forward(self, z: torch.Tensor, y: Any = None) -> torch.Tensor:
        """
        :param z: seed/noise for generation
        :param y: condition
        None means no condition.
        A generator knows the exact type of condition and how to use it for generation.
        If generator does not support conditions, it is expected to raise an exception.
        """
        pass


# взят из репозитория https://github.com/LucaAmbrogioni/Wasserstein-GAN-on-MNIST/blob/master/Wasserstein%20GAN%20playground.ipynb
class MNISTGenerator(Generator):
    def __init__(self, noise_dim: int, condition_classes_cnt: int = 0):
        """
        Uses one-hot-encoded label as optional condition
        0 means no condition
        """
        self.condition_classes_cnt = condition_classes_cnt

        super().__init__()

        conv_channels = 512
        base_width = 3
        y_out = 1000     # размерность вектора, в который переводится y (ohe)
        noise_out = 200

        self.noise_transform = nn.Linear(in_features=noise_dim, out_features=noise_out)
        self.y_transform = nn.Linear(in_features=condition_classes_cnt, out_features=y_out)

        self.model = nn.Sequential(
            nn.Linear(in_features=noise_out + y_out,
                      out_features=base_width * base_width * conv_channels),
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

    def forward(self, z: torch.Tensor, y: Any = None) -> torch.Tensor:
        """
        :param y: integer labels
        """
        z = self.noise_transform(z)
        if y is not None:
            assert isinstance(y, torch.Tensor)
            # apply one-hot-encoding
            y_vec = aux.ohe_labels(y, self.condition_classes_cnt)
            y_trans = self.y_transform(y_vec)
            z = torch.concat((z, y_trans), dim=1)
        x = self.model(z)
        return x


class SimpleImageGenerator(Generator):
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

    def forward(self, z: torch.Tensor, y: Any = None) -> torch.Tensor:
        if y is not None:
            raise RuntimeError('Generator does not support condition')
        x = self.model(z)
        batch_size = z.shape[0]
        return x.reshape(batch_size, *self.output_shape)
