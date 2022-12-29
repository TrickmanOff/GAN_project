from abc import abstractmethod
from typing import Tuple, Any

import torch
from torch import nn

import aux


class Discriminator(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, y: Any = None) -> torch.Tensor:
        """
        :param x: object from the considered space
        :param y: condition
        None means no condition.
        A discriminator knows the exact type of condition and how to use it.
        If discriminator does not support conditions, it is expected to raise an exception.
        """
        pass


def save_dimensions_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    works only for odd kernel size values
    returns padding size such that the output has the same coordinate dimensions
    """
    res = []
    for sz in kernel_size:
        if sz % 2 == 0:
            raise ValueError('Only odd kernel size values are supported')
        res.append((sz - 1) // 2)
    return tuple(res)


# взят из репозитория https://github.com/LucaAmbrogioni/Wasserstein-GAN-on-MNIST/blob/master/Wasserstein%20GAN%20playground.ipynb
class MNISTDiscriminator(Discriminator):
    def __init__(self, condition_classes_cnt: int = 0):
        super().__init__()
        self.condition_classes_cnt = condition_classes_cnt

        y_out = 256  # размерность вектора, в который переводится y (ohe)

        self.y_transform = nn.Linear(in_features=condition_classes_cnt, out_features=y_out)
        self.x_to_vector = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(in_features=3*3*512 + y_out, out_features=1)

    def forward(self, x: torch.Tensor, y: Any = None) -> torch.Tensor:
        x_vec = self.x_to_vector(x)
        if y is not None:
            assert isinstance(y, torch.Tensor)
            y = aux.ohe_labels(y, self.condition_classes_cnt)
            y_vec = self.y_transform(y)
            x_vec = torch.concat((x_vec, y_vec), dim=1)

        return self.fc(x_vec)


class SimpleImageDiscriminator(Discriminator):  # for (1 x 28 x 28) images
    def __init__(self):
        super().__init__()

        # backbone
        conv_channels = 28

        self.backbone_seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=(3, 3),
                      padding=save_dimensions_padding((3, 3))),
            nn.BatchNorm2d(num_features=conv_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(3, 3),
                      padding=save_dimensions_padding((3, 3))),
            nn.BatchNorm2d(num_features=conv_channels),
        )

        self.backbone_end = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(7, 7)),  # -> conv_channels x 4 x 4
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=conv_channels * 4 * 4, out_features=1)
        )

    def forward(self, x: torch.Tensor, y: Any = None) -> torch.Tensor:
        if y is not None:
            raise RuntimeError('Discriminator does not support condition')
        backbone_seq_out = self.backbone_seq(x)
        backbone_out = self.backbone_end(backbone_seq_out)
        out = self.head(backbone_out)
        return out
