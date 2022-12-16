import torch
from torch import nn


def save_dimensions_padding(kernel_size: tuple[int, int]) -> tuple[int, int]:
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
class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
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
            nn.Linear(in_features=3*3*512, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SimpleImageDiscriminator(nn.Module):  # for (1 x 28 x 28) images
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

    def forward(self, x):
        backbone_seq_out = self.backbone_seq(x)
        backbone_out = self.backbone_end(backbone_seq_out)
        out = self.head(backbone_out)
        return out
