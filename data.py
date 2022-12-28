from typing import Tuple

import torch
import torchvision
from torch import nn
from torchvision import transforms


class LinearTransform(nn.Module):
    def __init__(self, min_to: float, max_to: float, dim: int) -> None:
        super().__init__()
        self.min_to = min_to
        self.max_to = max_to
        self.dim = dim

    def forward(self, X) -> torch.Tensor:
        mins = X.min(dim=self.dim).values
        maxs = X.max(dim=self.dim).values

        coefs = (self.max_to - self.min_to) / (maxs - mins)
        biases = self.min_to - coefs * mins

        shape = list(X.shape)
        shape[self.dim] = 1

        coefs = coefs.reshape(*shape)
        biases = biases.reshape(*shape)

        y = coefs * X + biases
        y = torch.clip(y, self.min_to, self.max_to)  # clipping for making sure

        return y


class ExtractIndicesDataset:
    def __init__(self, dataset, indices: Tuple[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, n: int):
        obj = self.dataset[n]
        return tuple(obj[i] for i in self.indices)


def get_default_image_transform(dim: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        # Переводим цвета пикселей в отрезок [-1, 1] афинным преобразованием, изначально они в отрезке [0, 1]
        transforms.Normalize(tuple(0.5 for _ in range(dim)), tuple(0.5 for _ in range(dim)))
    ])


default_image_transform = get_default_image_transform(3)


def get_cifar_10_dataset(root='./cifar10'):
    cifar_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True,
                                                 transform=default_image_transform, )

    dataset = ExtractIndicesDataset(cifar_dataset, indices=(0,))
    return dataset


def get_mnist_dataset(root='./mnist'):
    mnist_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True,
                                               transform=get_default_image_transform(1))

    dataset = ExtractIndicesDataset(mnist_dataset, indices=(0,))
    return dataset
