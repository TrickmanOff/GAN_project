from typing import Tuple, List, Optional, Union

import torch
import torchvision
from torch import nn
from torchvision import transforms


"""
Датасеты могут быть двух типов:
1. Элемент - объект, не являющийся tuple. В этом случае элемент рассматривается как x в GAN
2. Элемент - tuple длины 2. В этом случае 1-ый элемент tuple - x, 2-й - y (условие)
"""


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
    def __init__(self, dataset, indices: Union[Tuple[int], int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, n: int):
        obj = self.dataset[n]
        if isinstance(self.indices, int):
            return obj[self.indices]
        else:
            return tuple(obj[i] for i in self.indices)


def get_default_image_transform(dim: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        # Переводим цвета пикселей в отрезок [-1, 1] афинным преобразованием, изначально они в отрезке [0, 1]
        transforms.Normalize(tuple(0.5 for _ in range(dim)), tuple(0.5 for _ in range(dim)))
    ])


default_image_transform = get_default_image_transform(3)


def get_cifar_10_dataset(root='./cifar10', keep_labels: bool = True, kept_labels: Optional[List[int]] = None):
    """
    :param labels: which labels to keep
    """
    cifar_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True,
                                                 transform=default_image_transform, )

    if kept_labels is not None:
        kept_indices = []
        for i in range(len(cifar_dataset)):
            if cifar_dataset[i][1] in kept_labels:
                kept_indices.append(i)

        cifar_dataset = torch.utils.data.Subset(cifar_dataset, kept_indices)

    if not keep_labels:
        cifar_dataset = ExtractIndicesDataset(cifar_dataset, indices=0)
    return cifar_dataset


def get_mnist_dataset(root='./mnist', keep_labels: bool = True):
    mnist_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True,
                                               transform=get_default_image_transform(1))

    if not keep_labels:
        mnist_dataset = ExtractIndicesDataset(mnist_dataset, indices=0)
    return mnist_dataset
