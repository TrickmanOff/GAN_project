from typing import Tuple, List, Optional, Union

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms


"""
Датасеты могут быть двух типов:
1. Элемент - число или тензор. В этом случае элемент рассматривается как x в GAN
2. Элемент - tuple длины 2. В этом случае 1-ый элемент tuple - x, 2-й - y (условие)
y - либо число, либо тензор, либо tuple с числами/тензорами
"""

def collate_fn(els_list: List[Union[Tuple, int, torch.Tensor]]):
    if isinstance(els_list[0], tuple):
        return tuple(collate_fn(list(a)) for a in zip(*els_list))
    elif isinstance(els_list[0], int):
        return torch.Tensor(els_list)
    elif isinstance(els_list[0], torch.Tensor):
        return torch.stack(els_list)
    else:
        raise RuntimeError


def move_batch_to(batch, device):
    if isinstance(batch, tuple):
        return tuple(move_batch_to(subbatch, device) for subbatch in batch)
    else:
        return batch.to(device)


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


class PhysicsDataset(torch.utils.data.Dataset):
    """
    one element: (energy deposit, (point, momentum))
    """
    def __init__(self, energy: torch.Tensor, point: torch.Tensor, momentum: torch.Tensor) -> None:
        self.energy = energy
        self.point = point
        self.momentum = momentum

    def __getitem__(self, idx: int) -> tuple:
        return (self.energy[idx], (self.point[idx], self.momentum[idx]))

    def __len__(self) -> int:
        return self.energy.shape[0]


def get_physics_dataset(path: str) -> torch.utils.data.Dataset:
    data_train = np.load(path)

    np.random.seed(42)
    ind_arr = np.random.choice(np.arange(len(data_train['EnergyDeposit'])),
                               size=len(data_train['EnergyDeposit']) // 2)
    #     energy   = torch.tensor(data_train['EnergyDeposit'][ind_arr].reshape(-1, 900)).float()
    energy = torch.tensor(data_train['EnergyDeposit'][ind_arr]).float()
    energy = torch.log1p(energy)  # !
    energy = torch.permute(energy, dims=(0, 3, 1, 2))
    point = torch.tensor(data_train['ParticlePoint'][:, :2][ind_arr]).float()
    momentum = torch.tensor(data_train['ParticleMomentum'][ind_arr]).float()

    return PhysicsDataset(energy, point, momentum)
