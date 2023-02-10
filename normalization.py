from abc import abstractmethod
from typing import TypeVar, Type, Dict, Any

import torch
from torch import nn


"""
тут можно ещё подумать над удобной реализацией
"""


class ModuleNotSupported(Exception):
    pass


class Normalizer(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        # кидаем исключение ModuleNotSupported, если не поддерживаем модуль
        self.module = module

    @abstractmethod
    def update_stats(self) -> None:
        """
        Предполагаем, что эта функция вызывается после обновления весов
        """
        pass

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass

    def train(self, mode: bool = True) -> 'Normalizer':
        self.module.train(mode)
        return self

    def eval(self) -> 'Normalizer':
        self.module.eval()
        return self


T = TypeVar('T', bound=nn.Module)


class ClippingNormalizer(Normalizer):
    def __init__(self, module: T, clip_c: float) -> None:
        """
        Clip in range [-clip_c, clip_c]
        """
        super().__init__(module)
        self.clip_c = nn.Parameter(torch.Tensor([clip_c]))
        self.clip_c.requires_grad = False

        if type(module) not in [nn.Linear]:
            raise ModuleNotSupported

    def update_stats(self) -> None:
        with torch.no_grad():
            for param in self.module.parameters():
                param.clip_(min=-self.clip_c, max=self.clip_c)

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.module(X, *args, **kwargs)


class SpectralNormalizer(Normalizer):
    def __init__(self, module: T) -> None:
        super().__init__(module)

        if isinstance(module, nn.Linear):
            self.weight_matrix_fn = lambda: module.weight.data
        elif isinstance(module, nn.Conv2d):
            self.weight_matrix_fn = lambda: module.weight.data.reshape(
                (module.weight.data.shape[0], -1))
        else:
            raise ModuleNotSupported

        self.weight_matrix_shape = self.weight_matrix_fn().shape
        self.u = nn.Parameter(2*torch.rand(self.weight_matrix_shape[0], 1, requires_grad=False)-1)
        self.v = nn.Parameter(2*torch.rand(self.weight_matrix_shape[1], 1, requires_grad=False)-1)
        self.u.requires_grad = False
        self.v.requires_grad = False

    def _sync_device(self) -> None:
        device = self.weight_matrix_fn().device
        self.u = self.u.to(device)
        self.v = self.v.to(device)

    def calc_singular_value_approx(self) -> float:
        self._sync_device()
        return (self.u.T @ self.weight_matrix_fn() @ self.v).item()

    def update_stats(self) -> None:
        self._sync_device()
        with torch.no_grad():
            W = self.weight_matrix_fn()
            self.v.data = W.T @ self.u / torch.linalg.norm(W.T @ self.u)
            self.u.data = W @ self.v / torch.linalg.norm(W @ self.v)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        sing_approx = self.calc_singular_value_approx()
        return self.module(X) / sing_approx


def apply_normalization(module: nn.Module,
                        normalizer_cls: Type[Normalizer], *normalizer_args,
                        **normalizer_kwargs) -> nn.Module:
    try:
        return normalizer_cls(module, *normalizer_args, **normalizer_kwargs)
    except ModuleNotSupported:
        pass

    for name, submodule in module.named_children():
        normalized_submodule = apply_normalization(submodule, normalizer_cls, *normalizer_args,
                                                   **normalizer_kwargs)
        setattr(module, name, normalized_submodule)

    return module


def update_normalizers_stats(module: nn.Module):
    """
    Call when after a change of weights
    """
    for name, submodule in module.named_modules():
        if isinstance(submodule, Normalizer):
            submodule.update_stats()
