from abc import abstractmethod
from typing import TypeVar, Type, Dict, Any, Optional

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
        # TODO: кидаем исключение ModuleNotSupported, если не поддерживаем модуль
        self.module = module

    @abstractmethod
    def update_stats(self, **kwargs) -> None:
        """
        Предполагаем, что эта функция вызывается после обновления весов
        """
        pass

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.module(X, *args, **kwargs)

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
        self.register_buffer('clip_c', torch.tensor(clip_c))

        if type(module) not in [nn.Linear]:
            raise ModuleNotSupported

    def update_stats(self, **kwargs) -> None:
        with torch.no_grad():
            for param in self.module.parameters():
                param.clip_(min=-self.clip_c, max=self.clip_c)


# class SpectralNormApproximator:
#     """
#     Класс, вычисляющий оценку спектральной нормы для слоя
#     """
#     def __init__(self, module: T) -> None:
#         if isinstance(module, nn.Linear):
#             self.weight_matrix_fn = lambda: module.weight.data
#         elif isinstance(module, nn.Conv2d):
#             self.weight_matrix_fn = lambda: module.weight.data.reshape(
#                 (module.weight.data.shape[0], -1))
#         else:
#             raise ModuleNotSupported
#
#         self.weight_matrix_shape = self.weight_matrix_fn().shape
#         self.u = nn.Parameter(2*torch.rand(self.weight_matrix_shape[0], 1, requires_grad=False)-1)
#         self.v = nn.Parameter(2*torch.rand(self.weight_matrix_shape[1], 1, requires_grad=False)-1)
#         self.u.requires_grad = False
#         self.v.requires_grad = False
#
#     def _sync_device(self) -> None:
#         device = self.weight_matrix_fn().device
#         self.u = self.u.to(device)
#         self.v = self.v.to(device)
#
#     def step(self) -> None:
#         """
#         improve current approximation
#         """
#         self._sync_device()
#         with torch.no_grad():
#             W = self.weight_matrix_fn()
#             self.v.data = W.T @ self.u / torch.linalg.norm(W.T @ self.u)
#             self.u.data = W @ self.v / torch.linalg.norm(W @ self.v)
#
#     def get_approx(self) -> float:
#         self._sync_device()
#         return (self.u.T @ self.weight_matrix_fn() @ self.v).item()


class SpectralNormalizer(Normalizer):
    """
    Базовая спектральная нормализация
    """
    def __init__(self, module: T, beta: Optional[torch.Tensor] = None) -> None:
        """
        :param module:
        :param beta:
        """
        super().__init__(module)

        if isinstance(module, nn.Linear):
            self.weight_matrix_fn = lambda: module.weight.data
        elif isinstance(module, nn.Conv2d):
            self.weight_matrix_fn = lambda: module.weight.data.reshape(
                (module.weight.data.shape[0], -1))
        else:
            raise ModuleNotSupported

        self.weight_matrix_shape = self.weight_matrix_fn().shape
        u = 2*torch.rand(self.weight_matrix_shape[0], 1) - 1
        v = 2*torch.rand(self.weight_matrix_shape[1], 1) - 1
        self.register_buffer('u', u)
        self.register_buffer('v', v)

        self.beta = beta

    def _sync_device(self) -> None:
        device = self.weight_matrix_fn().device
        self.u = self.u.to(device)
        self.v = self.v.to(device)

    def calc_singular_value_approx(self) -> float:
        self._sync_device()
        return (self.u.T @ self.weight_matrix_fn() @ self.v).item()

    def update_stats(self, **kwargs) -> None:
        self._sync_device()
        with torch.no_grad():
            W = self.weight_matrix_fn()
            self.v.data = W.T @ self.u / torch.linalg.norm(W.T @ self.u)
            self.u.data = W @ self.v / torch.linalg.norm(W @ self.v)

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        sing_approx = self.calc_singular_value_approx()
        output = self.module(X, *args, **kwargs)
        if self.beta is None:  # default strict mode
            return output / sing_approx
        elif sing_approx > self.beta:
            return self.beta * output / sing_approx
        else:
            return output


class WeakSpectralNormalizer(Normalizer):
    def __init__(self, module: T, beta: float, is_trainable_beta: bool = False):
        super().__init__(module)
        self.is_trainable_beta = is_trainable_beta
        if is_trainable_beta:
            self.beta = nn.Parameter(torch.tensor(beta), requires_grad=True)  # one for all layers
        else:
            self.register_buffer('beta', torch.tensor(beta))
        self.module = apply_normalization(self.module, SpectralNormalizer, beta=self.beta)


# https://arxiv.org/pdf/2211.06595v1.pdf
class ABCASNormalizer(Normalizer):
    def __init__(self, module: T, b: float = 4., alpha: float = 0.9999, m_const: float = 0.9) -> None:
        super().__init__(module=module)
        self.b = b  # beta in the paper
        self.r = 0.
        self.register_buffer('dm', torch.tensor(0.))
        self.alpha = alpha
        self.module = apply_normalization(module, SpectralNormalizer)
        self.m_const = m_const

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        m = self.m_const ** self.r
        return self.module(X, *args, **kwargs) * m

    def update_stats(self, disc_real_vals: torch.Tensor, disc_gen_vals: torch.Tensor, **kwargs) -> None:
        super().update_stats()
        dist = (disc_real_vals.max() - disc_gen_vals.min()).item()
        self.dm = self.alpha * self.dm + (1 - self.alpha) * dist
        clbr_dm = self.dm / self.b
        self.r = max(0., clbr_dm / (1 - clbr_dm))


class MultiplyOutputNormalizer(Normalizer):
    def __init__(self, module: T, coef: float = 1., is_trainable_coef: bool = False):
        super().__init__(module)
        self.is_trainable_coef = is_trainable_coef
        if is_trainable_coef:
            self.coef = nn.Parameter(torch.tensor(coef), requires_grad=True)
        else:
            self.register_buffer('coef', torch.tensor(coef))

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.coef * self.module(X, *args, **kwargs)


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


def update_normalizers_stats(module: nn.Module, **kwargs):
    """
    Call when after a change of weights
    """
    for name, submodule in module.named_modules():
        if isinstance(submodule, Normalizer):
            submodule.update_stats(**kwargs)
