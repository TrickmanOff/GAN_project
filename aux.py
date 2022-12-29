import numpy as np
import torch


def ohe_labels(y: torch.Tensor, classes_cnt: int) -> torch.Tensor:
    batch_size = y.shape[0]
    m = torch.zeros(batch_size, classes_cnt).to(y.device)
    m[np.arange(batch_size), y.long()] = 1
    return m
