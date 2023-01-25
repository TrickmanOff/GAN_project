import numpy as np
import torch


def ohe_labels(y: torch.Tensor, classes_cnt: int) -> torch.Tensor:
    batch_size = y.shape[0]
    m = torch.zeros(batch_size, classes_cnt).to(y.device)
    m[np.arange(batch_size), y.long()] = 1
    return m


# for physics data
def add_angle_and_norm(points: torch.Tensor) -> torch.Tensor:
    angles = torch.atan2(points[:, 1], points[:, 0])[:, None]
    norms = torch.linalg.norm(points, dim=1)[:, None]
    return torch.concat([points, angles, norms], dim=1)
