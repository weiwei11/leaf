import numpy as np
import pytest
import torch


@pytest.fixture
def model():
    model_xyz = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    return model_xyz


@pytest.fixture
def model_diameter():
    return np.sqrt(3)


@pytest.fixture
def pose_list():
    pose_pred = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1.0]])
    pose_target = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1.0]])
    pose_pred1 = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2.0]])
    pose_target1 = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1.0]])
    pose_pred2 = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 3.0]])
    pose_target2 = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1.0]])

    pose_pred = torch.stack([pose_pred, pose_pred1, pose_pred2])
    pose_target = torch.stack([pose_target, pose_target1, pose_target2])
    return [(pose_pred, pose_target)]


@pytest.fixture
def K():
    return torch.tensor([[320, 0, 320], [0, 320, 240], [0, 0, 1.0]]).unsqueeze(0).repeat(3, 1, 1)
