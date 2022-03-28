import numpy as np
import pytest


@pytest.fixture
def model():
    model_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return model_xyz


@pytest.fixture
def model_diameter():
    return np.sqrt(3)


@pytest.fixture
def pose_list():
    pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    pose_pred1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2]])
    pose_target1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    pose_pred2 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 3]])
    pose_target2 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    return [(pose_pred, pose_target), (pose_pred1, pose_target1), (pose_pred2, pose_target2)]


@pytest.fixture
def K():
    return np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]])