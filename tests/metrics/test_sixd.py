# Author: weiwei

from leaf.metrics.sixd import Projection2d, ADD, Cmd, ADDAUC
from . import model, model_diameter, pose_list, K


def test_projection2d(model, pose_list, K):
    proj_metric = Projection2d(model=model, threshold=5)
    for pose_pred, pose_target in pose_list:
        proj_metric(pose_pred, pose_target, K)
    result = proj_metric.summarize()

    assert result == 0.3333333333333333


def test_add(model, model_diameter, pose_list):
    add_metric = ADD(model=model, symmetric=False, threshold=None, diameter=model_diameter, percentage=0.1)
    for pose_pred, pose_target in pose_list:
        add_metric(pose_pred, pose_target)
    result = add_metric.summarize()

    assert result == 0.3333333333333333


def test_cmd(pose_list):
    cmd_metric = Cmd(cm_threshold=5, degree_threshold=5, unit_scale=1.0)
    for pose_pred, pose_target in pose_list:
        cmd_metric(pose_pred, pose_target)
    result = cmd_metric.summarize()

    assert result == 0.33333333333333333


def test_addauc(model, pose_list):
    add_auc_metric = ADDAUC(model=model, max_threshold=0.1, unit_scale=1.0, symmetric=False)
    for pose_pred, pose_target in pose_list:
        add_auc_metric(pose_pred, pose_target)
    result = add_auc_metric.summarize()

    assert result == 0.3333333333333333
