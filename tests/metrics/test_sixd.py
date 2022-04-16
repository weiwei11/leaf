# Author: weiwei

from leaf.metrics.sixd import Projection2d, ADD, Cmd, ADDAUC, PoseCompose, MeanTranslationError, MeanRotationError
from . import model, model_diameter, pose_list, K
import numpy as np


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


def test_re(pose_list):
    re_metric = MeanRotationError()
    for pose_pred, pose_target in pose_list:
        re_metric(pose_pred, pose_target)
    result = re_metric.summarize()

    assert result == 0.0


def test_te(pose_list):
    te_metric = MeanTranslationError(unit_scale=1.0)
    for pose_pred, pose_target in pose_list:
        te_metric(pose_pred, pose_target)
    result = te_metric.summarize()

    assert result == 1.0


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


def test_pose_compose(model, model_diameter, pose_list, K):
    pose_metric = PoseCompose('pose_metric', model, False, False,
                              Projection2d('proj2d', model),
                              ADD('add-1', model, diameter=model_diameter, percentage=1),
                              ADD('add-0.1', model, diameter=model_diameter, percentage=0.1),
                              ADD('add-0.05', model, diameter=model_diameter, percentage=0.05),
                              Cmd('cmd5'),
                              ADDAUC('add_auc', model, 0.1)
                              )
    pose_metric.get_params()

    for pose_pred, pose_target in pose_list:
        pose_metric({'predict_pose': pose_pred, 'target_pose': pose_target, 'K': K})
    result = pose_metric.summarize()

    target = {'proj2d': 0.3333333333333333, 'add-1': 0.6666666666666666, 'add-0.1': 0.3333333333333333, 'add-0.05': 0.3333333333333333,
              'cmd5': 0.3333333333333333, 'add_auc': 0.3333333333333333}
    assert result == target

    pose_metric.reset()
    for k, v in pose_metric.summarize().items():
        assert np.isnan(v) or v == 0
