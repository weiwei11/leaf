# Author: weiwei

from . import model, model_diameter, pose_list, K
from leaf.metrics.metric import Compose
from leaf.metrics.sixd import Projection2d, ADD, Cmd, ADDAUC
import numpy as np


def test_compose(model, model_diameter, pose_list, K):
    pose_metric = Compose('pose_metric',
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
