# Author: weiwei

import numpy as np

from leaf.metrics.metric import BaseMetric
from leaf.metrics.functional.sixd import projection_2d, add, cm_degree, add_error, add_auc


class Projection2d(BaseMetric):
    """
    2D projection

    :param name: name of the metric
    :param model: shape (N, 3), 3D points cloud of object
    :param threshold: default is 5 pixel
    :return: float

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    >>> model_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> proj_metric = Projection2d(model=model_xyz, threshold=5)
    >>> proj_metric(pose_pred, pose_target, K)
    >>> proj_metric.summarize()
    1.0
    """

    def __init__(self, name='Projection2d', model=None, threshold=5):

        super().__init__(name)
        self.model = model
        self.threshold = threshold

    def __call__(self, predict_pose, target_pose, K):
        result = projection_2d(predict_pose, target_pose, K, self.model, self.threshold)
        self.result_list.append(result)


class ADD(BaseMetric):
    """
    ADD

    :param name: name of the metric
    :param model: shape (N, 3), 3D points cloud of object
    :param symmetric: whether the object is symmetric or not
    :param threshold: distance threshold, 'threshold' and 'model_diameter percentage' is not compatible
    :param diameter: the diameter of object
    :param percentage: percentage of model diameter
    :return: float

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> model_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> model_diameter = np.sqrt(3)
    >>> add_metric = ADD(model=model_xyz, symmetric=False, threshold=None, diameter=model_diameter, percentage=0.1)
    >>> add_metric(pose_pred, pose_target)
    >>> add_metric.summarize()
    1.0
    """
    def __init__(self, name='ADD', model=None, symmetric=False, threshold=None, diameter=None, percentage=0.1):
        super().__init__(name)
        self.model = model
        self.symmetric = symmetric
        self.threshold = threshold if threshold is not None else diameter * percentage

    def __call__(self, predict_pose, target_pose):
        result = add(predict_pose, target_pose, self.model, self.symmetric, self.threshold)
        self.result_list.append(result)


class Cmd(BaseMetric):
    """
    Degree and cm

    :param name: name of the metric
    :param cm_threshold: unit is centimeter
    :param degree_threshold:
    :param unit_scale: scale for meter
    :return: float

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_pred1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2]])
    >>> pose_target1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> cmd_metric = Cmd(cm_threshold=5, degree_threshold=5, unit_scale=1.0)
    >>> cmd_metric(pose_pred, pose_target)
    >>> cmd_metric(pose_pred1, pose_target1)
    >>> cmd_metric.summarize()
    0.5
    """
    def __init__(self, name='Cmd', cm_threshold=5, degree_threshold=5, unit_scale=1.0):
        super().__init__(name)
        self.cm_threshold = cm_threshold
        self.degree_threshold = degree_threshold
        self.unit_scale = unit_scale

    def __call__(self, predict_pose, target_pose):
        pred = predict_pose.copy()
        target = target_pose.copy()
        pred[:3, 3] = pred[:3, 3] * self.unit_scale
        target[:3, 3] = target[:3, 3] * self.unit_scale
        result = cm_degree(pred, target, self.cm_threshold, self.degree_threshold)
        self.result_list.append(result)


class ADDAUC(BaseMetric):
    """
    ADD AUC

    :param name: name of the metric
    :param model: shape (N, 3), 3D points cloud of object
    :param max_threshold: max error threshold, so threshold is [0, max]
    :param unit_scale: scale for meter unit
    :param symmetric: whether the object is symmetric or not
    :return:

    >>> import numpy as np
    >>> model_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_pred1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2]])
    >>> pose_target1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_pred2 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 3]])
    >>> pose_target2 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> add_auc_metric = ADDAUC(model=model_xyz, max_threshold=0.1, unit_scale=1.0, symmetric=False)
    >>> add_auc_metric(pose_pred, pose_target)
    >>> add_auc_metric(pose_pred1, pose_target1)
    >>> add_auc_metric(pose_pred2, pose_target2)
    >>> add_auc_metric.summarize()
    0.3333333333333333
    """
    def __init__(self, name='ADD_AUC', model=None, max_threshold=None, unit_scale=1.0, symmetric=False):
        super().__init__(name)
        self.model = model
        self.max_threshold = max_threshold
        self.unit_scale = unit_scale
        self.symmetric = symmetric

    def __call__(self, predict_pose, target_pose):
        result = add_error(predict_pose, target_pose, self.model, self.symmetric)
        self.result_list.append(result)

    def summarize(self):
        result = add_auc(self.result_list, self.max_threshold, self.unit_scale)
        return result


if __name__ == '__main__':
    import doctest
    doctest.testmod()
