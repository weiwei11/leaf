import numpy as np
import pytest
from leaf.metrics.segmentation import AveragePrecision


def test_average_precision():
    mask_pred = np.array([[1, 0], [0, 1]], dtype=np.bool_)
    mask_gt = np.array([[1, 1], [1, 1]], dtype=np.bool_)
    mask_pred1 = np.array([[1, 1], [0, 1]], dtype=np.bool_)
    mask_gt1 = np.array([[1, 1], [1, 1]], dtype=np.bool_)
    mask_pred2 = np.array([[1, 1], [1, 1]], dtype=np.bool_)
    mask_gt2 = np.array([[1, 1], [1, 1]], dtype=np.bool_)

    ap = AveragePrecision()
    ap(mask_pred, mask_gt)
    ap(mask_pred1, mask_gt1)
    ap(mask_pred2, mask_gt2)
    result = ap.summarize()
    assert result == 0.6666666666666666
