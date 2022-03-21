# Author: weiwei
import cv2
import numpy as np

from leaf.matrix_transforms.affine_transform import affine_transform


def transform_image(image, m, new_size, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0):
    new_img = cv2.warpAffine(image, m[:2, :], (new_size[1], new_size[0]), flags=interpolation,
                             borderMode=borderMode, borderValue=borderValue)
    return new_img


def transform_points2d(pts2d, m):
    new_pts2d = affine_transform(pts2d, m)
    return new_pts2d


def transform_bboxes(bboxes, m):
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    new_lt = affine_transform(np.column_stack([x1, y1]), m)
    new_lb = affine_transform(np.column_stack([x1, y2]), m)
    new_rt = affine_transform(np.column_stack([x2, y1]), m)
    new_rb = affine_transform(np.column_stack([x2, y2]), m)

    x1, x2, x3, x4 = new_lt[:, 0], new_lb[:, 0], new_rt[:, 0], new_rb[:, 0]
    y1, y2, y3, y4 = new_lt[:, 1], new_lb[:, 1], new_rt[:, 1], new_rb[:, 1]
    x, y = np.column_stack([x1, x2, x3, x4]), np.column_stack([y1, y2, y3, y4])
    new_bboxes = np.column_stack([np.min(x, axis=-1), np.min(y, axis=-1),
                                  np.max(x, axis=-1), np.max(y, axis=-1)])

    return new_bboxes
