# coding=utf8
from __future__ import absolute_import, division, print_function
import numpy as np
from matplotlib.path import Path


def create_mask_by_landmarks(landmarks, Image):
    """
    create mask use fiducials of Image
    :param landmarks: the 68 landmarks detected using dlib
    :type landmarks np.ndarray
    :param Image: a 3-channel image
    :type Image np.ndarray
    :return:
    """
    # fiducals is 2x68
    landmarks = np.float32(landmarks)
    border_fid = landmarks[:, 0:17]
    face_fid = landmarks[:, 17:]

    c1 = np.array([border_fid[0, 0], face_fid[1, 2]])  # left
    c2 = np.array([border_fid[0, 16], face_fid[1, 7]])  # right
    eye = np.linalg.norm(face_fid[:, 22] - face_fid[:, 25])
    c3 = face_fid[:, 2]
    c3[1] = c3[1] - 0.3 * eye
    c4 = face_fid[:, 7]
    c4[1] = c4[1] - 0.3 * eye

    border = [c1, border_fid, c2, c4, c3]
    border = [item.reshape(2, -1) for item in border]
    border = np.hstack(border)

    M = Image.shape[0]  # row -> y
    N = Image.shape[1]  # col -> x

    y = np.arange(0, M, step=1, dtype=np.float32)
    x = np.arange(0, N, step=1, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    _in, _on = inpolygon(X, Y, border[0, :].T, border[1, :].T)

    mask = np.round(np.reshape(_in | _on, [M, N]))
    mask = 255 * np.uint8(mask)
    mask = np.repeat(np.expand_dims(mask, -1), 3, axis=-1)
    return mask


def inpolygon(xq, yq, xv, yv):
    """
    reimplement inpolygon in matlab
    :type xq: np.ndarray
    :type yq: np.ndarray
    :type xv: np.ndarray
    :type yv: np.ndarray
    """
    # http://blog.sina.com.cn/s/blog_70012f010102xnel.html
    # merge xy and yv into vertices
    vertices = np.vstack((xv, yv)).T
    # define a Path object
    path = Path(vertices)
    # merge X and Y into test_points
    test_points = np.hstack([xq.reshape(xq.size, -1), yq.reshape(yq.size, -1)])
    # get mask of test_points in path
    _in = path.contains_points(test_points)
    # get mask of test_points in path(include the points on path)
    _in_on = path.contains_points(test_points, radius=-1e-10)
    # get the points on path
    _on = _in ^ _in_on
    return _in_on, _on
