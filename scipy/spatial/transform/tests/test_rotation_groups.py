from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform._rotation_groups import rotation_group
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.constants import golden as phi


tol = 1E-12


def _register(P, Q):
    distance_matrix = cdist(P, Q)
    matching = linear_sum_assignment(distance_matrix)
    return distance_matrix[matching].sum()


def _generate_pyramid(n):
    thetas = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    pyramid = np.vstack([np.cos(thetas), np.sin(thetas), np.zeros(n)]).T
    pyramid = np.concatenate((pyramid, [[0, 0, 1]]))
    return pyramid


def _generate_prism(n):
    thetas = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    bottom = np.vstack([np.cos(thetas), np.sin(thetas), -np.ones(n)]).T
    top = np.vstack([np.cos(thetas), np.sin(thetas), +np.ones(n)]).T
    return np.concatenate((bottom, top))


def _generate_icosahedron():
    x = np.array([[0, -1, -phi],
                  [0, -1, +phi],
                  [0, +1, -phi],
                  [0, +1, +phi]])
    return np.concatenate([np.roll(x, i, axis=1) for i in range(3)])


def _generate_octahedron():
    return np.array([[-1, 0, 0], [+1, 0, 0], [0, -1, 0],
                     [0, +1, 0], [0, 0, -1], [0, 0, +1]])


def _generate_tetrahedron():
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])


def test_cyclic():
    for n in range(1, 13):
        P = _generate_pyramid(n)
        for g in rotation_group('C%d' % n):
            assert _register(P, g.apply(P)) < tol


def test_dicyclic():
    for n in range(1, 13):
        P = _generate_prism(n)
        for g in rotation_group('D%d' % n):
            assert _register(P, g.apply(P)) < tol


def test_tetrahedral():
    P = _generate_tetrahedron()
    for g in rotation_group('T'):
        assert _register(P, g.apply(P)) < tol


def test_icosahedral():
    P = _generate_icosahedron()
    for g in rotation_group('I'):
        g = Rotation.from_quat(g.as_quat())
        assert _register(P, g.apply(P)) < tol


def test_octahedral():
    P = _generate_octahedron()
    for g in rotation_group('O'):
        assert _register(P, g.apply(P)) < tol
