"""
Spatial Transformations (:mod:`scipy.spatial.transform`)
========================================================

.. currentmodule:: scipy.spatial.transform

This package implements various spatial transformations. For now,
only rotations are supported.

Rotations in 3 dimensions
-------------------------
.. autosummary::
   :toctree: generated/

   Rotation
   Slerp
   RotationSpline
"""
from __future__ import division, print_function, absolute_import

from .rotation import Rotation, Slerp
from ._rotation_spline import RotationSpline
from ._rotation_groups import rotation_group

__all__ = ['Rotation', 'Slerp', 'RotationSpline', 'rotation_group']

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
