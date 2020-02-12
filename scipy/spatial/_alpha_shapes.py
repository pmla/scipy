"""
Alpha Shapes Code

.. versionadded:: 1.5.0

"""
import itertools
import numpy as np
from scipy.spatial import Delaunay


__all__ = ['AlphaShapes']


class AlphaShapes:
    """
    AlphaShapes(points, qhull_options=None, rcond=1e-15)

    Alpha shapes in N dimensions.

    .. versionadded:: 1.5.0

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndim)
        Coordinates of points to triangulate
    qhull_options : str, optional
        Additional options to pass to Qhull. See Qhull manual for
        details. Option "Qt" is always enabled.
        Default:"Qbb Qc Qz Qx Q12" for ndim > 4 and "Qbb Qc Qz Q12" otherwise.
        Incremental mode omits "Qz".
    rcond : float, optional
        Cutoff for small singular values for handling degenerate simplices.

    Attributes
    ----------
    points : ndarray of double, shape (npoints, ndim)
        Coordinates of input points.
    simplices : ndarray of ints, shape (nsimplex, ndim+1)
        Indices of the points forming the simplices in the alpha complex.
        For 2-D, the points are oriented counterclockwise.
    equations : ndarray of double, shape (nsimplex, ndim+2)
        [normal, offset] forming the hyperplane equation of the facet
        on the paraboloid
        (see `Qhull documentation <http://www.qhull.org/>`__ for more).
    radii : ndarray of double, shape (nsimplex,)
        Circumradii of the simplices in sorted order.
    circumcenters : ndarray of double, shape (nsimplex, ndim)
        Circumcenters of the simplices.

    Raises
    ------
    QhullError
        Raised when Qhull encounters an error condition, such as
        geometrical degeneracy when options to resolve are not enabled.
    ValueError
        Raised if an incompatible array is given as input.

    Notes
    -----
    The tessellation is computed using the Qhull library
    `Qhull library <http://www.qhull.org/>`__.

    .. note::

       Unless you pass in the Qhull option "QJ", Qhull does not
       guarantee that each input point appears as a vertex in the
       Delaunay triangulation. Omitted points are listed in the
       `coplanar` attribute.

    Examples
    --------
    Alpha complex of a set of points:

    >>> import numpy as np
    >>> from scipy.spatial import AlphaShapes
    >>> rng = np.random.RandomState(seed=0)
    >>> points = rng.normal(size=(1000, 2))
    >>> alpha = AlphaShapes(points)

    The simplices are ordered by circumradius:

    >>> alpha.radii
    array([1.12687281e-02 1.18887521e-02 1.39000778e-02 ... 2.91044209e+00
           3.88209547e+00 1.34510564e+01], dtype=float64)

    We can plot the alpha complex at a chosen radius threshold:

    >>> import matplotlib.pyplot as plt
    >>> from matplotlib import collections
    >>> facets = alpha.get_surface_facets(0.7)
    >>> fig, ax = plt.subplots()
    >>> ax.scatter(points.T[0], points.T[1])
    >>> lc = collections.LineCollection(points[facets])
    >>> ax.add_collection(lc)
    >>> plt.show()
    """
    def __init__(self, points, qhull_options=None, rcond=1e-15):
        delaunay = Delaunay(points, qhull_options=qhull_options)
        self.points = delaunay.points
        self.simplices = delaunay.simplices
        self.equations = delaunay.equations

        self._calculate_circumcenters(rcond)
        self._calculate_circumradii()
        self._calculate_surface_intervals()

    def _calculate_circumcenters(self, rcond):
        tetrahedra = self.points[self.simplices]

        # build overdetermined system of linear equations (Ax=b)
        gramian = np.einsum('lij,lkj->lik', tetrahedra, tetrahedra)
        A = 2 * (gramian - np.roll(gramian, 1, axis=1))

        squared_norms = np.einsum('ij,ij->i', self.points, self.points)
        squared_tet_norms = squared_norms[self.simplices]
        b = squared_tet_norms - np.roll(squared_tet_norms, 1, axis=1)

        # handle rank deficiencies with Moore-Penrose pseudoinverse
        penrose = np.linalg.pinv(A, rcond=rcond)
        tp = np.einsum('lji,ljk->lik', tetrahedra, penrose)
        self.circumcenters = np.einsum('lij,lj->li', tp, b)

    def _calculate_circumradii(self):
        # calculate circumradii of each tetrahedron
        deltas = self.circumcenters - self.points[self.simplices[:, 0]]
        self.radii = np.linalg.norm(deltas, axis=1)

        # now sort the radii, simplices, and circumcenters by radius
        indices = np.argsort(self.radii)
        self.radii = self.radii[indices]
        self.circumcenters = self.circumcenters[indices]
        self.simplices = self.simplices[indices]
        self.equations = self.equations[indices]

    def _calculate_surface_intervals(self):
        dim = self.points.shape[1]
        nsimplices = len(self.simplices)

        # decompose simplices into facets
        facets = []
        for indices in itertools.combinations(range(dim + 1), dim):
            facets.append(self.simplices[:, indices])
        facets = np.hstack(facets).reshape((nsimplices * (dim + 1), dim))
        facets = np.sort(facets)

        # calculate intervals in which facets are surface facets
        unique_facets, start, inverse = np.unique(facets, axis=0,
                                                  return_index=True,
                                                  return_inverse=True)
        _, end = np.unique(inverse[::-1], return_index=True)
        end = len(inverse) - 1 - end

        # facets which only appear once in list can only be surface facets
        indices = np.where(start == end)[0]
        end[indices] = nsimplices * (dim + 1)

        self._start = start
        self._end = end
        self._unique_facets = unique_facets

    def get_surface_facets(self, alpha):
        dim = self.points.shape[1]
        indices = np.where(self.radii <= alpha)[0]
        if not len(indices):
            return np.array([])

        index = (np.max(indices) + 1) * (dim + 1)
        indices = np.where((self._start < index) & (self._end >= index))[0]
        return self._unique_facets[indices]
