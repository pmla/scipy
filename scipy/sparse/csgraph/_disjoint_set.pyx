"""
Disjoint set data structure
"""

import numpy as np
cimport numpy as np
cimport cython


cdef class DisjointSet:
    """ Disjoint set data structure for incremental connectivity queries.

    .. versionadded: 1.6.0

    Attributes
    ----------
    n_nodes : int
        The number of nodes in the set.
    n_components : int
        The number of components/subsets.

    Methods
    -------
    union
    find

    Notes
    -----
    This class implements the disjoint set [1]_, also known as the *union-find*
    data structure. The *find* method implements the *path compression*
    variant. The *union* method implements the *union by size* variant.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    See Also
    --------
    connected_components : Analyze the connected components of a sparse graph.

    Examples
    --------
    >>> from scipy.sparse.csgraph import DisjointSet

    Initialize a disjoint set with 4 nodes:

    >>> dis = DisjointSet(4)

    Merge some subsets:

    >>> dis.union(0, 1)
    True
    >>> dis.union(2, 3)
    True
    >>> dis.union(3, 3)
    False

    Find a root node:

    >>> dis.find(1)
    0

    """
    cdef:
        readonly np.npy_intp n_nodes
        readonly np.npy_intp n_components
        readonly dict _sizes
        readonly dict _parents

    def __init__(DisjointSet self):
        self.n_nodes = 0
        self.n_components = 0
        self._sizes = {}
        self._parents = {}

    def find(DisjointSet self, np.intp_t x):
        """Find the root node of `x`.

        Parameters
        ----------
        x : int
            Input node.

        Returns
        -------
        root : int
            Root node of `x`.
        """
        if x not in self._parents:
            self._sizes[x] = 1
            self._parents[x] = x
            self.n_nodes += 1
            self.n_components += 1

        parents = self._parents
        parent = parents[x]
        while parent != parents[parent]:
            parent = parents[parent]
        parents[x] = parent
        return parent

    def union(DisjointSet self, np.intp_t a, np.intp_t b):
        """Merge the subsets of `a` and `b`.

        The smaller subset (the child) is merged into the the larger subset
        (the parent). If the subsets are of equal size, the parent is
        determined by subset root with the smallest index.

        Parameters
        ----------
        a, b : int
            Node indices to merge.

        Returns
        -------
        merged : bool
            `True` if `a` and `b` were in disjoint sets, `False` otherwise.
        """
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return False

        if b < a:
            a, b = b, a
        sizes = self._sizes
        parents = self._parents
        if sizes[a] < sizes[b]:
            parents[a] = b
            sizes[b] += sizes[a]
        else:
            parents[b] = a
            sizes[a] += sizes[b]

        self.n_components -= 1
        return True
