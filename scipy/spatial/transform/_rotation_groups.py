import numpy as np
from . import Rotation
from scipy.constants import golden as phi


_iI = Rotation.from_quat([0.5, 0.5 / phi, 0.5 * phi, 0])
_iO = Rotation.from_quat([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])
_iT = Rotation.from_quat([1, 0, 0, 0])
_w = Rotation.from_quat([0.5, 0.5, 0.5, -0.5])


def _generate_group(generators):

    # Perform a breadth-first search to find group elements of the binary
    # quaternion group.
    tol = 1E-10
    identity = Rotation.from_quat([0, 0, 0, 1])
    queue = [identity] + generators
    visited = np.array([identity.as_quat()])
    while queue:
        node = queue.pop(0)
        mindist = np.min(np.linalg.norm(node.as_quat() - visited, axis=1))
        if mindist > tol:
            visited = np.concatenate((visited, [node.as_quat()]))
            for e in generators:
                queue.append(node * e)

    # Discard half the binary quaternion group to get the unique elements of
    # the rotation group.
    anchor = [1, 10, 100, 1000]
    indices = np.where(np.dot(visited, anchor) > 0)[0]
    visited = visited[indices]

    # Clean up numerical noise for cleaner printing.
    indices = np.where(np.abs(visited) < tol)
    visited[indices] *= 0

    return Rotation.from_quat(visited)


def _generate_cyclic_group(n):

    g = np.zeros((n, 4))
    thetas = np.linspace(0, np.pi, n + 1)[:n]
    g[:, 2] = np.sin(thetas)
    g[:, 3] = np.cos(thetas)
    return Rotation.from_quat(g)


def _generate_dihedral_group(n):

    g = np.zeros((2 * n, 4))
    thetas = np.linspace(0, np.pi, n + 1)[:n]
    g[:n, 2] = np.sin(thetas)
    g[:n, 3] = np.cos(thetas)
    g[-n:, 0] = np.cos(thetas)
    g[-n:, 1] = np.sin(thetas)
    return Rotation.from_quat(g)


_groups = {'C1': _generate_cyclic_group(1),
           'D1': _generate_dihedral_group(1),
           'T': _generate_group([_iT, _w]),
           'O': _generate_group([_iO, _w]),
           'I': _generate_group([_iI, _w])}


def rotation_group(group='C1'):
    """Get a 3D rotation group.

    Parameters
    ----------
    group : string
        The name of the 3D rotation group.  Permissible group names are:
            "I" Icosahedral
            "O": Octahedral
            "T": Tetrahedral
            "Cn": Cyclic
            "Dn": Dihedral
        where 'n' is a non-negative integer, e.g. "C4", "D6".  For the cyclic
        and dicyclic groups, the z-axis is the invariant axis.

    Returns
    -------
    rotation : `Rotation` instance
        Object containing the elements of the rotation group.

    Raises
    ------
    ValueError
        If the group name is not valid.

    Notes
    -----
    This function returns rotation groups only.  The full 3-dimensional groups
    also contain reflections.  See e.g. [Conway]_ for a good description of
    these differences.

    References
    ----------
    .. [Conway] John H Conway and Derek A Smith, On quaternions and octonions,
                2003, AK Peters/CRC Press, ISBN 978-1568811345
    """

    if type(group) != str:
        raise ValueError("group argument must be a string")

    if group in _groups.keys():
        return _groups[group]

    g = group[0]
    digits = group[1:]
    if g not in ["C", "D"] or not len(digits) or not digits.isdigit():
        raise ValueError("group must be one of 'I', 'O', 'T', 'Cn', 'Dn'")

    n = int(digits)
    if n < 1:
        raise ValueError("n must be a positive integer")

    if g == "C":
        _groups[group] = _generate_cyclic_group(n)
    else:
        _groups[group] = _generate_dihedral_group(n)
    return _groups[group]
