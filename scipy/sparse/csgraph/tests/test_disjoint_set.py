import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.sparse.csgraph import DisjointSet
import string


def generate_random_token():
    k = len(string.ascii_letters)
    tokens = list(np.arange(k, dtype=int))
    tokens += list(np.arange(k, dtype=float))
    tokens += list(string.ascii_letters)
    tokens += [None for i in range(k)]
    rng = np.random.RandomState(seed=0)

    while 1:
        size = rng.randint(1, 3)
        node = rng.choice(tokens, size)
        if size == 1:
            yield node[0]
        else:
            yield tuple(node)


def get_nodes(n):
    nodes = set()
    for node in generate_random_token():
        if node not in nodes:
            nodes.add(node)
            if len(nodes) >= n:
                break
    return list(nodes)


@pytest.mark.parametrize("n", [10, 100])
def test_linear_union_sequence(n):
    nodes = get_nodes(n)
    dis = DisjointSet(nodes)
    assert dis.n_components == n

    for i in range(n - 1):
        assert not dis.connected(nodes[i], nodes[i + 1])
        assert dis.merge(nodes[i], nodes[i + 1])
        assert dis.connected(nodes[i], nodes[i + 1])
        assert dis.n_components == n - 1 - i

    assert nodes == list(dis)
    roots = [dis[i] for i in nodes]
    assert all([nodes[0] == r for r in roots])
    assert not dis.merge(nodes[0], nodes[-1])


@pytest.mark.parametrize("n", [10, 100])
def test_self_unions(n):
    nodes = get_nodes(n)
    dis = DisjointSet(nodes)

    for x in nodes:
        assert dis.connected(x, x)
        assert not dis.merge(x, x)
        assert dis.connected(x, x)
    assert dis.n_components == len(nodes)

    assert nodes == list(dis)
    roots = [dis[x] for x in nodes]
    assert nodes == roots


@pytest.mark.parametrize("order", ["ab", "ba"])
@pytest.mark.parametrize("n", [10, 100])
def test_equal_size_ordering(n, order):
    nodes = get_nodes(n)
    dis = DisjointSet(nodes)

    rng = np.random.RandomState(seed=0)
    indices = np.arange(n)
    rng.shuffle(indices)

    for i in range(0, len(indices), 2):
        a, b = nodes[indices[i]], nodes[indices[i + 1]]
        if order == "ab":
            assert dis.merge(a, b)
        else:
            assert dis.merge(b, a)

        expected = nodes[min(indices[i], indices[i + 1])]
        assert dis[a] == expected
        assert dis[b] == expected


@pytest.mark.parametrize("kmax", [5, 10])
def test_binary_tree(kmax):
    n = 2**kmax
    nodes = get_nodes(n)
    dis = DisjointSet(nodes)
    rng = np.random.RandomState(seed=0)

    for k in 2**np.arange(kmax):
        for i in range(0, n, 2 * k):
            r1, r2 = rng.randint(0, k, size=2)
            a, b = nodes[i + r1], nodes[i + k + r2]
            assert not dis.connected(a, b)
            assert dis.merge(a, b)
            assert dis.connected(a, b)

        assert nodes == list(dis)
        roots = [dis[i] for i in nodes]
        expected_indices = np.arange(n) - np.arange(n) % (2 * k)
        expected = [nodes[i] for i in expected_indices]
        assert roots == expected


def test_node_not_present():
    nodes = get_nodes(n=10)
    dis = DisjointSet(nodes)

    with assert_raises(KeyError):
        dis["dummy"]

    with assert_raises(KeyError):
        dis.merge(nodes[0], "dummy")

    with assert_raises(KeyError):
        dis.connected(nodes[0], "dummy")


def test_contains():
    nodes = get_nodes(n=10)
    dis = DisjointSet(nodes)
    for x in nodes:
        assert x in dis

    assert "dummy" not in dis
