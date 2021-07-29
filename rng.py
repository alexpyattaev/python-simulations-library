from functools import partial
from typing import Union

import numpy as np
import numpy.random as nprandom
from lib.numba_opt import jit_hardcore

_rng = nprandom.RandomState()

_seed = [0, 0]


def set_seed(x=7):
    _seed[0] = x
    _seed[1] = x
    _rng.seed(x)


def make_local_random() -> nprandom.RandomState:
    _seed[1] += 1
    return nprandom.RandomState(_seed[1])


set_seed()

randint = _rng.randint
choice = _rng.choice
uniform = _rng.uniform
sample = partial(_rng.choice, replace=False)
shuffle = _rng.shuffle
normal = _rng.normal
laplace = _rng.laplace
poisson = _rng.poisson
rand = _rng.rand
gamma = _rng.gamma
exponential = _rng.exponential
binomial = _rng.binomial
rayleigh = _rng.rayleigh


def erlang(shape: int = 2, mean: float = 1.0, size=None, rng=_rng) -> Union[float, np.ndarray]:
    """
    Return Erlang-distributed numbers by feeding correct numbers into Gamma distribution generator.
    :param shape: the Erlang distribution k parameter. Should be natural number.
    :param mean: desired mean value
    :param size: shape of returned array, None for single value
    :param rng: custom RNG to use. Will use default RNG if not provided.
    :return: float or array depending on size
    """
    assert isinstance(shape, int)
    assert shape > 0
    return rng.gamma(shape, mean / shape, size)


def test_erlang():
    k = 4  # order
    e = 10  # expectation
    r = k / e  # rate parameter
    zz = erlang(shape=k, mean=e, size=10000)
    assert np.isclose(np.mean(zz), 10, rtol=0.05)
    v = k / (r ** 2)  # variance
    assert np.isclose(np.var(zz), v, rtol=0.1)


@jit_hardcore
def first_value_below(arr, thr):
    for i in arr:
        if i < thr:
            return i
    return thr


def exponential_capped(mean: float, cap_rate=5.0, _fsample=8, rng=_rng) -> float:
    """Capped exponential, i.e. with constrained output
    :param mean: desired mean value
    :param cap_rate: mean*cap_rate defines max value this function will ever return
    :param _fsample: amount of random samples to make in oreder to hit cap rate.
    :param rng: custom RNG to use. Will use default RNG if not provided.
    :returns float value for value requested
    """
    x = rng.exponential(mean, size=_fsample)
    return first_value_below(x, cap_rate * mean)


def toss_coin(p):
    """Toss a coin with a given success probability
    :param p: the probability to use. If p is a vector, returns results of multiple tosses
    :returns: the result of toss as boolean value or array"""
    try:
        return _rng.rand(len(p)) < p
    except TypeError:
        return _rng.rand() < p


def rand_sign():
    return choice((-1, 1))


def random_DAG(size: int = 10, connectivity_pattern=lambda imax: randint(0, imax),
               weight_distribution=lambda: randint(1, 10)) -> np.ndarray:
    """
     Create the connectivity matrix for random DAG

     One can convert into e.g. networkx graph with
     G = nx.convert_matrix.from_numpy_matrix(W, parallel_edges=False, create_using=nx.DiGraph)

    :param size: number of nodes
    :param connectivity_pattern: function defining connectivity of the nodes.
    :param weight_distribution: weights for the values in the matrix
    :return: connectivity matrix for the DAG
    """
    if size < 1:
        raise ValueError('Size must be positive for a graph to be made!')
    W = np.zeros([size, size])
    for i in range(1, size):
        W[i, connectivity_pattern(i)] = weight_distribution()

    return W


def test_random_DAG():
    import networkx as nx
    # import matplotlib.pyplot as plt
    # from networkx.drawing.nx_pydot import graphviz_layout
    con_pattern = lambda imax: min(int(exponential(5)), imax - 1)
    for i in range(5, 50):
        W = random_DAG(i)
        G = nx.convert_matrix.from_numpy_matrix(W, parallel_edges=False, create_using=nx.DiGraph)
        assert nx.is_directed_acyclic_graph(G), "G must be a DAG!"
        assert nx.is_tree(G), "G must be a tree!"

        # plt.figure()
        # nx.draw_networkx(G, pos= graphviz_layout(G, prog="dot"), with_labels=True)   # default spring_layout
        # plt.show()
