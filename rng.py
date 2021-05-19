from functools import partial
import numpy as np
import numpy.random as nprandom
import networkx as nx
from debug_log import error


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


def exponential_capped(mean, cap_rate=5.0, _fsample=10, rng=_rng):
    """Capped exponential, i.e. with constrained output"""
    x = rng.exponential(mean, size=_fsample)

    x = x[x < cap_rate * mean]
    try:
        return x[0]
    except IndexError:
        error("Exponential capping failed, consider increasing cap_rate!")
        return cap_rate*mean


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
               weight_distribution=lambda: randint(1, 10)):
    if size < 1:
        raise ValueError('Size must be positive for a graph to be made!')
    W = np.zeros([size, size])
    for i in range(1, size):
        W[i, connectivity_pattern(i)] = weight_distribution()
    G = nx.convert_matrix.from_numpy_matrix(W, parallel_edges=False, create_using=nx.DiGraph)
    return G


def test_random_DAG():
    #import matplotlib.pyplot as plt
    #from networkx.drawing.nx_pydot import graphviz_layout
    con_pattern = lambda imax: min(int(exponential(5)), imax-1)
    for i in range(5, 50):
        G = random_DAG(i)#, connectivity_pattern=con_pattern)
        assert nx.is_directed_acyclic_graph(G), "G must be a DAG!"
        assert nx.is_tree(G), "G must be a tree!"

        #plt.figure()
        #nx.draw_networkx(G, pos= graphviz_layout(G, prog="dot"), with_labels=True)   # default spring_layout
        #plt.show()


