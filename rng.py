from abc import ABC, abstractmethod
from functools import partial

from numbers import Number
from typing import Union

import numpy as np

from numpy.random import Generator, default_rng

_seed = [7]


def make_local_random(seed: int = None) -> Generator:
    if seed is not None:
        s = seed
    else:
        _seed[0] += 1
        s = _seed[0]
    return default_rng(s)


_rng: Generator = make_local_random()


def set_seed(x=7):
    global _rng
    _seed[0] = x
    _rng = make_local_random(x)


randint = _rng.integers
choice = _rng.choice
uniform = _rng.uniform
sample = partial(_rng.choice, replace=False)
shuffle = _rng.shuffle
normal = _rng.normal
laplace = _rng.laplace
poisson = _rng.poisson
rand = _rng.random
gamma = _rng.gamma
exponential = _rng.exponential
binomial = _rng.binomial
rayleigh = _rng.rayleigh
geometric = _rng.geometric


def exponential_capped(mean: float, cap_rate=5.0, N=None, rng=_rng) -> Union[float, np.ndarray]:
    """Capped exponential, i.e. with constrained output
    :param mean: desired mean value
    :param cap_rate: mean*cap_rate defines max value this function will ever return
    :param N: amount of random samples to make
    :param rng: custom RNG to use. Will use default RNG if not provided.
    :returns float value for value requested
    """
    return np.minimum(rng.exponential(mean, size=N), cap_rate * mean)


def geometric_capped(mean: float, cap_rate=5.0, N=None, rng=_rng) -> Union[int, np.ndarray]:
    """Capped geometric, i.e. with constrained output
    :param mean: desired mean value
    :param cap_rate: mean*cap_rate defines max value this function will ever return
    :param N: amount of random samples to make
    :param rng: custom RNG to use. Will use default RNG if not provided.
    :returns float value for value requested
    """
    return np.minimum(rng.geometric(1 / mean, size=N), cap_rate * mean)


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


class RandomVar(ABC, Number):
    def __init__(self, seed: int = None):
        self._rng = make_local_random(seed)

    @abstractmethod
    def __float__(self) -> float:
        """Return a float sample"""
        ...

    @abstractmethod
    def __int__(self) -> int:
        """Return an int sample"""
        ...

    @abstractmethod
    def __str__(self) -> float:
        ...

    @abstractmethod
    def __repr__(self) -> float:
        ...

    @property
    def mean(self) -> float:
        raise NotImplementedError()

    @property
    def stdev(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def vector(self, N: int, dtype=float) -> np.ndarray:
        """Return a vector of samples with length N and type dtype"""
        ...

    def __hash__(self):
        raise RuntimeError("Random numbers are not hashable")

    def to_json(self):
        return {"RV": self.__class__.__name__.replace("Random", ""), "mean": self.mean, "stdev": self.stdev}


class Random_Const(RandomVar):
    def __init__(self, mean: float, seed: int = None) -> None:
        RandomVar.__init__(self)
        self._mean = mean

    def __float__(self):
        return self.mean

    def __int__(self):
        return int(self._mean)

    @property
    def mean(self):
        return self._mean

    def vector(self, N, dtype=float):
        return np.full(N, fill_value=self._mean, dtype=dtype)

    def __str__(self) -> str:
        return f"Const({self._mean:.2g})"

    def __repr__(self) -> str:
        return f"Random_Const({self._mean:.2g})"

    @property
    def stdev(self) -> float:
        return 0.0


class Random_Gamma(RandomVar):
    def __init__(self, mean: float, stdev: float, cap_rate: float = 5.0, seed: int = None) -> None:
        # https://en.wikipedia.org/wiki/Gamma_distribution
        #
        # ... and keeping in mind that Python calls (alpha, beta)
        # what wikipedia calls (k, theta)
        #
        # (1) mean = k * theta
        # (2) variance = k * (theta ** 2)
        #
        # keeping in mind variance = stdev**2
        #
        # dividing eq. (2) by (1) we get:
        # theta = stdev**2 / mean
        #
        # ... and then from (1):
        # k = mean / theta
        if mean==0:
            raise ValueError("Gamma distribution can not have zero mean")
        RandomVar.__init__(self, seed)
        self._mean = mean
        self._stdev = stdev
        self.cap_rate = cap_rate
        variance = stdev ** 2
        theta = variance / mean
        k = mean / theta

        assert k > 0 and theta > 0
        self._k = k
        self._theta = theta

    def __str__(self) -> str:
        return f"Gamma({self._mean:.2g} +- {self._stdev:.2g})"

    def __repr__(self) -> str:
        return f"Random_Gamma({self._mean:.2g},{self._stdev:.2g})"

    def __float__(self):
        return np.clip(self._rng.gamma(self._k, self._theta),
                       a_min=self._mean - self.cap_rate * self._stdev,
                       a_max=self._mean + self.cap_rate * self._stdev)

    def __int__(self):
        return int(np.clip(self._rng.gamma(self._k, self._theta),
                           a_min=self._mean - self.cap_rate * self._stdev,
                           a_max=self._mean + self.cap_rate * self._stdev))

    def vector(self, N, dtype=float):
        return np.array(np.clip(self._rng.gamma(self._k, self._theta, size=N),
                                a_min=self._mean - self.cap_rate * self._stdev,
                                a_max=self._mean + self.cap_rate * self._stdev,
                                ), dtype=dtype)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def stdev(self) -> float:
        return self._stdev


class Random_Expo(RandomVar):
    def __init__(self, mean: float, cap_rate=5.0, seed: int = None) -> None:
        RandomVar.__init__(self, seed)
        self._mean = mean
        self._lambda = 1 / self._mean
        self.cap_rate = cap_rate

    def __str__(self) -> str:
        return f"Exp({self._mean:.3g})"

    def __repr__(self) -> str:
        return f"Random_Expo({self._mean})"

    def __float__(self):
        return exponential_capped(mean=self._mean, cap_rate=self.cap_rate, rng=self._rng)

    def __int__(self):
        return geometric_capped(mean=self.mean, cap_rate=self.cap_rate, rng=self._rng)

    @property
    def stdev(self) -> float:
        return self._mean

    def vector(self, N: int, dtype=float) -> np.ndarray:
        return exponential_capped(mean=self._mean, cap_rate=self.cap_rate, N=N, rng=self._rng) if dtype == float else (
            geometric_capped(mean=self._mean, cap_rate=self.cap_rate, N=N, rng=self._rng))

    @property
    def mean(self) -> float:
        return self._mean


# noinspection PyArgumentList
def test_rand_gamma():
    rv = Random_Gamma(mean=1.0, stdev=1.0, cap_rate=3.0)
    vals = rv.vector(1000)
    assert 0 < vals.min() < 0.1
    assert 3.95 < vals.max() <= 4.0


def test_rand_exp():
    M = 5.0
    R = 10.0
    rtol = 0.1
    N = 10000
    rv = Random_Expo(mean=M, cap_rate=R)

    # test individual exponential sampling
    vals = np.array([exponential_capped(M, cap_rate=R) for _ in range(N)])
    assert (vals <= M * R).all()
    assert np.isclose(np.mean(vals), M, rtol=rtol)
    assert np.isclose(np.var(vals), rv.stdev ** 2, rtol=rtol * 3)

    # test vector exponential sampling
    vals = rv.vector(N, dtype=float)
    assert (vals <= M * R).all()
    assert np.isclose(np.mean(vals), M, rtol=rtol)
    assert np.isclose(np.var(vals), rv.stdev ** 2, rtol=rtol * 3)

    # test geometric sampling
    vals = rv.vector(N, dtype=int)
    assert (vals <= M * R).all()
    assert np.isclose(np.mean(vals), M, rtol=rtol)
    assert np.isclose(np.var(vals), rv.stdev ** 2, rtol=rtol * 3)

    # test individual exponential sampling
    vals = np.array([float(rv) for _ in range(N)])
    assert (vals <= M * R).all()
    assert np.isclose(np.mean(vals), M, rtol=rtol)
    assert np.isclose(np.var(vals), rv.stdev ** 2, rtol=rtol * 3)


class Random_Uniform(RandomVar):
    def __init__(self, a: float, b: float, seed: int = None) -> None:
        RandomVar.__init__(self, seed)
        assert b >= a
        self._a = a
        self._b = b
        self._delta = b - a

    def __str__(self) -> str:
        return f"Uniform({self._a:.3g}, {self._b:.3g})"

    def __repr__(self) -> str:
        return f"Uniform({self._a:g}, {self._b:g})"

    def __float__(self):
        return uniform(self._a, self._b)

    def __int__(self):
        return randint(round(self._a), round(self._b))

    @property
    def stdev(self) -> float:
        # return sqrt(1./12) * self._delta
        return 0.28867513459481287 * self._delta

    def vector(self, N: int, dtype=float) -> np.ndarray:
        return uniform(self._a, self._b, size=N) if dtype == float else randint(round(self._a), round(self._b), size=N)

    @property
    def mean(self) -> float:
        return (self._a + self._b) / 2


def test_erlang():
    k = 4  # order
    e = 10  # expectation
    r = k / e  # rate parameter
    zz = erlang(shape=k, mean=e, size=10000)
    assert np.isclose(np.mean(zz), 10, rtol=0.05)
    v = k / (r ** 2)  # variance
    assert np.isclose(np.var(zz), v, rtol=0.1)


# noinspection PyArgumentList
def toss_coin(p):
    """Toss a coin with a given success probability
    :param p: the probability to use. If p is a vector, returns results of multiple tosses
    :returns: the result of toss as boolean value or array"""
    try:
        return rand(len(p)) < p
    except TypeError:
        return rand() < p


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
    for i in range(5, 50):
        W = random_DAG(i, connectivity_pattern=lambda imax: min(int(exponential(5)), imax - 1))
        G = nx.convert_matrix.from_numpy_matrix(W, parallel_edges=False, create_using=nx.DiGraph)
        assert nx.is_directed_acyclic_graph(G), "G must be a DAG!"
        assert nx.is_tree(G), "G must be a tree!"

        # plt.figure()
        # nx.draw_networkx(G, pos= graphviz_layout(G, prog="dot"), with_labels=True)   # default spring_layout
        # plt.show()


def select_weighted(weights):
    """Selects option based on given sample probabilities/weights/biases"""
    wcs = np.cumsum(weights, dtype=float)
    return np.searchsorted(wcs / wcs[-1], uniform())
