from functools import partial
import numpy.random as nprandom

from debug_log import error

#TODO: make this nice and proper
_rng = nprandom.RandomState()

_seed = [0, 0]


def set_seed(x=7):
    _seed[0] = x
    _seed[1] = x
    _rng.seed(x)


def make_local_random()->nprandom.RandomState:
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


def exponential_capped(mean, cap_rate=5.0, _fsample=10):
    x = _rng.exponential(mean, size=_fsample)

    Q = x[x < cap_rate * mean]
    if len(Q):
        return Q[0]
    else:
        error("Exponential capping failed, consider increasing cap_rate!")
        return x.min()


rayleigh = _rng.rayleigh


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

