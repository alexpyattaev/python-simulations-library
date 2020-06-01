import numpy as np
from scipy.stats import norm as normal_distribution


def event_signature_gaussian(W: int, eps=1e-5, normalize=1.0, bounds=(-1, 1)) -> np.ndarray:
    """
    Generate the shape of gaussian pdf in a fixed length window, with max normalized to a value.

    The tails are configured with eps parameter. Normalization can be disabled.
    :param W: support window length
    :param eps: value at the tail end
    :param bounds: the boundary of the generation. By default, whole curve will be made, corresponding to (-1..1) bounds
    :param normalize: a value to normalize the peak against. If None or 0 normalization is disabled.
    :return: samples
    """
    q = normal_distribution.ppf(1-eps)
    x = np.linspace(q * bounds[0], q * bounds[1], W)
    z = normal_distribution.pdf(x)
    if not normalize:
        return z
    else:
        return z * normalize / np.max(z)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure()
    for w in [30, 60, 100]:
        plt.plot(event_signature_gaussian(w), label=f'norm pdf, {w}')
    s = np.concatenate([event_signature_gaussian(60, bounds=[-1, 0]), event_signature_gaussian(20, bounds=[0, 1])])
    assert len(s) == 60+20
    plt.plot(s, label=f'norm imbalanced')

    plt.legend()
    plt.show(block=True)
