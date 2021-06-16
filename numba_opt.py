import functools
__author__ = 'Alex Pyattaev'
import os

import numpy as np

try:

    if 'NO_NUMBA' in os.environ:
        print("Numba disabled by environment variable")
        raise ImportError("Numba is disabled")

    import numba
    import numba.experimental
    numba_available = True
    jit_hardcore = functools.partial(numba.jit, nopython=True, nogil=True, cache=True)
    njit = numba.njit(nogil=True, cache=True)
    njit_nocache = numba.njit(nogil=True, cache=False)
    jitclass = numba.experimental.jitclass
    int64 = numba.int64
    int32 = numba.int32
    int16 = numba.int16
    double = numba.double
    complex128 = numba.complex128
    from numba.typed import List as TypedList
    vectorize = numba.vectorize
except ImportError:
    TypedList = list
    numba = None
    numba_available = False
    int64 = int
    int32 = int
    int16 = int
    double = float
    complex128 = complex
    vectorize = np.vectorize

    # define stub functions for Numba placeholders
    def njit(f, *args, **kwargs):
        return f

    def njit_nocache(f, *args, **kwargs):
        return f

    def jitclass(c, *args, **kwargs):
        def x(cls):
            return cls
        return x

    def jit_hardcore(f, *args, **kwargs):
        return f


