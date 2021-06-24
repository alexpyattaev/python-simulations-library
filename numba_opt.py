import functools
__author__ = 'Alex Pyattaev'
import os
from copy import deepcopy
from dataclasses import dataclass, fields, replace

import numpy as np


@dataclass
class deepcopy_compat:
    """Makes it possible to deepcopy dataclasses with numba objects in them. Keep in mind that putting numba instances
    into containers will still crash on deepcopy.

    Numba jitclass needs to specify the 'copy' method for this to work
    """
    def __deepcopy__(self, memodict={}):
        items = {}
        for f in fields(self):
            k = f.name
            t = f.type
            v = getattr(self, k)

            if hasattr(v, "_numba_type_"):  # Special handler for numba jitclasses
                if hasattr(v, "copy"):
                    items[k] = v.copy()
                else:
                    raise ValueError(f"Can not deepcopy {k} of type <{t}>: numba classes need 'copy' method defined to be copyable")
            else:
                items[k] = deepcopy(v, memodict)
        return replace(self, **items)



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


