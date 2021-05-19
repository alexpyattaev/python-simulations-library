import functools
__author__ = 'Alex Pyattaev'
import os

try:

    if 'NO_NUMBA' in os.environ:
        raise ImportError

    import numba
    import numba.experimental
    print("Numba support enabled")
    numba_available = True
    jit_hardcore = functools.partial(numba.jit, nopython=True, nogil=True, cache=True)
    jit = functools.partial(numba.jit, forceobj=True, nopython=False, cache=True)
    jitclass = numba.experimental.jitclass
    int64 = numba.int64
    int16 = numba.int16
    double = numba.double
    complex128 = numba.complex128
except ImportError:
    numba = None
    print("Numba support not available")
    numba_available = False
    int64 = int
    int16 = int
    double = float
    complex128 = complex
    #define stub functions for Numba placeholders
    def jit(f, *args, **kwargs):
        return f

    def jitclass(c, *args, **kwargs):
        def x(cls):
            return cls
        return x

    def jit_hardcore(f, *args, **kwargs):
        return f



if __name__ == "__main__":
    class A:
        def __init__(self):
            self.x = 5

        @jit
        def t(self, y: int):
            return self.x + y

    a = A()
    print(a.t(5))
