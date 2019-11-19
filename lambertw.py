__author__ = 'mishanya'


# http://en.wikipedia.org/wiki/Lambert_W_function
# FB - 201105297
import math
eps = 0.00000001 # max error allowed
def w0(x): # Lambert W function using Newton's method
    w = x
    while True:
        ew = math.exp(w)
        wNew = w - (w * ew - x) / (w * ew + ew)
        if abs(w - wNew) <= eps: break
        w = wNew
    return w