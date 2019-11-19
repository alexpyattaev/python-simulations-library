import collections
from numbers import Number

from lib import integer_types, numeric_types

__author__ = 'Alex Pyattaev'

trantab = dict(map(lambda x: (ord(x), None), '"\'\\%'))


def remove_matlab_unsafe_chars(s):
    """Removes all known Matlab-unsafe characters from valid ASCII strings

    If those are let through, they mess up with the Matlab parser in a bad way.
    Naturally, many more characters may be harmful (e.g. non-printalbes), but those are largely out of scope."""
    safe = s.translate(trantab)
    return safe


def matlab_format(name, val, cells=False):
    if isinstance(val, Number):
        return name + " = " + str(val) + ";"
    elif isinstance(val, str):
        return name + "='" + val + "';"
    elif isinstance(val, collections.Iterable):
        if cells:
            border = "{}"
        else:
            border = "[]"

        s = name + "=" + border[0]
        for i in val:
            if isinstance(i, str):
                if not cells:
                    raise ValueError("Strings should be stored in cell arrays in Matlab!")
                s += "'{}',".format(i)
            else:
                s += str(i) + ','
        s = s[0:-1]
        s += border[1] + ';'
        return s
    else:
        raise TypeError("Unsupported export type " + str(type(val)))