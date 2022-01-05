__author__ = 'Yaniv Aknin (objwalk), S. Lott (nested break)'

from contextlib import contextmanager
from typing import Mapping, Set, Sequence, Iterator, Tuple

string_types = (str, bytes)


def _iter_items(mapping):
    return getattr(mapping, 'iteritems', mapping.items)()


def objwalk(obj, path=(), memo=None) -> Iterator[Tuple[Tuple[str], object]]:
    """Walks iterable objects recursively. Supports Sequence, Set and Mapping instances.
    :param obj: the object to walk
    :param path: the path to current object (used internally)
    :param memo: memo dict to avoid infinite loops (used internally)
    :returns iterator over paths to objects and objects themselves
    """
    if memo is None:
        memo = set()
    iterator = None
    if isinstance(obj, Mapping):
        iterator = _iter_items
    elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, string_types):
        iterator = enumerate
    if iterator:
        if id(obj) not in memo:
            memo.add(id(obj))
            for path_component, value in iterator(obj):
                for result in objwalk(value, path + (path_component,), memo):
                    yield result
            memo.remove(id(obj))
    else:
        yield path, obj


@contextmanager
def nested_break():
    """Allows for neatly arranged breaks out of deeply nested loops"""
    class NestedBreakException(Exception):
        pass

    try:
        yield NestedBreakException
    except NestedBreakException:
        pass


def test_nested_break():
    input_string = "nnttNNttttny"
    iter_input = iter(input_string)
    num_N = 0
    num_other = 0
    iterations = 100
    with nested_break() as my_label:
        while iterations:  # prevent truly infinite loops, in reality this could be "while True:"
            while iterations:  # prevent truly infinite loops, in reality this could be "while True:"
                ok = next(iter_input)
                if ok == "y" or ok == "Y":
                    raise my_label  # breaks out of outer loop
                if ok == "n" or ok == "N":
                    num_N += 1
                    break
                num_other += 1
            iterations -= 1
    assert num_N == input_string.lower().count("n")
    assert num_other == input_string.lower().count("t")
    assert iterations > 0
