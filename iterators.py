__author__ = 'Alex Pyattaev'

import unittest
from typing import List, Dict


def loop_over(iterable):
    """Works similarly to Python's standard itertools.cycle function,
    but does not make internal copies.

    Note: loop_over produces infinite loops!!!

    Note: the loop_over iterator has to be reset once it is polled over empty container
    """
    while iterable:
        for element in iterable:
            yield element


class LoopedList(list):
    """
    Just like normal list but will produce infinite loops over elements when used in for loop.
    Handles reasonably when elements are removed while iterating.
    """

    def __init__(self, iterable):
        list.__init__(self, iterable)
        self._current = 0
        self.append = None

    def __iter__(self):
        return self

    def __next__(self):
        if len(self) == 0:
            raise StopIteration
        else:
            v = list.__getitem__(self, self._current)
            self._current += 1
            if self._current >= len(self):
                self._current = 0
            return v

    def __delitem__(self, key):
        self._current -= key < self._current
        list.__delitem__(self, key)

    def remove(self, obj):
        key = list.index(self, obj)
        self.__delitem__(key)

    def insert(self, index: int, obj):
        if index <= self._current:
            self._current += 1
        list.insert(self, index, obj)

    def __str__(self):
        return ",".join([list.__getitem__(self, i) for i in range(len(self))])

    def __repr__(self):
        return ",".join([list.__getitem__(self, i) for i in range(len(self))])


def list_of_dict_to_dict_of_lists(x: List[dict], key_set: set = None) -> Dict[list, object]:
    """
    Reorganizes data from multiple dicts into dictionary of lists.
    [{"a": 1}, {"a": 2}, {"a": 3}] => {'a': [1, 2, 3]}
    
    The keys in all dicts must match, or keyset must be provided in order to use a subset of keys
    :param key_set: a set of keys to expect in every element
    :param x: the list of dicts to convert
    :return: converted dict of lists
    """
    if key_set is None:
        key_set = set(x[0].keys())
    res = {k: [] for k in key_set}
    for it in x:
        if not key_set.issubset(set(it.keys())):
            raise KeyError(f"Key set {key_set} not satisfied by keys present in {it}")

        for k in key_set:
            res[k].append(it[k])
    return res


class Test_ml_utils(unittest.TestCase):
    def test_looped_list(self):
        x = LoopedList(range(5))
        self.assertEqual(next(x), 0)
        self.assertEqual(next(x), 1)
        self.assertEqual(next(x), 2)
        x.remove(3)
        x.remove(2)
        self.assertEqual(next(x), 4)
        self.assertEqual(next(x), 0)
        self.assertEqual(next(x), 1)
        self.assertEqual(next(x), 4)
        self.assertEqual(next(x), 0)

    def test_list_of_dict_to_dict_of_lists(self):
        r1 = list_of_dict_to_dict_of_lists([{"a": 1}, {"a": 2}, {"a": 3}])
        self.assertEqual(r1, {'a': [1, 2, 3]})
        r2 = list_of_dict_to_dict_of_lists([{"a": 1, "b": 1}, {"a": 2}, {"a": 3}], key_set={"a"})
        self.assertEqual(r2, {'a': [1, 2, 3]})
        with self.assertRaises(KeyError):
            list_of_dict_to_dict_of_lists([{"a": 1, "b": 1}, {"a": 2}, {"a": 3}])
        with self.assertRaises(KeyError):
            list_of_dict_to_dict_of_lists([{"b": 1}, {"c": 2}, {"a": 3}])


if __name__ == "__main__":
    unittest.main()
