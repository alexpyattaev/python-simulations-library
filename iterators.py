__author__ = 'Alex Pyattaev'

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
    def __init__(self, iterable):
        list.__init__(self,iterable)
        self._current = 0
        self.append = None


    def __iter__(self):
        return self

    def __next__(self):
        if len(self)==0:
            raise StopIteration
        else:
            v= list.__getitem__(self, self._current)
            self._current += 1
            if self._current == len(self):
                self._current = 0
            return v

    def __delitem__(self, key):

        self._current -=key < self._current
        list.__delitem__(self, key)

    def remove(self, object):
        key = list.index(self,object)
        self.__delitem__(key)

    def insert(self, index: int, object):
        if index <= self._current:
            self._current += 1
        list.insert(self, index, object)

    def __str__(self):
        return ",".join([list.__getitem__(self, i) for i in range(len(self))])

    def __repr__(self):
        return ",".join([list.__getitem__(self, i) for i in range(len(self))])


if __name__ == "__main__":
    x = LoopedList(range(5))
    print(next(x))
    print(next(x))
    print(next(x))
    print("--",x._current)
    x.remove(3)
    print("--",x._current)
    x.remove(2)
    print("--",x._current)

    print(next(x))
    print(next(x))
    print(next(x))
    print(next(x))
    print(next(x))