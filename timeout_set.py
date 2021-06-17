from typing import MutableMapping, MutableSet, List, Hashable, Callable

from lib.stuff import ANY


class Timeout_Set(MutableSet):
    """A set-like container that automatically removes entries when they are outdated.

    Do not use the container with massive number of entries, it is not optimized"""

    def __init__(self, timeout: int, timefunc):
        """
        Create a timeout set
        :param timeout: The timeout in ticks until entry is invalidated
        :param timefunc:
        """
        self.timeout = timeout
        self.timefunc = timefunc
        self._members = dict()
        self.on_member_added = []

    def add(self, m):
        if m not in self._members:
            for f in self.on_member_added:
                f(m)
        self._members[m] = self.timefunc()

    def remove(self, m)-> None:
        self._members.pop(m)

    def discard(self, m) -> None:
        try:
            self._members.pop(m)
        except KeyError:
            pass

    def __iter__(self):
        now = self.timefunc()

        items = list(self._members.items())
        for x, t in items:
            if t + self.timeout >= now:
                yield x
            else:
                self._members.pop(x)

    def __contains__(self, key):
        now = self.timefunc()

        ret = False
        for item, t in list(self._members.items()):
            if t + self.timeout < now:
                self._members.pop(item)
            elif item == key:
                ret = True
        return ret

    def __repr__(self):
        return str(self._members)

    def __len__(self):
        now = self.timefunc()

        items = list(self._members.items())
        for x, t in items:
            if t + self.timeout < now:
                self._members.pop(x)
        return len(self._members)


class Timeout_Dict(MutableMapping):
    def __init__(self, timeout: int, timefunc):
        self.timeout = timeout
        self.timefunc = timefunc
        self._members = dict()
        self.on_member_added: List[Callable[[Hashable, object], None]] = []  # Functions to be called when item is added
        self.on_member_removed: List[
            Callable[[Hashable, object], None]] = []  # Functions to be called when item is removed (or times out)

    def __setitem__(self, key, value):
        assert key is not ANY, "ANY is not allowed as key!!!"
        if key not in self._members:
            for f in self.on_member_added:
                f(key, value)
        self._members[key] = (self.timefunc(), value)

    def touch(self, key):
        t, item = self._members[key]
        self._members[key] = (self.timefunc(), item)

    def pop(self, key):
        assert key is not ANY, "ANY is not allowed as key!!!"
        try:
            __, v = self._members.pop(key)
            for f in self.on_member_removed:
                f(key, v)
        except KeyError:
            pass

    def item_age(self, key) -> int:
        """Returns the age of the item

        i.e. the time elapsed since it was added to the container
        :param key: lookup key to use
        :returns time in ticks"""
        assert key is not ANY, "ANY is not allowed as key!!!"
        t, item = self._members[key]
        return self.timefunc() - t

    def __contains__(self, key):
        assert key is not ANY, "ANY is not allowed as key!!!"
        ret = False
        for i in list(self._members.keys()):
            t, item = self._members[i]
            if t + self.timeout < self.timefunc():
                __, v = self._members.pop(i)
                for f in self.on_member_removed:
                    f(i, v)
            elif i == key:
                ret = True
        return ret

    def __getitem__(self, key):
        assert key is not ANY, "ANY is not allowed as key!!!"
        t, item = self._members[key]
        if t + self.timeout >= self.timefunc():
            return item
        else:
            self._members.pop(key)
            raise KeyError

    def __iter__(self):
        now = self.timefunc()
        items = list(self._members.items())
        for key, (t, v) in items:
            if t + self.timeout >= now:
                yield key
            else:
                __, v = self._members.pop(key)
                for f in self.on_member_removed:
                    f(key, v)

    def items(self):
        now = self.timefunc()
        items = list(self._members.items())
        for key, (t, v) in items:
            if t + self.timeout >= now:
                yield key, v
            else:
                __, v = self._members.pop(key)
                for f in self.on_member_removed:
                    f(key, v)

    def keys(self):
        now = self.timefunc()
        items = list(self._members.items())
        for key, (t, v) in items:
            if t + self.timeout >= now:
                yield key
            else:
                __, v = self._members.pop(key)
                for f in self.on_member_removed:
                    f(key, v)

    def __delitem__(self, key):
        assert key is not ANY, "ANY is not allowed as key!!!"
        __, v = self._members.pop(key)
        for f in self.on_member_removed:
            f(key, v)

    def __len__(self):
        now = self.timefunc()
        items = list(self._members.items())
        for key, (t, v) in items:
            if t + self.timeout < now:
                __, v = self._members.pop(key)
                for f in self.on_member_removed:
                    f(key, v)

        return len(self._members)

    def __repr__(self):
        return str(self._members)
