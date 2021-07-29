import argparse
import dataclasses
from enum import EnumMeta
from io import IOBase
from typing import Callable, Union, Dict, TypeVar, Type
import inspect

__all__ = ('Arg', 'Int', 'Float', 'Str', 'Choice', 'File', 'Bool', 'List', 'parse_to',
           'Arg_Container', 'Force_Annotation')


@dataclasses.dataclass
class Force_Annotation:
    """Forces all dataclass fields to be annotated. You can inherit this class to get the behaviour,
    make sure to call __post_init__.

     This will not check for fields which are functions or start with __"""

    def __post_init__(self):
        for vn, v in self.__class__.__dict__.items():
            if vn.startswith('__'):
                continue
            if inspect.isfunction(v):
                continue
            assert vn in self.__annotations__, f"All variables in {self.__class__} must be annotated, {vn} was not!"


class Arg:
    """Basic argument, type is not inferred, kwargs are passed to argparse """

    def __init__(self, typ: Union[type, Callable[[str], object]], pos: bool = False, **kwargs):
        """
        :param typ: type of data / function to convert from str to object
        :param pos: flag to indicate if positional
        :param kwargs: passed to argparse
        """
        kwargs['type'] = typ
        self.kwargs = kwargs
        self.pos = pos

    def set_default(self, default) -> None:
        """
        Called when default value is known. Internal use.
        :param default:
        """
        self.kwargs['default'] = default


class Int(Arg):
    """Int argument"""

    def __init__(self, **kwargs):
        Arg.__init__(self, typ=int, **kwargs)


class Bool(Arg):
    """Bool argument"""

    def __init__(self, flag=True, **kwargs):
        """
        :param flag: if true the value will act as a flag (as in store_true)
        :param kwargs: passed to argparse
        """
        self.flag = flag
        Arg.__init__(self, typ=bool, **kwargs)

    def set_default(self, default):
        if self.flag:
            self.kwargs.pop('type')
            if default is True:
                self.kwargs['action'] = 'store_false'
            elif default is False:
                self.kwargs['action'] = 'store_true'
            else:
                raise ValueError('Bool flags must have default set!')
        else:
            Arg.set_default(self, default)


class Float(Arg):
    """Float argument"""

    def __init__(self, **kwargs):
        Arg.__init__(self, typ=float, **kwargs)


class Str(Arg):
    """String argument"""

    def __init__(self, **kwargs):
        Arg.__init__(self, typ=str, **kwargs)


class _MetaList(type):
    def __getitem__(self, item):
        return self(typ=item, action="extend", nargs="+", default=[])


class List(Arg, metaclass=_MetaList):
    """List of certain homogenous type items"""

    def __init__(self, **kwargs, ):
        Arg.__init__(self, **kwargs)

    def __call__(self, **kwargs):
        self.kwargs.update(kwargs)
        return self


class _MetaChoice(type):
    """Metaclass for Choice"""

    def __getitem__(self, item):
        choices = list(item)
        if isinstance(item, EnumMeta):
            extra_help = "; ".join(f"{f.name}: {f.value}" for f in item)
            # TODO: support IntEnum better maybe?
            typ = item
        else:
            typ = type(item[0])
            extra_help = None
        return self(choices=choices, typ=typ, extra_help=extra_help)


class Choice(Arg, metaclass=_MetaChoice):
    """Choice out of iterable or Enum"""

    def __init__(self, choices, extra_help=None, **kwargs):
        self.extra_help = extra_help
        Arg.__init__(self, choices=choices, **kwargs)

    def __call__(self, **kwargs):
        if self.extra_help is not None:
            if 'help' in kwargs:
                kwargs['help'] += '[' + self.extra_help + ']'
            else:
                kwargs['help'] += '[' + self.extra_help + ']'
        assert 'choices' not in kwargs

        self.kwargs.update(kwargs)

        return self


class File(Arg):
    """File argument"""

    def __init__(self, mode='r', bufsize=-1, encoding=None, errors=None, **kwargs):
        Arg.__init__(self, typ=argparse.FileType(mode, bufsize, encoding, errors), **kwargs)


autocast_types = {int: Int,
                  float: Float,
                  str: Str,
                  bool: Bool}

T = TypeVar('T')


def parse_to(container_class: Type[T], epilog: str = "", transform_names: Callable[[str], str] = None,
             verbose: bool = False, args=None) -> T:
    """
    Parse command line using argparse into the provided container class.

    :param container_class: a frozen dataclass which will hold the parsed values
    :param epilog: epilog message for argparse
    :param transform_names: callable to mess with variable names (e.g. to do char translation/capitalization etc)
    :param verbose: set if you want the produced argparse code to be dumped
    :param args: passed verbatim to ArgumentParser.parse_args
    :return: container_class filled in with parsed args.
    """

    def mangle_name(n: str, positional: bool):
        s = "" if positional else "--"
        if transform_names is not None:
            n = transform_names(n)
        return s + n

    assert dataclasses.is_dataclass(container_class), "container_class should be a dataclass"
    parser = argparse.ArgumentParser(description=container_class.__doc__, epilog=epilog)

    for field in dataclasses.fields(container_class):
        name = field.name
        default = field.default
        value_or_class = field.type
        if isinstance(value_or_class, type):  # Type is not an instance (e.g. int or float)
            if issubclass(value_or_class, Arg):
                # noinspection PyArgumentList
                value = value_or_class(default=default)
            elif value_or_class in autocast_types:  # this handles primitive types
                # noinspection PyTypeChecker
                value = autocast_types[value_or_class](default=default)  # type: ignore
            else:
                raise TypeError(f"Values must be typed as subclasses of Arg or be one of {autocast_types}")
        else:
            value = value_or_class
            if default is not None:
                value.set_default(default)
        if verbose:
            print("add_argument", mangle_name(name, value.pos), value.kwargs)
        parser.add_argument(mangle_name(name, value.pos), **value.kwargs)

    arg_dict = parser.parse_args(args=args)
    return container_class(**vars(arg_dict))


@dataclasses.dataclass
class Arg_Container(Force_Annotation):
    def asdict(self) -> Dict[str, object]:
        result = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, IOBase) and hasattr(value, 'name'):
                value = value.name
            result[f.name] = value
        return result
