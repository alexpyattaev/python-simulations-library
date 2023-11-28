import argparse
import dataclasses
import distutils.util
from enum import EnumMeta, IntEnum, Enum
from io import IOBase
from typing import Callable, Union, Dict, TypeVar, Type, Iterable
import inspect
import pytest

__all__ = (
    "Arg",
    "Int",
    "Float",
    "Str",
    "Choice",
    "File",
    "Bool",
    "List",
    "parse_to",
    "Arg_Container",
    "Force_Annotation",
)


class repr_override:
    """Provides enum-specific repr override for nice looks"""

    def __init__(self, v):
        self.v = v

    def __repr__(self):
        return str(self.v.name)

    def __str__(self):
        return str(self.v.name)

    def __eq__(self, other):
        return other == self.v

    def __hash__(self):
        return hash(self.v)


@dataclasses.dataclass
class Force_Annotation:
    """Forces all dataclass fields to be annotated. You can inherit this class to get the behaviour,
    make sure to call __post_init__.

     This will not check for fields which are functions or start with __"""

    def __post_init__(self):
        for vn, v in self.__class__.__dict__.items():
            if vn.startswith("__"):
                continue
            if inspect.isfunction(v) or inspect.isdatadescriptor(v):
                continue
            assert (
                vn in self.__annotations__
            ), f"All variables in {self.__class__} must be annotated, {vn} was not!"


class Arg:
    """Basic argument, type is not inferred, kwargs are passed to argparse"""

    def __init__(
        self, typ: Union[type, Callable[[str], object]], pos: bool = False, **kwargs
    ):
        """
        :param typ: type of data / function to convert from str to object
        :param pos: flag to indicate if positional
        :param kwargs: passed to argparse
        """
        kwargs = kwargs.copy()
        kwargs["type"] = typ
        self.kwargs = kwargs
        self.pos = pos

    def set_default(self, default) -> None:
        """
        Called when default value is known. Internal use.
        :param default:
        """
        if isinstance(default, Enum):
            self.kwargs["default"] = repr_override(default)
        else:
            self.kwargs["default"] = default


class Int(Arg):
    """Int argument"""

    def __init__(self, bounds=(None, None), **kwargs):
        self.bounds = bounds
        Arg.__init__(self, typ=int, **kwargs)


class Bool(Arg):
    """Bool argument"""

    def __init__(self, flag=False, **kwargs):
        """
        :param flag: if true the value will act as a flag (as in store_true)
        :param kwargs: passed to argparse
        """
        self.flag = flag
        Arg.__init__(self, typ=bool, **kwargs)
        if "default" in kwargs:
            self.set_default(kwargs["default"])

    def set_default(self, default):
        def strtobool(val):
            """Convert a string representation of truth to true (1) or false (0).

            True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
            are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
            'val' is anything else.
            """
            val = val.lower()
            if val in ("y", "yes", "t", "true", "on", "1"):
                return True
            elif val in ("n", "no", "f", "false", "off", "0"):
                return False
            else:
                raise ValueError("invalid truth value %r" % (val,))

        if self.flag:
            try:
                # This is ok to ignore since same instance may be used to parse many times,
                # in which case this is already done
                self.kwargs.pop("type")
            except KeyError:
                pass
            if default is True:
                self.kwargs["action"] = "store_false"
            elif default is False:
                self.kwargs["action"] = "store_true"
            else:
                raise ValueError("Bool flags must have default set!")
        else:
            self.kwargs["type"] = strtobool
            Arg.set_default(self, default)


class Float(Arg):
    """Float argument"""

    def __init__(self, bounds=(None, None), **kwargs):
        self.bounds = bounds
        Arg.__init__(self, typ=float, **kwargs)


class Str(Arg):
    """String argument"""

    def __init__(self, **kwargs):
        Arg.__init__(self, typ=str, **kwargs)


class _MetaList(type):
    def __getitem__(self, item):
        return self(typ=item, action="extend", nargs="+")


class List(Arg, metaclass=_MetaList):
    """List of certain homogenous type items

    Format is
     " --arg 1 2 3 4 "
    where 1 2 3 4 are the elements to be in the list.

    Alternatively, one can specify
     " --arg 1 --arg 2 --arg 3 --arg 4 "
    """

    def __init__(
        self,
        **kwargs,
    ):
        Arg.__init__(self, **kwargs)

    def __call__(self, **kwargs):
        self.kwargs.update(kwargs)
        return self


class _MetaChoice(type):
    """
    Implementation details (metaclass) for Choice class.
    """

    def __getitem__(self, item):
        """Get a variant of Choice for a given type"""
        if isinstance(item, EnumMeta):
            choices = [repr_override(f) for f in item]
            extra_help = "; ".join(f"{f.name}: {f.value}" for f in item)
            if issubclass(item, IntEnum):

                def typ(x):
                    try:
                        return getattr(item, x)
                    except AttributeError:
                        try:
                            return item(int(x))
                        except ValueError:
                            pass
                    raise argparse.ArgumentTypeError(
                        f"invalid {item.__name__} value: {x}"
                    )

            else:

                def typ(x):
                    try:
                        return getattr(item, x)
                    except AttributeError:
                        raise argparse.ArgumentTypeError(
                            f"invalid {item.__name__} value: {x}"
                        )

        else:
            choices = list(item)
            typ = type(item[0])
            extra_help = None
        return self(choices=choices, typ=typ, extra_help=extra_help)


class Choice(Arg, metaclass=_MetaChoice):
    """Choice out of iterable or Enum subclass.

    If enum is given as argument, the names of the fields will be used, 
    and values will be returned.
    """

    def __init__(self, choices, extra_help=None, **kwargs):
        self.extra_help = extra_help
        self.choices = choices
        Arg.__init__(self, choices=choices, **kwargs)

    def __call__(self, **kwargs):
        assert "choices" not in kwargs
        if self.extra_help is not None:
            if "help" in kwargs:
                kwargs["help"] += " [" + self.extra_help + "]"
            else:
                kwargs["help"] = "[" + self.extra_help + "]"

        self.kwargs.update(kwargs)

        return self


class File(Arg):
    """File argument"""

    def __init__(self, mode="r", bufsize=-1, encoding=None, errors=None, **kwargs):
        Arg.__init__(
            self, typ=argparse.FileType(mode, bufsize, encoding, errors), **kwargs
        )


autocast_types = {int: Int, float: Float, str: Str, bool: Bool}


@dataclasses.dataclass
class Arg_Container(Force_Annotation):
    """Argument Container class"""

    def asdict(self) -> Dict[str, object]:
        result = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, IOBase) and hasattr(value, "name"):
                value = value.name
            result[f.name] = value
        return result

    def to_json(self):
        result = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, IOBase):
                if hasattr(value, "name"):
                    value = value.name
                else:
                    raise TypeError(f"Could not convert filed {f.name}={value}")
            if isinstance(value, Enum):
                value = value.name
            result[f.name] = value
        return result


A = TypeVar("A", Arg_Container, Arg_Container)


def parse_to(
    container_class: Type[A],
    epilog: str = "",
    transform_names: Callable[[str], str] = None,
    verbose: bool = False,
    args=None,
) -> A:
    """
    Parse command line using argparse into the provided container class.

    :param container_class: a frozen dataclass which will hold the parsed values
    :param epilog: epilog message for argparse
    :param transform_names: callable to mess with variable names (e.g. to do char translation/capitalization etc)
    :param verbose: set if you want the produced argparse code to be dumped
    :param args: passed verbatim to ArgumentParser.parse_args
    :return: container_class filled in with parsed args.
    """
    assert issubclass(
        container_class,
        Arg_Container,
    ), "container_class should be a subclass of Arg_Container"

    def mangle_name(n: str, positional: bool):
        s = "" if positional else "--"
        if transform_names is not None:
            n = transform_names(n)
        return s + n

    parser = argparse.ArgumentParser(description=container_class.__doc__, epilog=epilog)

    for field in dataclasses.fields(container_class):
        name = field.name
        default = field.default
        default_factory = field.default_factory
        value_or_class = field.type
        if isinstance(
            value_or_class, type
        ):  # Type is not an instance (e.g. int or float)
            if issubclass(value_or_class, Arg):
                # noinspection PyArgumentList
                value = value_or_class(default=default)
            elif value_or_class in autocast_types:  # this handles primitive types
                # noinspection PyTypeChecker
                value = autocast_types[value_or_class](default=default)  # type: ignore
            else:
                raise TypeError(
                    f"Values must be typed as subclasses of Arg or be one of {autocast_types}"
                )
        else:
            value = value_or_class
            if default is not None and default_factory == dataclasses.MISSING:
                value.set_default(default)
        if verbose:
            print("add_argument", mangle_name(name, value.pos), value.kwargs)
        parser.add_argument(mangle_name(name, value.pos), **value.kwargs)

    arg_dict = parser.parse_args(args=args)
    corrected_dict = {}
    for k, v in vars(arg_dict).items():
        if isinstance(v, repr_override):
            v = v.v
        corrected_dict[k] = v
    return container_class(**corrected_dict)


@pytest.fixture
def arg_definitions():
    class str_enum(Enum):
        """Enum of strings of things"""

        A = "All the things"
        B = "Best of things"

    class int_enum(IntEnum):
        ONE = 1
        TWO = 2

    @dataclasses.dataclass
    class Args(Arg_Container):
        """Example of description for your application"""

        req_str: Str(help="required str field") = "boo"

        opt_str: Str(help="str field") = "bla"
        bare_str: str = "aaa"

        int_field: Int(help="Int field") = 120
        bare_int: int = 150

        float_field: Float(help="Float field") = 10.0
        bare_float: float = 15.0

        str_enum_field: Choice[str_enum](help="choice from string enum") = str_enum.A
        int_enum_field: Choice[int_enum](help="choice from int enum") = int_enum.TWO

        list_choice: Choice[[1, 4, 7]](help="choice from iterable") = 0
        list_of_int: List[int](help="List of integers") = dataclasses.field(
            default_factory=list
        )
        bool_field: bool = False
        bool_flag: Bool(flag=True) = False
        bool_switch: Bool(flag=False) = False

    return Args


def test_parse(arg_definitions):
    opt = "--list_of_int 1 2 3 --req_str=bla --opt_str=foo --bare_str=ads --int_field=10 --bare_int=20 --float_field=1.2 \
    --bare_float=35.0  --str_enum_field=A --int_enum_field=2 --list_choice=7 ".split()
    args = parse_to(arg_definitions, args=opt, verbose=True)
    print(args)


def test_fail(arg_definitions):
    opt = "--req_str=asd --opt_str=foo --bare_str=ads --int_field=10 --bare_int=20 --float_field=1.2 \
    --bare_float=35.0  --str_enum_field=C --int_enum_field=2 --list_choice=7".split()
    with pytest.raises(SystemExit):
        parse_to(arg_definitions, args=opt)


def test_help(arg_definitions):
    with pytest.raises(SystemExit):
        parse_to(arg_definitions, args=["--help"])


def test_bool(arg_definitions):
    opt = "--bool_field=False  --bool_switch=False".split()
    args = parse_to(arg_definitions, args=opt, verbose=True)
    # print(args.bool_field, args.bool_flag, args.bool_switch)
    assert not args.bool_field
    assert not args.bool_flag
    assert not args.bool_switch

    opt = "--bool_field=True --bool_flag --bool_switch=True".split()
    args = parse_to(arg_definitions, args=opt, verbose=True)
    assert args.bool_field
    assert args.bool_flag
    assert args.bool_switch
