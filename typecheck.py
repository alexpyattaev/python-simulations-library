"""Type-checking tools. Allows to quickly force type-checking for a function or a method

.. DANGER::
   contains unprotected calls to eval!!! Do not ever use in code with untrusted input!!!

"""
__author__ = 'Alex Pyattaev'
import inspect



class arg_check(object):
    """
    A special wrapper class to submit lambdas and other functions for parameter checking.

    .. DANGER::
    contains unprotected calls to eval!!!

    """
    def __init__(self, t, l):
        """
        constructs the arg-check object

        .. DANGER::
        contains unprotected calls to eval!!!

        :param t: The type of the input
        :param l: Lambda-expression to test the variable. Should return something with truth value.
        """
        self._l = l
        self._t = t

    def check(self, val, name):
        """
        Run the type-check

        :param val: value to assign
        :param name: the name of the variable for  eval

        .. DANGER::
        contains unprotected calls to eval!!!
        """
        try:
            ll = eval("lambda {}: {}".format(name, self._l))(val)
            #print(ll)
        except:
            ll = False
        return isinstance(val, self._t) and ll

    def __str__(self):
        return "Type: {}, {}".format(self._t, self._l)


def validate(f):
    """
    Decorator to enforce strict type-checking. Uses the python TypeHints. If the hint is an instance of arg_check,
    it will assume it is a callable that should be used to test input. True means input is ok.
    :param f: the function to wrap
    :return: Wrapped function
    """
    def safe_str(o):
        try:
            return str(o)
        except Exception as e:
            return type(o)

    def wrapper(*args, **kwargs):
        fname = f.__name__
        fsig = inspect.signature(f)
        params = dict(zip(list(fsig.parameters)[0:len(args)], args))
        params.update(kwargs)
        #Prepare debug messages
        vars = ', '.join('{}={}'.format(*map(safe_str, pair)) for pair in params.items())
        msg = 'call to {}({}) failed: '.format(fname, vars)
        #print('wrapped call to {}({})'.format(fname, params))
        for k, v in fsig.parameters.items():
            if k in params:
                p = params[k]
            else:
                continue
            if v.annotation == inspect._empty:
                #If no type hint is given do nothing
                pass
            elif isinstance(v.annotation, arg_check):
                #see if our annotation is an arg_check instance holding a lambda or a function of some sort
                test = v.annotation.check(params[k], k)
                assert test, msg + "parameter {} did not match condition {}".format(
                    k, str(v.annotation))
            else:
                #Otherwise just do plain type check
                assert isinstance(p, v.annotation), msg + "parameter {} did not match type {}".format(
                    k, v.annotation)
        ret = f(*args, **kwargs)
        #Now test the output
        #see if our annotation is an arg_check instance holding a lambda or a function of some sort
        ra = fsig.return_annotation
        if ra == inspect._empty:
            #If no type hint is given do nothing
            pass
        elif isinstance(ra, arg_check):
            test = ra.check(ret, 'ret')
            assert test, "Output condition violation {}".format(ra)
            #Otherwise just do plain type check
        else:
            assert isinstance(ret, ra), "Output type mismatch, expected {} got {}:{}".format(
                ra, ret, type(ret))
        return ret
    return wrapper


if __name__ == "__main__":
    class T(object):
        @validate
        def __init__(self, a: int, b: str):
            self.a = a
            self.b = b * a

        @validate
        def test(self, x: str, y: int=5)-> arg_check(str, "len(ret)>0"):
            return x * self.a

        def __str__(self):
            return self.b
    x = T(5, "foo")
    print(x)
    print(x.test(x="lol"))
    print(x.test("5", y=0))
    @validate
    def foo(x: arg_check(float, "10 < x < 100"), y: float) -> float:
        return x * y

    print(foo(20.0, 0.0))
    print(foo(1.0, 6.8))
