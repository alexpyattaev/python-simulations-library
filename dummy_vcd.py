
class VCDWriter:

    def change(self, var, timestamp, value):
        """Change variable's value in VCD stream.

        This is the fundamental behavior of a :class:`VCDWriter` instance. Each
        time a variable's value changes, this method should be called.

        The *timestamp* must be in-order relative to timestamps from previous
        calls to :meth:`change()`. It is okay to call :meth:`change()` multiple
        times with the same *timestamp*, but never with a past *timestamp*.

        .. Note::

            :meth:`change()` may be called multiple times before the timestamp
            progresses past 0. The last value change for each variable will go
            into the $dumpvars section.

        :param Variable var: :class:`Variable` instance (i.e. from
                             :meth:`register_var()`).
        :param int timestamp: Current simulation time.
        :param value:
            New value for *var*. For :class:`VectorVariable`, if the variable's
            *size* is a tuple, then *value* must be a tuple of the same arity.

        :raises ValueError: if the value is not valid for *var*.
        :raises VCDPhaseError: if the timestamp is out of order or the
                               :class:`VCDWriter` instance is closed.

        """
        pass

    def register_var(self, scope, name, var_type, size=None, init=None, ident=None):
        """Register a VCD variable and return function to change value.

        All VCD variables must be registered prior to any value changes.

        .. Note::

            The variable `name` differs from the variable's `ident`
            (identifier). The `name` (also known as `ref`) is meant to refer to
            the variable name in the code being traced and is visible in VCD
            viewer applications.  The `ident`, however, is only used within the
            VCD file and can be auto-generated (by specifying ``ident=None``)
            for most applications.

        :param scope: The hierarchical scope that the variable belongs within.
        :type scope: str or sequence of str
        :param str name: Name of the variable.
        :param str var_type: One of :const:`VAR_TYPES`.
        :param size:
            Size, in bits, of the variable. The *size* may be expressed as an
            int or, for vector variable types, a tuple of int. When the size is
            expressed as a tuple, the *value* passed to :meth:`change()` must
            also be a tuple of same arity as the *size* tuple. Some variable
            types ('integer', 'real', 'realtime', and 'event') have a default
            size and thus *size* may be ``None`` for those variable types.
        :type size: int or tuple(int) or None
        :param init: Optional initial value; defaults to 'x'.
        :param str ident: Optional identifier for use in the VCD stream.
        :raises VCDPhaseError: if any values have been changed
        :raises ValueError: for invalid var_type value
        :raises TypeError: for invalid parameter types
        :raises KeyError: for duplicate var name
        :returns: :class:`Variable` instance appropriate for use with
                  :meth:`change()`.

        """
        pass

    def flush(self):
        pass

    def close(self):
        pass
