import sys
import io


class TestableIO(io.BytesIO):
    def __init__(self, old_stream, trigger_string:str):
        """
        Use this to override existing stream such as stdout and trace every write to it.
        Example:
        >>> sys.stdout = TestableIO(sys.stdout, "test")
        >>> sys.stderr = TestableIO(sys.stderr, "test")
        :param old_stream: the stream to replace
        :param trigger_string: trigger to search in printed strings
        """
        io.BytesIO.__init__(self, None)
        self.old_stream = old_stream
        self.trigger_string = trigger_string

    def write(self, bytes):
        if self.trigger_string in bytes:
            import traceback
            traceback.print_stack(file=self.old_stream)
        self.old_stream.write(bytes)


