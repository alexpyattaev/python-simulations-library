import json
from typing import *

_JSONType_0 = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
_JSONType_1 = Union[str, int, float, bool, None, Dict[str, _JSONType_0], List[_JSONType_0]]
_JSONType_2 = Union[str, int, float, bool, None, Dict[str, _JSONType_1], List[_JSONType_1]]
_JSONType_3 = Union[str, int, float, bool, None, Dict[str, _JSONType_2], List[_JSONType_2]]
JSONObject = Dict[str, _JSONType_3]
JSONType = Union[str, int, float, bool, None, JSONObject, List[JSONObject]]


def json_dump_nicely(x: JSONType, file, *args, **kwargs):
    for chunk in json.JSONEncoder(*args, indent=4, sort_keys=True, **kwargs).iterencode(x):
        file.write(chunk)

