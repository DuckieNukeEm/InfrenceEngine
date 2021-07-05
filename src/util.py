from pandas import DataFrame, Series
from numpy import ndarray
from datatable import Frame
from typing import Union
from os.path import isfile, isdir


def v_print(test: bool = False, *args):
    if test is True:
        print(args)


def typeof(Obj, simple: bool = True) -> str:
    """retuns what the item is in as astring

    Argumnet:
        Obj:    any python objet to check
        simple {bool}:  simplefies down a few things
                        (default: False)
    Returns:
        str
    """
    if isinstance(Obj, str):
        if simple:
            return "Str"
        else:
            if isfile(Obj):
                return "File"
            elif isdir(Obj):
                return "Dir"
            else:
                return "Str"
    elif isinstance(Obj, bool):
        return "Bool"
    elif isinstance(Obj, int):
        return "Int"
    elif isinstance(Obj, float):
        return "Float"
    elif isinstance(Obj, complex):
        return "Complex"
    elif isinstance(Obj, dict):
        return "Dict"
    elif isinstance(Obj, list):
        return "List"
    elif isinstance(Obj, DataFrame):
        return "DataFrame"
    elif isinstance(Obj, Series):
        if simple:
            return "DataFrame"
        else:
            return "Series"
    elif isinstance(Obj, Frame):
        return "Frame"
    elif isinstance(Obj, ndarray):
        return "Array"
    else:
        return "Other"


def dtype_simple(Obj: Union[Series, ndarray, Frame]) -> str:
    """Simplifies the dtypes down to int, float, str, date, datetime"""

    if typeof(Obj, False) in ["Series", "Array"]:
        obj_type = str(Obj.dtype)
    elif typeof(Obj) == "Frame":
        obj_type = str(Obj.stype).split(".")[1]
    else:
        return None

    if obj_type[0:5] == "float":
        return "float"
    elif obj_type[0:4] == "bool":
        return "bool"
    elif obj_type[0:3] == "int":
        return "int"
    elif obj_type[0:6] in ["object", "string", "unicod", "mixed", "str32", "str64"]:
        return "str"
    elif obj_type[0:8] in ("datetime", "time64"):  # time64 is datetime in datatable
        return "datetime"
    elif obj_type[0:6] == "date32":
        return "datetime"
    elif obj_type[0:9] == "timedelta":
        return "timedelta"
    elif obj_type[0:7] == "complex":
        return "complex"
    else:
        return obj_type
