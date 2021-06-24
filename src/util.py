from pandas import DataFrame, Series
from datatable import Frame


def v_print(test: bool = False, *args):
    if test is True:
        print(*args)


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
            # here will test if it's a file,or a folder
            pass
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
    elif isinstance(Obj, bool):
        return "Bool"
    elif isinstance(Obj, DataFrame):
        return "DataFrame"
    elif isinstance(Obj, Series):
        if simple:
            return "DataFrame"
        else:
            return "Series"
    elif isinstance(Obj, Frame):
        return "Frame"
    else:
        return "Other"
