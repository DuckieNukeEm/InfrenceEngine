from numpy.lib.arraysetops import isin


def raise_type_error(Obj, name: str, *args):
    """raise an error with a common theme"""
    if len(args) == 0:
        Msg = "Incorrect type was passed for %s " % str(name)
    else:
        sb = "[" + ", ".join([str(x) for x in args]) + "]"
        Msg = "%s should be type %s, type %s was passed instead" % (
            str(name),
            sb,
            type(Obj),
        )
    raise TypeError(Msg)


def test_type(*args, **kwargs):
    """test a list of kwargs type to determine if it matches the corrisponding type"""
    item_types = [a for a in args if type(a) == "type"]

    for k, v in kwargs:
        pass_flag = 0
        for t in item_types:
            if isinstance(v, t) is True:
                pass_flag = 1
        if pass_flag == 0:
            raise TypeError("%s should be of type" % k)
