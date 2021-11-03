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
    """test a list of kwargs type to determine if it matches the corrisponding type(s)"""
    item_types = [a for a in args if type(a).__name__ == "type"]

    if len(item_types) == 0:
        raise Warning("No types were passed to test_type as args")

    if len(kwargs) == 0:
        raise Warning("No keyword variable was passed to evaluate to test_type")

    for k, v in kwargs.items():
        if sum([isinstance(v, test) for test in item_types]) == 0:
            raise TypeError(
                "%s should be of type %s, instead type %s was given"
                % (k, " or ".join([i.__name__ for i in item_types]), type(v).__name__)
            )
