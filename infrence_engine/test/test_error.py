from typing import Type
import pytest
import infrence_engine.error as er

# #############################################################################
# #############################################################################
# test_type
# #############################################################################
# #############################################################################


def test_test_type_warning_args():
    with pytest.raises(Warning) as err:
        er.test_type(b="Bob")
    assert "No types were passed to test_type as args" in str(err.value)


def test_test_type_warning_kwargs():
    with pytest.raises(Warning) as err:
        er.test_type(int)
    assert "No keyword variable was passed to evaluate to test_type" in str(err.value)


def test_test_type_warning_args_kwargs():
    with pytest.raises(Warning) as err:
        er.test_type()
    assert "No types were passed to test_type as args" in str(err.value)


def test_test_type_single_wrong():
    with pytest.raises(TypeError) as err:
        er.test_type(int, b="Bob")
    assert "b should be of type int, instead type str was given" in str(err.value)


def test_test_type_single_wrong_dif_Type():
    with pytest.raises(TypeError) as err:
        er.test_type(int, b=1.1231)
    assert "b should be of type int, instead type float was given" in str(err.value)


def test_test_type_double_wrong():
    with pytest.raises(TypeError) as err:
        er.test_type(int, float, b="Bob")
    assert "b should be of type int or float, instead type str was given" in str(
        err.value
    )


def test_test_type_single_correct():
    try:
        er.test_type(int, b=1)
    except Warning:
        assert False, "Warning was raised on a correct statement"
    except TypeError:
        assert False, "Type error was raised on a correct statement"


def test_test_type_single_wrong_one_wrong_kwargs_one_right_kwargs():
    with pytest.raises(TypeError) as err:
        er.test_type(int, b=1, c="bob")
    assert "c should be of type int, instead type str was given" in str(err.value)


def test_test_type_single_wrong_two_wrong_kwargs():
    with pytest.raises(TypeError) as err:
        er.test_type(int, b=1.034, c="bob")
    assert "b should be of type int, instead type float was given" in str(err.value)


def test_test_type_single_right_two_right_kwargs():
    try:
        er.test_type(int, b=1, c=2)
    except Warning:
        assert False, "Warning was raised on a correct statement"
    except TypeError:
        assert False, "Type error was raised on a correct statement"


def test_test_type_double_wrong_one_wrong_kwargs_one_right_kwargs():
    with pytest.raises(TypeError) as err:
        er.test_type(int, float, b="gg", c=2)
    assert "b should be of type int or float, instead type str was given" in str(
        err.value
    )


def test_test_type_double_wrong_one_right_kwargs_one_wrong_kwargs():
    with pytest.raises(TypeError) as err:
        er.test_type(bool, float, b=False, c=b"a")
    assert "c should be of type bool or float, instead type bytes was given" in str(
        err.value
    )


def test_test_type_double_right_two_right_kwargs():
    try:
        er.test_type(int, bool, b=1, c=True)
    except Warning:
        assert False, "Warning was raised on a correct statement"
    except TypeError:
        assert False, "Type error was raised on a correct statement"
