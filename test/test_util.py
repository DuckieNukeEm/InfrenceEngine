from pandas import DataFrame, Series
from numpy import array
from datatable import Frame, f, time
from test_dataframe import make_test_dataframe
import pytest
import sys

sys.path.insert(0, "../src")
import util as U

# Creating common params
use_columns = ["Name", "DoB", "Hieght", "City"]

# #############################################################################
# #############################################################################
# typeof
# #############################################################################
# #############################################################################


def test_str_simple():
    assert U.typeof("Bob", True) == "Str"


def test_str():
    assert U.typeof("Bob", False) == "Str"


def test_dir_simeple():
    assert U.typeof("../src") == "Str"


def test_dir():
    assert U.typeof("../src", False) == "Dir"


def test_file_simeple():
    assert U.typeof("../src/util.py") == "Str"


def test_file():
    assert U.typeof("../src/util.py", False) == "File"


def test_int():
    assert U.typeof(1) == "Int"


def test_float():
    assert U.typeof(1.0) == "Float"


def test_complex():
    assert U.typeof(complex(1, 2)) == "Complex"


def test_dict():
    assert U.typeof({"Bob": 11}) == "Dict"


def test_list():
    assert U.typeof([1, 2.0]) == "List"


def test_bool():
    assert U.typeof(True) == "Bool"


def test_dataframe():
    assert U.typeof(DataFrame()) == "DataFrame"


def test_series_simple():
    Df = DataFrame([[1, 2], [3, 4]])
    assert U.typeof(Df[0], True) == "DataFrame"


def test_series():
    Df = DataFrame([[1, 2], [3, 4]])
    assert U.typeof(Df[0], False) == "Series"


def test_frame():
    Df = DataFrame([[1, 2], [3, 4]])
    Df = Frame(Df)
    assert U.typeof(Df[0], False) == "Frame"


def test_ndarray_all():
    Df = DataFrame([[1, 2], [3, 4]])
    assert U.typeof(Df.values) == "Array"


def test_ndarray_one():
    Df = DataFrame([[1, 2], [3, 4]])
    assert U.typeof(Df[0].values) == "Array"


# #############################################################################
# #############################################################################
# typeof
# #############################################################################
# #############################################################################


def test_dtypes_simple_pd_float():
    test_df = make_test_dataframe()
    assert U.dtype_simple(test_df["PayRate"]) == "float"
    assert U.dtype_simple(test_df["PayRate"].astype("float32")) == "float"
    assert U.dtype_simple(test_df["PayRate"].astype("float16")) == "float"


def test_dtypes_simple_np_float():
    test_df = make_test_dataframe()
    assert U.dtype_simple(array(test_df["PayRate"])) == "float"
    assert U.dtype_simple(array(test_df["PayRate"].astype("float32"))) == "float"
    assert U.dtype_simple(array(test_df["PayRate"].astype("float16"))) == "float"


def test_dtypes_simple_pd_int():
    test_df = make_test_dataframe()
    assert U.dtype_simple(test_df["DoB"]) == "int"


def test_dtypes_simple_np_int():
    test_df = make_test_dataframe()
    assert U.dtype_simple(array(test_df["DoB"])) == "int"


def test_dtypes_simple_pd_bool():
    test_df = make_test_dataframe()
    assert U.dtype_simple(test_df["Married"]) == "bool"


def test_dtypes_simple_np_bool():
    test_df = make_test_dataframe()
    assert U.dtype_simple(array(test_df["Married"])) == "bool"


def test_dtypes_simple_pd_str():
    test_df = make_test_dataframe()
    assert U.dtype_simple(test_df["Name"]) == "str"


def test_dtypes_simple_np_str():
    test_df = make_test_dataframe()
    assert U.dtype_simple(array(test_df["Name"])) == "str"


def test_dtypes_simple_pd_complex():
    test_df = make_test_dataframe(Complex=True)
    assert U.dtype_simple(test_df["Imaginary"]) == "complex"


def test_dtypes_simple_np_complex():
    test_df = make_test_dataframe(Complex=True)
    assert U.dtype_simple(array(test_df["Imaginary"])) == "complex"


def test_dtypes_simple_pd_cat():
    test_df = make_test_dataframe(Categorical=True)
    assert U.dtype_simple(test_df["City"]) == "category"


def test_dtypes_simple_np_cat():
    test_df = make_test_dataframe(Categorical=True)
    assert U.dtype_simple(array(test_df["City"])) == "str"


def test_dtypes_simple_pd_timedelta():
    test_df = make_test_dataframe(Time_Diff=True)
    assert U.dtype_simple(test_df["Hire_Age"]) == "timedelta"


def test_dtypes_simple_np_timedelta():
    test_df = make_test_dataframe(Time_Diff=True)
    assert U.dtype_simple(array(test_df["Hire_Age"])) == "timedelta"


def test_dtypes_simple_pd_datetime():
    test_df = make_test_dataframe()
    assert U.dtype_simple(test_df["Birth_Date"]) == "datetime"


def test_dtypes_simple_np_datetime():
    test_df = make_test_dataframe()
    assert U.dtype_simple(array(test_df["Birth_Date"])) == "datetime"


# DATATABLE


def test_dtypes_simple_dt_float():
    test_df = Frame(make_test_dataframe())
    assert U.dtype_simple(test_df["PayRate"]) == "float"


def test_dtypes_simple_dt_int():
    test_df = Frame(make_test_dataframe())
    assert U.dtype_simple(test_df["DoB"]) == "int"


def test_dtypes_simple_dt_bool():
    test_df = Frame(make_test_dataframe())
    assert U.dtype_simple(test_df["Married"]) == "bool"


def test_dtypes_simple_dt_str():
    test_df = Frame(make_test_dataframe())
    assert U.dtype_simple(test_df["Name"]) == "str"


def test_dtypes_simple_dt_complex():
    with pytest.raises(ValueError):
        Frame(make_test_dataframe(Complex=True))


def test_dtypes_simple_dt_categorical():
    test_df = Frame(make_test_dataframe(Categorical=True))
    assert U.dtype_simple(test_df["City"]) == "str"


def test_dtypes_simple_dt_datetime():
    test_df = Frame(make_test_dataframe())
    assert U.dtype_simple(test_df["Birth_Date"]) == "datetime"


def test_dtypes_simple_dt_timedelta():
    test_df = Frame(make_test_dataframe(Time_Diff=True))
    assert U.dtype_simple(test_df["Hire_Age"]) == "str"
