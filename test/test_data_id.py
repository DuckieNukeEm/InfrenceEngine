from numpy import nan, inf
import numpy as np
from datatable import Frame, f, time
from dataframe import make_test_dataframe
import pytest
import sys

sys.path.insert(0, "../src")
import data_id as di

test_df = make_test_dataframe()

# #############################################################################
# #############################################################################
# replace Non Numeric
# #############################################################################
# #############################################################################


def test_replace_nonnumeric():
    assert di.replace_nonnumeric(1) == 1
    assert di.replace_nonnumeric(1.1) == 1.1
    assert di.replace_nonnumeric(1, 0) == 1
    assert di.replace_nonnumeric(nan, 2) == 2
    assert di.replace_nonnumeric(inf, 2) == 2
    assert di.replace_nonnumeric(inf * -1, 2) == 2
    assert di.replace_nonnumeric(0, 1, False) == 0
    assert di.replace_nonnumeric(0, 1, True) == 1


# #############################################################################
# #############################################################################
# clip edges
# #############################################################################
# #############################################################################


def test_clip_edges_works():
    test_col = test_df.sort_values(["Hieght"]).Hieght.values
    ret = di.clip_edges(test_col, 0.1, 0.8)
    assert len(ret) == 6
    assert np.sum(ret == [92, 149, 170, 178, 179, 180]) == 6


def test_clip_edge_pct_upper():
    test_col = test_df.sort_values(["Hieght"]).Hieght.values
    with pytest.raises(AssertionError) as err:
        di.clip_edge(test_col, 0.1, 5)
    assert "pct_upper needs to be of type int" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.clip_edge(test_col, 0.1, 5.0)
    assert "pct_upper needs to be at most 1.0 and greater than 0.0" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.clip_edge(test_col, 0.1, 0.0)
    assert "pct_upper needs to be at most 1.0 and greater than 0.0" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.clip_edge(test_col, 0.1, -0.5)
    assert "pct_upper needs to be at most 1.0 and greater than 0.0" in str(err.value)


def test_clip_edge_lower_upper():
    test_col = test_df.sort_values(["Hieght"]).Hieght.values
    with pytest.raises(AssertionError) as err:
        di.clip_edge(test_col, -4, 0.8)
    assert "pct_lower needs to be of type int" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.clip_edge(test_col, 1.2, 5.0)
    assert "pct_lower needs to be at least 0.0 and less than 1.0" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.clip_edge(test_col, 1.00, 0.0)
    assert "pct_lower needs to be at least 0.0 and less than 1.0" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.clip_edge(test_col, -0.1, -0.5)
    assert "pct_lower needs to be at least 0.0 and less than 1.0" in str(err.value)

    res = di.clip_edge(test_col, 0.0, 1.0)
    assert len(res) == 8


# #############################################################################
# #############################################################################
# standardize
# #############################################################################
# #############################################################################


def test_standardize_works():
    Df = make_test_dataframe()
    res = di.standardize(Df.Flag)
    assert np.sum(res == [1, 1, -1, -1, 1, -1, -1, 1, 1]) == 6


def test_standardize_np_works():
    Df = make_test_dataframe()
    res = di.standardize(Df.Flag.values)
    assert np.sum(res == [1, 1, -1, -1, 1, -1, -1, 1, 1]) == 6


def test_standardize_not_right_data():
    with pytest.raises(AssertionError) as err:
        di.standardize([1, 2, 3])
    assert "Data for standarize isn't a numpy array or a pandas Series" in str(
        err.value
    )


def test_standardize_not_numerical():
    Df = make_test_dataframe()
    with pytest.raises(AssertionError) as err:
        di.standardize(Df.Hieght)
    assert "Column provided to standardize isn't numerical" in str(err.value)


# #############################################################################
# #############################################################################
# Data Frequency
# #############################################################################
# #############################################################################


def test_data_frequency():
    Df = make_test_dataframe()
    res1, res2 = di.data_frequency(Df.DoB.values, 4)
    assert np.sum(res1 == np.array([2, 3, 3])) == 3
    assert np.sum(res2 == np.array([1901, 1953, 1953.66666667, 1993])) == 4


def test_data_frequency_not_int():
    Df = make_test_dataframe()
    with pytest.raises(AssertionError) as err:
        di.data_frequency(Df.Dob.values, "a")
    assert "Buckets needs to be an int" in str(err.value)


def test_data_frequency_not_np():
    Df = make_test_dataframe()
    with pytest.raises(AssertionError) as err:
        di.data_frequency(Df.Dob)
    assert "Data need to be an array" in str(err.value)
