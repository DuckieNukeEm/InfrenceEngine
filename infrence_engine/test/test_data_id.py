from numpy import nan, inf
import numpy as np
from datatable import Frame, f, time
from test_dataframe import make_test_dataframe
import sys
import pytest


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
        di.clip_edges(test_col, 0.1, 5)
    assert "pct_upper needs to be of type float" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.clip_edges(test_col, 0.1, 5.0)
    assert "pct_upper needs to be at most 1.0 and greater than 0.0" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.clip_edges(test_col, 0.1, 0.0)
    assert "pct_upper needs to be at most 1.0 and greater than 0.0" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.clip_edges(test_col, 0.1, -0.5)
    assert "pct_upper needs to be at most 1.0 and greater than 0.0" in str(err.value)


def test_clip_edge_lower_upper():
    test_col = test_df.sort_values(["Hieght"]).Hieght.values
    with pytest.raises(AssertionError) as err:
        di.clip_edges(test_col, -4, 0.8)
    assert "pct_lower needs to be of type float" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.clip_edges(test_col, 1.2, 5.0)
    assert "pct_lower needs to be at least 0.0 and less than 1.0" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.clip_edges(test_col, 1.00, 0.0)
    assert "pct_lower needs to be at least 0.0 and less than 1.0" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.clip_edges(test_col, -0.1, -0.5)
    assert "pct_lower needs to be at least 0.0 and less than 1.0" in str(err.value)

    res = di.clip_edges(test_col, 0.0, 1.0)
    assert len(res) == 8


# #############################################################################
# #############################################################################
# standardize
# #############################################################################
# #############################################################################


def test_standardize_works():
    Df = make_test_dataframe()
    res = di.standardize(Df.Flag)
    assert np.sum(round(res, 0) == [1, 1, -1, -1, 1, -1, -1, 1]) == 8


def test_standardize_np_works():
    Df = make_test_dataframe()
    res = di.standardize(Df.Flag.values)
    assert np.sum(np.round(res, 0) == [1, 1, -1, -1, 1, -1, -1, 1]) == 8


def test_standardize_not_right_data():
    with pytest.raises(AssertionError) as err:
        di.standardize([1, 2, 3])
    assert "Data for standarize isn't a numpy array or a pandas Series" in str(
        err.value
    )


def test_standardize_not_numerical():
    Df = make_test_dataframe()
    with pytest.raises(AssertionError) as err:
        di.standardize(Df.Name)
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
    assert np.sum(np.round(res2, 2) == np.array([1901, 1953, 1953.67, 1993])) == 4


def test_data_frequency_not_int():
    Df = make_test_dataframe()
    with pytest.raises(AssertionError) as err:
        di.data_frequency(Df.DoB.values, "a")
    assert "Buckets needs to be an int" in str(err.value)


def test_data_frequency_not_np():
    Df = make_test_dataframe()
    with pytest.raises(AssertionError) as err:
        di.data_frequency(Df.DoB)
    assert "Data need to be an array" in str(err.value)


# #############################################################################
# #############################################################################
# expectation_of_distro
# #############################################################################
# #############################################################################


def test_expectation_of_distro_non_string():
    Dt = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    Buck = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    with pytest.raises(AssertionError) as err:
        di.expectation_of_distro(Dt, Buck, [])
    assert "Distribution must be a string value" in str(err.value)


def test_expectation_of_distro_bad_distro():
    Dt = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    Buck = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    with pytest.raises(AttributeError) as err:
        di.expectation_of_distro(Dt, Buck, "bobs your uncle")
    assert "module 'scipy.stats' has no attribute" in str(err.value)


def test_expectation_of_distro_bad_bucket():
    Dt = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    with pytest.raises(AssertionError) as err:
        di.expectation_of_distro(Dt, {}, "uniform")
    assert "buckets need to be a list" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.expectation_of_distro(Dt, [], "uniform")
    assert "buckets needs to be a list with at least one element" in str(err.value)


def test_expectation_of_distro_works():
    Dt = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    Buck = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ret1, ret2, ret3 = di.expectation_of_distro(Dt, Buck, "uniform")
    print(ret2)
    assert np.sum(np.round(ret1, 3) == np.array([1.125, 2.25, 2.25, 2.25, 1.125])) == 5
    assert (
        np.sum(np.round(ret2, 3) == np.array([0, 0.125, 0.375, 0.625, 0.875, 1])) == 6
    )
    assert len(ret3) == 2


# #############################################################################
# #############################################################################
# find_distribution
# #############################################################################
# #############################################################################


def test_find_distribution_bad_limits():
    Df = make_test_dataframe()
    with pytest.raises(AssertionError) as err:
        di.find_distribution(Df.DoB.values, ["norm", "uniform"], 3, upper_bound={})
    assert "pct_upper needs to be of type float" in str(err.value)
    with pytest.raises(AssertionError) as err:
        di.find_distribution(Df.DoB.values, ["norm", "uniform"], 3, lower_bound={})
    assert "pct_lower needs to be of type float" in str(err.value)


def test_find_distribution_bad_distro():
    Df = make_test_dataframe()
    with pytest.raises(AssertionError) as err:
        di.find_distribution(Df.DoB.values, {"norm": "uniform"}, 3)
    assert "distribution must be a list" in str(err.value)

    with pytest.raises(AssertionError) as err:
        di.find_distribution(Df.DoB.values, [4], 3)
    assert "Distribution must be a string value" in str(err.value)


def test_find_distribution_bad_bucket():
    Df = make_test_dataframe()
    with pytest.raises(AssertionError) as err:
        di.find_distribution(Df.DoB.values, ["norm", "uniform"], "3")
    assert "Buckets needs to be an int" in str(err.value)


def test_find_distribution_bad_datat():
    Df = make_test_dataframe()
    with pytest.raises(AssertionError) as err:
        di.find_distribution(Df, ["norm", "uniform"], 3)
    assert "Data for standarize isn't a numpy array or a pandas Series" in str(
        err.value
    )


def test_find_distribution():
    Df = make_test_dataframe()
    res = di.find_distribution(Df.DoB.values, ["norm", "uniform"], 3)
    assert np.all(res.Distribution == ["norm", "uniform"])
    assert np.all(res.Chi_Square == [5.0, 5.0])
    assert np.all(res.P_Value == [0.0289, 0.0242])
