from pandas import DataFrame, Series
from datatable import Frame, f
from test_dataframe import make_test_dataframe
import pytest
import sys

sys.path.insert(0, "../src")
import data_cleaning as dc


# Creating common params
use_columns = ["Name", "DoB", "Hieght", "City"]
STD_P = 2.00
OL_null_p = []
OL_nm_p = ["DoB"]
OL_pos_p = [1]
TG_null_p = {}
TG_p = {"City": "Tokyo"}

test_df = make_test_dataframe(cols=use_columns)
# #############################################################################
# #############################################################################
# flag_outlier_sd_pandas
# #############################################################################
# #############################################################################


# ###
# basic assert
# ###


def test_flag_outliers_sd_pandas_df_assert():
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_pandas(
            Df=["1", "2", "3"], STD=STD_P, outlier_vars=OL_null_p, test_group=TG_null_p
        )
    assert "the object passed to flag_sd is not a pandas Dataframe" in str(err.value)


def test_flag_outliers_sd_pandas_STD_assert():
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_pandas(
            Df=test_df, STD="A", outlier_vars=OL_null_p, test_group=TG_null_p
        )
    assert "the variable STD is not of type Float" in str(err.value)


def test_flag_outliers_sd_pandas_outlier_Var_assert():
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_pandas(
            Df=test_df.DoB, STD=STD_P, outlier_vars=1, test_group=TG_null_p
        )
    assert "the variable outlier_vars is not a list" in str(err.value)


def test_flag_outliers_sd_pandas_test_group_assert():
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_pandas(
            Df=test_df, STD=STD_P, outlier_vars=OL_null_p, test_group="b"
        )
    assert "the variable test_group needs to be a dict" in str(err.value)


# ###
# Advanced assert
# ###


def test_flag_outliers_sd_pandas_unkown_outlier_vars():
    with pytest.raises(TypeError) as err:
        dc.flag_outliers_sd_pandas(
            Df=test_df,
            STD=STD_P,
            outlier_vars=["dob", {"ted": 1}],
            test_group=TG_null_p,
        )
    assert "Can not identify the elements in vars" in str(err.value)


def test_flag_outliers_sd_pandas_mixed_outlier_vars():
    with pytest.raises(TypeError) as err:
        dc.flag_outliers_sd_pandas(
            Df=test_df, STD=STD_P, outlier_vars=["dob", 1], test_group=TG_null_p
        )
    assert "the elements of the list Vars are of mixed type" in str(err.value)


def test_flag_outliers_sd_pandas_too_many_outlier_vars():
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_pandas(
            Df=test_df, STD=STD_P, outlier_vars=[1, 2, 3, 4, 5], test_group=TG_null_p
        )
    assert "Max value of outlier_vars exceeds the dataframe shape" in str(err.value)


def test_flag_outliers_sd_pandas_too_many_keys_test_group():
    with pytest.raises(KeyError) as err:
        dc.flag_outliers_sd_pandas(
            Df=test_df,
            STD=STD_P,
            outlier_vars=OL_null_p,
            test_group={"City": "Tokyo", "DoB": 1985},
        )
    assert (
        "More keys exists than can be handeld in test_group. Plan to support multiple SD in the future"
        in str(err.value)
    )


def test_flag_outliers_sd_pandas_key_not_in_list():
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_pandas(
            Df=test_df,
            STD=STD_P,
            outlier_vars=OL_null_p,
            test_group={"BobsBurgers": "Linda"},
        )
    assert "Key of test_group needs to be a column name" in str(err.value)


def test_flag_outliers_sd_pandas_value_not_in_list():
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_pandas(
            Df=test_df,
            STD=STD_P,
            outlier_vars=OL_null_p,
            test_group={"City": "Las Vegas"},
        )
    assert "Value of test_group needs to be an element in the Data Frame" in str(
        err.value
    )


# ###
# calculation test
# ###


def test_flag_outliers_all_vars_correct():
    test_df = make_test_dataframe(cols=use_columns)
    return_val = dc.flag_outliers_sd_pandas(
        Df=test_df, STD=STD_P, outlier_vars=OL_null_p, test_group=TG_null_p
    )
    # Manually calculated the SD by hand and found the records that would have the delta
    assert return_val.Outlier.sum() == 2
    assert return_val.iloc[0, 4] == 1
    assert return_val.iloc[1, 4] == 1


def test_flag_outliers_sel_vars_correct():
    test_df = make_test_dataframe(cols=use_columns)
    return_val = dc.flag_outliers_sd_pandas(
        Df=test_df, STD=STD_P, outlier_vars=OL_nm_p, test_group=TG_null_p
    )
    # Manually calculated the SD by hand and found the records that would have the delta
    assert return_val.Outlier.sum() == 1
    assert return_val.iloc[0, 4] == 1
    assert return_val.iloc[1, 4] == 0


def test_flag_outliers_all_vars_test_groupcorrect():
    test_df = make_test_dataframe(cols=use_columns)
    return_val = dc.flag_outliers_sd_pandas(
        Df=test_df, STD=STD_P, outlier_vars=OL_null_p, test_group=TG_p
    )
    # Manually calculated the SD by hand and found the records that would have the delta
    assert return_val.Outlier.sum() == 3
    assert return_val.iloc[0, 4] == 0
    assert return_val.iloc[1, 4] == 2
    assert return_val.iloc[7, 4] == 1


def test_flag_outliers_sel_vars_test_groupcorrect():
    test_df = make_test_dataframe(cols=use_columns)
    return_val = dc.flag_outliers_sd_pandas(
        Df=test_df, STD=STD_P, outlier_vars=OL_nm_p, test_group=TG_p
    )
    # Manually calculated the SD by hand and found the records that would have the delta
    assert return_val.Outlier.sum() == 1
    assert return_val.iloc[0, 4] == 0
    assert return_val.iloc[1, 4] == 1
    assert return_val.iloc[7, 4] == 0


# #############################################################################
# #############################################################################
# flag_outlier_sd_datatable
# #############################################################################
# #############################################################################


# ###
# basic assert
# ###


def test_flag_outliers_sd_datatable_df_assert():
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_datatable(
            Df=["1", "2", "3"], STD=STD_P, outlier_vars=OL_null_p, test_group=TG_null_p
        )
    assert "the object passed to flag_sd is not a datatable Frame" in str(err.value)


def test_flag_outliers_sd_datatable_STD_assert():
    test_dt = Frame(make_test_dataframe(cols=use_columns))
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_datatable(
            Df=test_dt, STD="A", outlier_vars=OL_null_p, test_group=TG_null_p
        )
    assert "the variable STD is not of type Float" in str(err.value)


def test_flag_outliers_sd_datatable_outlier_Var_assert():
    test_dt = Frame(make_test_dataframe(cols=use_columns))
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_datatable(
            Df=test_dt[:, f.DoB], STD=STD_P, outlier_vars=1, test_group=TG_null_p
        )
    assert "the variable outlier_vars is not a list" in str(err.value)


def test_flag_outliers_sd_datatable_test_group_assert():
    test_dt = Frame(make_test_dataframe(cols=use_columns))
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_datatable(
            Df=test_dt, STD=STD_P, outlier_vars=OL_null_p, test_group="b"
        )
    assert "the variable test_group needs to be a dict" in str(err.value)


# ###
# Advanced assert
# ###


def test_flag_outliers_sd_datatable_unkown_outlier_vars():
    test_dt = Frame(make_test_dataframe(cols=use_columns))
    with pytest.raises(TypeError) as err:
        dc.flag_outliers_sd_datatable(
            Df=test_dt,
            STD=STD_P,
            outlier_vars=["dob", {"ted": 1}],
            test_group=TG_null_p,
        )
    assert "Can not identify the elements in vars" in str(err.value)


def test_flag_outliers_sd_datatable_mixed_outlier_vars():
    test_dt = Frame(make_test_dataframe(cols=use_columns))
    with pytest.raises(TypeError) as err:
        dc.flag_outliers_sd_datatable(
            Df=test_dt, STD=STD_P, outlier_vars=["dob", 1], test_group=TG_null_p
        )
    assert "the elements of the list Vars are of mixed type" in str(err.value)


def test_flag_outliers_sd_datatable_too_many_outlier_vars():
    test_dt = Frame(make_test_dataframe(cols=use_columns))
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_datatable(
            Df=test_dt, STD=STD_P, outlier_vars=[1, 2, 3, 4, 5], test_group=TG_null_p
        )
    assert "Max value of outlier_vars exceeds the Frame shape" in str(err.value)


def test_flag_outliers_sd_datatable_too_many_keys_test_group():
    test_dt = Frame(make_test_dataframe(cols=use_columns))
    with pytest.raises(KeyError) as err:
        dc.flag_outliers_sd_datatable(
            Df=test_dt,
            STD=STD_P,
            outlier_vars=OL_null_p,
            test_group={"City": "Tokyo", "DoB": 1985},
        )
    assert (
        "More keys exists than can be handeld in test_group. Plan to support multiple SD in the future"
        in str(err.value)
    )


def test_flag_outliers_sd_datatable_key_not_in_list():
    test_dt = Frame(make_test_dataframe(cols=use_columns))
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_datatable(
            Df=test_dt,
            STD=STD_P,
            outlier_vars=OL_null_p,
            test_group={"BobsBurgers": "Linda"},
        )
    assert "Key of test_group needs to be a column name" in str(err.value)


def test_flag_outliers_sd_datatable_value_not_in_list():
    test_dt = Frame(make_test_dataframe(cols=use_columns))
    with pytest.raises(AssertionError) as err:
        dc.flag_outliers_sd_datatable(
            Df=test_dt,
            STD=STD_P,
            outlier_vars=OL_null_p,
            test_group={"City": "Las Vegas"},
        )
    assert "Value of test_group needs to be an element in the datatable Frame" in str(
        err.value
    )


# ###
# calculation test
# ###


def test_flag_outliers_all_vars_correct_dt():
    test_dt = Frame(make_test_dataframe(cols=use_columns))
    return_val = dc.flag_outliers_sd_datatable(
        Df=test_dt[:, 0:4], STD=STD_P, outlier_vars=OL_null_p, test_group=TG_null_p
    )
    # Manually calculated the SD by hand and found the records that would have the delta
    assert return_val["Outlier"].sum()[0, 0] == 2
    assert return_val[0, 4] == 1
    assert return_val[1, 4] == 1


def test_flag_outliers_sel_vars_correct_dt():
    test_dt = Frame(make_test_dataframe(cols=use_columns))
    return_val = dc.flag_outliers_sd_datatable(
        Df=test_dt, STD=STD_P, outlier_vars=OL_nm_p, test_group=TG_null_p
    )
    # Manually calculated the SD by hand and found the records that would have the delta
    assert return_val["Outlier"].sum()[0, 0] == 1
    assert return_val[0, 4] == 1
    assert return_val[1, 4] == 0


def test_flag_outliers_all_vars_test_groupcorrect_dt():
    test_dt = Frame(make_test_dataframe(cols=use_columns))
    return_val = dc.flag_outliers_sd_datatable(
        Df=test_dt[:, 0:4], STD=STD_P, outlier_vars=OL_null_p, test_group=TG_p
    )
    # Manually calculated the SD by hand and found the records that would have the delta
    assert return_val["Outlier"].sum()[0, 0] == 3
    assert return_val[0, 4] == 0
    assert return_val[1, 4] == 2
    assert return_val[7, 4] == 1


def test_flag_outliers_sel_vars_test_groupcorrect_dt():
    test_dt = Frame(make_test_dataframe(cols=use_columns))
    return_val = dc.flag_outliers_sd_datatable(
        Df=test_dt, STD=STD_P, outlier_vars=OL_nm_p, test_group=TG_p
    )
    # Manually calculated the SD by hand and found the records that would have the delta
    assert return_val["Outlier"].sum()[0, 0] == 1
    assert return_val[0, 4] == 0
    assert return_val[1, 4] == 1
    assert return_val[7, 4] == 0


# #############################################################################
# #############################################################################
# boxcox
# #############################################################################
# #############################################################################
