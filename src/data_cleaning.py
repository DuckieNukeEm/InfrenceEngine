import pandas as pd
import numpy as np
from typing import Union
from datatable import f
import util as U

import datatable as dt


def flag_via_sd(Df: Union[pd.DataFrame, dt.Frame]):
    pass


def flag_outliers_sd_pandas(
    Df: pd.DataFrame,
    STD: float = 1.96,
    outlier_vars: list = [],
    test_group: dict = {},
    v: bool = False,
    **kwargs
) -> pd.DataFrame:
    """flaggs records that fall out of range for an SD

    Args:
        Df (pd.Dataframe):  input dataset
        STD (float):    standard deviations out to flag as outliers (two sided)
                        (default: 1.96)
        outlier_vars (list of str or ints): list of variables to check for
                                            outliers using number of standard
                                            devaitions (STD) from the mean.
                                            A empty list uses the entire
                                            data frame.
                                            (default: [])
        test_group (dict):  a dictionary containing key-value pairs of:
                                key: the name of the variable that contains
                                      filter
                                value: the value of the group to get the STD
                                       limits on and apply to their couterpart
        v (bool): have it natter at you while it runs
                        (default: False)
    Returns:
        pd.DataFrame: same as before, except with a new column: 'Outlier'

    Details:
         This uses the pandas dataframe framework to do the calculations

        It will ittertate through the list of vars, and will calculate the SD
        of each variables,

    """
    # basic checks
    assert U.typeof(Df) in [
        "DataFrame",
        "Series",
    ], "the object passed to flag_sd is not a pandas Dataframe"

    assert U.typeof(outlier_vars) == "List", "the variable outlier_vars is not a list"

    assert U.typeof(STD) == "Float", "the variable STD is not of type Float"

    assert U.typeof(test_group) == "Dict", "the variable test_group needs to be a dict"

    STD = abs(STD)

    # medium checks
    if len(outlier_vars) == 0:
        outlier_vars = list(Df.columns)

    int_flag_check = 0
    for v in outlier_vars:
        if isinstance(v, int):
            int_flag_check = int_flag_check + 1
        elif isinstance(v, str):
            pass
        else:
            raise TypeError("Can not identify the elements in vars")

    if int_flag_check != 0 and int_flag_check != len(outlier_vars):
        raise TypeError("the elements of the list Vars are of mixed type")

    if int_flag_check == len(outlier_vars):
        if max(outlier_vars) >= Df.shape[1]:
            raise AssertionError(
                "Max value of outlier_vars exceeds the dataframe shape"
            )

    if int_flag_check > 0:
        outlier_vars = Df.columns[outlier_vars]

    if len(test_group) == 1:
        for key, value in test_group.items():
            assert key in list(
                Df.columns
            ), "Key of test_group needs to be a column name"
            assert (
                value in Df[key].values
            ), "Value of test_group needs to be an element in the Data Frame"
            test_col = key
            test_value = value

    elif len(test_group) > 1:
        raise KeyError(
            "More keys exists than can be handeld in test_group. Plan to support multiple SD in the future"
        )
    else:
        pass

    # making sure the set of variables is unique
    outlier_vars = set(outlier_vars)

    # lets do this!

    if len(test_group) == 1:
        sd_vals = Df.loc[np.in1d(Df[test_col], test_value), outlier_vars].std()
        mean_vals = Df.loc[np.in1d(Df[test_col], test_value), outlier_vars].mean()

    else:
        sd_vals = Df[outlier_vars].std()
        mean_vals = Df[outlier_vars].mean()

    U.v_print(v, "STD:", sd_vals)
    U.v_print(v, "Means:", mean_vals)

    # walking through and replacing nans with infs
    vars_to_remove = []
    for col in test_col:
        if sd_vals[col] == np.nan:
            U.v_print(
                v,
                "the following variable as it has an STD of nan: %s removing from STD flag"
                % (col),
            )
            vars_to_remove.append(col)

        if mean_vals[col] == np.nan or abs(mean_vals[col]) == np.inf:
            U.v_print(
                v,
                "Removing the following variables as it has a mean of nan or inf: %s"
                % (col),
            )
            vars_to_remove.append(col)

    if len(vars_to_remove) > 0:
        vars_to_remove = set(vars_to_remove)
        outlier_vars = [x for x in outlier_vars if x not in vars_to_remove]

    lower_bound = mean_vals[outlier_vars] - STD * sd_vals[outlier_vars]
    upper_bound = mean_vals[outlier_vars] + STD * sd_vals[outlier_vars]

    U.v_print(v, "Lower Bounds:", lower_bound)
    U.v_print(v, "Upper Bounds:", upper_bound)

    Outliers = (
        ((Df[outlier_vars] > upper_bound) | (Df[outlier_vars] < lower_bound))
        .astype(int)
        .sum(axis=1)
    )

    Df["Outlier"] = Outliers

    return Df


def flag_outliers_sd(
    Df: dt.Frame,
    STD: float = 1.96,
    outlier_vars: list = [],
    test_group: dict = {},
    v: bool = False,
    **kwargs
) -> dt.Frame:
    """flaggs records that fall out of range for an SD using a datatable Frame

    Args:
        Df (datatable Drame):  input dataset
        STD (float):    standard deviations out to flag as outliers (two sided)
                        (default: 1.96)
        outlier_vars (list of str or ints): list of variables to check for
                                            outliers using number of standard
                                            devaitions (STD) from the mean.
                                            A empty list uses the entire
                                            data frame.
                                            (default: [])
        test_group (dict):  a dictionary containing key-value pairs of:
                                key: the name of the variable that contains
                                      filter
                                value: the value of the group to get the STD
                                       limits on and apply to their couterpart
        v (bool): have it natter at you while it runs
                        (default: False)
    Returns:
        datatable Frame: same as before, except with a new column: 'Outlier'

    Details:

    """
    assert (
        U.typeof(Df) == "Frame"
    ), "the object passed to flag_sd is not a datatable Frame"

    assert U.typeof(outlier_vars) == "List", "the variable outlier_vars is not a list"

    assert U.typeof(STD) == "Float", "the variable STD is not of type Float"

    assert U.typeof(test_group) == "Dict", "the variable test_group needs to be a dict"

    STD = abs(STD)

    # medium checks
    if len(outlier_vars) == 0:
        outlier_vars = list(Df.names)

    int_flag_check = 0
    for v in outlier_vars:
        if isinstance(v, int):
            int_flag_check = int_flag_check + 1
        elif isinstance(v, str):
            pass
        else:
            raise TypeError("Can not identify the elements in vars")

    if int_flag_check != 0 and int_flag_check != len(outlier_vars):
        raise TypeError("the elements of the list Vars are of mixed type")

    if int_flag_check == len(outlier_vars):
        if max(outlier_vars) >= Df.ncols:
            raise AssertionError("Max value of outlier_vars exceeds the Frame shape")

    if int_flag_check > 0:
        outlier_vars = Df.columns[outlier_vars]

    if len(test_group) == 1:
        for key, value in test_group.items():
            assert key in list(Df.names), "Key of test_group needs to be a column name"
            assert any(
                Df[f[key] == value, :]
            ), "Value of test_group needs to be an element in the Data Frame"
            test_col = key
            test_value = value

    elif len(test_group) > 1:
        raise KeyError(
            "More keys exists than can be handeld in test_group. Plan to support multiple SD in the future"
        )
    else:
        pass

    outlier_vars = set(outlier_vars)

    # lets do this!

    if len(test_group) == 1:
        sd_vals = Df[f[test_col] == test_value, outlier_vars].sd()
        mean_vals = Df[f[test_col] == test_value, outlier_vars].mean()

    else:
        sd_vals = Df[:, outlier_vars].sd()
        mean_vals = Df[:, outlier_vars].mean()

    U.v_print(v, "STD:", sd_vals)
    U.v_print(v, "Means:", mean_vals)

    lower_bound = {}
    upper_bound = {}
    for col in outlier_vars:
        lower_bound[col] = mean_vals[0, col] - STD * sd_vals[0, col]
        upper_bound[col] = mean_vals[0, col] + STD * sd_vals[0, col]

    U.v_print(v, "Lower Bounds:", lower_bound)
    U.v_print(v, "Upper Bounds:", upper_bound)

    Df = Df[:, {"Outlier": 0}]
    for col in outlier_vars:
        Df[
            :,
            {
                "Outlier": dt.ifelse(
                    f[col] < lower_bound[col], 1, f[col] > upper_bound[col], 1, 0
                )
                + f.Outlier
            },
        ]

    return Df
