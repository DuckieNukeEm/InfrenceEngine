import pandas as pd
from typing import Union
import util as U

import datatable as dt


def flag_via_sd(Df: Union[pd.DataFrame, dt.datatable]):
    pass


def flag_sd_pandas(
    Df: pd.Dataframe, STD: float = 1.96, outlier_vars: list = [], **kwargs
) -> pd.DataFrame:
    """flaggs records that fall out of range for an SD

    Args:
        Df (pd.Dataframe):  input dataset
        STD (float): standard deviations out to flag as outliers (two sided)
                    (default: 1.96)
        outlier_vars (list of str or ints): list of variables to check for
                                            outliers using number of standard
                                            devaitions (STD) from the mean.
                                            A empty list uses the entire
                                            data frame.
                                    (default: [])
    Returns:
        pd.DataFrame: [description]
    """
    # basic checks
    assert U.typeof(Df) in [
        "DataFrame",
        "Series",
    ], "the object passed to flag_sd is not a pandas Dataframe"

    assert U.typeof(outlier_vars) == "List", "the variable Vars is not a list"

    assert U.typeof(STD) == "Float", "the variable sd is not of type float"

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

    if int_flag_check != 0 and int_flag_check != len(vars):
        raise TypeError("the elements of the list Vars are of mixed type")

    if int_flag_check > 0:
        outlier_vars = Df.columns[outlier_vars]

    # making sure the set of variables is unique
    outlier_vars = set(outlier_vars)

    # lets do this!
    sd_vals = Df[outlier_vars].std()
    mean_vals = Df[outlier_vars].mean()

    lower_bound = mean_vals - STD * sd_vals
    upper_bound = mean_vals + STD * sd_vals

    Outliers = (
        ((Df[outlier_vars] > upper_bound) | (Df[outlier_vars] < lower_bound))
        .astype(int)
        .sum(axis=1)
    )

    Df["Outlier"] = Outliers

    return Df


def flag_sd_dt(Df: dt.datatable, sd: float = 1.96, vars: list = [], **kwargs):
    """a function that calculates standard deviations across

    Arguments:
        df (datatable): input datatable with variables to filter out
        sd (float): standard deviations out to flag as outliers (two sided)
                        (default: 1.96)
        vars (list of str or ints): list of variables to use to calculate
                                    standard deviations against

    Raises:
        None

    Returns:
        datatable Frame that has a new (or overwritten) variables called
        outlier. It's indicates the number of variables that a specific
        record  was beyond the STD of the entire variable. The number can
        range from 0 to n, the number of variables in vars.

    Details:
        This uses the dataframe framework to do the calculations

        It will ittertate through the list of vars, and will calculate the SD
        of each variables,

    """

    pass
