import pandas as pd
import numpy as np 
from typing import Union

import datatable as dt



def flag_via_sd(df: Union[pd.DataFrame, dt.datatable]):
    pass

def flag_sd_pd(df: pd.Dataframe, sd: float = 1.96, vars: list = [], **kwargs) -> pd.DataFrame:
    """flaggs records that fall out of range for an SD

    Args:
        df (pd.Dataframe):  input dataset
        sd (float): standard deviations out to flag as outliers (two sided)
                    (default: 1.96)
        vars (list of str or ints): list of variables to use to calculate
                                    standard deviations against

    Returns:
        pd.DataFrame: [description]
    """
    


def flag_sd_dt(df: df.datatable, sd: float = 1.96, vars: list = [], **kwargs):
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

    