import pandas as pd
import numpy as np 
from typing import Union

import datatable as dt


def flag_via_sd(df: Union[pd.DataFrame, dt.datatable]):
    pass


def flag_sd_dt(df: df.datatable, sd: float = 1.96, vars: list = []) -> df.datatable:
    """a function that calculates standard deviations across

    Arguments:
        df {datatable}  --   input datatable with variables to filter out

    Keyword Arguments:
        sd {float}  --  standard deviations out to flag as outliers (two sided)
                        (default: 1.96)
        vars {list of str or ints}  --  list of variables to use to calculate
                                        SD against. If it's a list of str, it
                                        will try to match the strings to var
                                        names, if it's a list of ints, will
                                        use the positional arguemnts.

    Raises:
        None

    Returns:
        datatable datatabke
    
    Details:
        This is the flag_sd varition for us with data table. 

        It will ittertate through the list of vars, and will calculate the SD
        of each variables, 
    
    """

    pass

    