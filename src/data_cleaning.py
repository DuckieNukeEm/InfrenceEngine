import pandas as pd
import numpy as np 
from typing import Union

import datatable as dt



def flag_via_sd(df: Union[pd.DataFrame, dt.datatable]):
    pass

def flag_sd_dt(df: df.datatable, sd: float = 1.96, vars: list = []):
    """a function that calculates standard deviations across

    Arguments:
        df {datatable} -- input datatable with variables to filter out
        sd {float} -- standard deviations out to flag as outliers (two sided)
                     (default: 1.96)
        vars {list of str or ints} -- list of variables to cacluate SD against
                        

    Keyword Arguments:
sadas

    Raises:

    Returns:

    Details:"""

    pass

    