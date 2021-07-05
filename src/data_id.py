import pandas as pd
import numpy as np
from typing import Union
import util as U


def standardize(Data: Union[pd.Series, np.array]) -> np.array:
    """Will make the mean zero and std 1

    Arguments:
        f {Union[pd.Series, np.array]} -- input numerical feature to standardize

    returns:
        np.array (if input is np.array)
    """

    assert U.typeof(Data) in (
        "Array",
        "Series",
    ), "Data for standarize isn't a numpy array or a pandas Series"
