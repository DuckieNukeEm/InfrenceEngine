import numpy as np
from scipy import stats
from typing import Union
from augmentation import Augmentation as Aug
from .numpy.data_types import data_types
import util as U


class Augmentation(Aug):
    """augmentation class built in numpy

    This class provides the tools that will modify, or 'augment' the data, and that is it's sole purpose

    This will also be the base structure for other augmentation classes using different packages
    """

    def __init__(self):
        self.data_types = data_types()

    def replace_single_nonnumeric(
        self,
        Num: Union[float, int],
        replace_value: Union[int, float] = 0,
        check_zero: bool = False,
    ) -> Union[float, int]:
        """replaces a number if it's a inf, nan, or null (NONE)

        Arguments:
            Num {float, int} -- value to replace if inf, null, or nan

        Keyword Arguments:
            replace_value {int} -- what to replace that value with (default: {0})
            check_zero {bool} -- additional chekc to make sure it isn't zero

        Returns:
            [float, int] -- return value

        Details:
            takes an input number, Num, and checks to see if it's a inf, Nan, or none.
            If it's any of them, it replaces that number with the replace_value. If it's
            none of them, just returns the value
        """
        if Num is None or np.isnan(Num) or np.isinf(Num):
            return replace_value
        elif check_zero is True and Num == 0:
            return replace_value
        else:
            return Num

    def clip_edges(Data: np.array, pct_lower: float = 0.001, pct_upper: float = 0.99):
        """WIll clipp the upper % and lower % off of the index

        Arguments:
            Data {np.Array} -- the input array to clip

        Keyword Arguments:
            pct_lower {float} -- the lower bound to clip (IE anything below lower_bound % is removed)
            pct_upper {float} -- the upper bound to clip (anything above the value is removed)

        Returns:
            np.array


        """
        if U.typeof(pct_lower) != "Float":
            raise TypeError("pct_lower needs to be of type float")

        if U.typeof(pct_upper) != "Float":
            raise TypeError("pct_upper needs to be of type float")

        if pct_lower < 0.0 or pct_lower >= 1.00:
            raise ValueError("pct_lower needs to be at least 0.0 and less than 1.0")

        if pct_lower <= 0.0 or pct_upper > 1.00:
            raise ValueError("pct_upper needs to be at most 1.0 and greater than 0.0")

        if pct_upper < pct_lower:
            pct_lower, pct_upper = pct_upper, pct_lower

        data_length = Data.size
        Data_Clip = Data[int(pct_lower * data_length) : int(data_length * pct_upper)]

        return Data_Clip

    def standardize(self, Data: np.array) -> np.array:
        """Will subtract she mean and divided by the std

        Arguments:
            f {pd.Series, np.array} -- input numerical feature to standardize

        returns:
            np.array
        """

        if Data.dtype not in self.numeric:
            raise TypeError("Data provided to standardize isn't numerical")

        v_sd = np.std(Data)
        v_sd = self.replace_single_nonnumeric(v_sd, 1, True)

        v_mean = np.mean(Data)
        v_mean = self.replace_single_nonnumeric(v_mean, 0)

        sdz_data = (Data - v_mean) / v_sd

        return sdz_data
