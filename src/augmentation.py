import pandas as pd
import numpy as np
from scipy import stats
from typing import Union
from augmentation import Augmentation as Aug
import util as U


class Augmentation(Aug):
    """Base augmentation class

    This class provides the tools that will modify, or 'augment' the data, and that is it's sole purpose

    This is the base class, an future enhancements should be built from this class
    """

    def __init__(self):
        pass

    def replace_single_nonnumeric(
        self, Num: Union[float, int], replace_value: int, check_zero: bool, **kwargs
    ) -> Union[float, int]:
        """replaces a number if it's a inf, nan, or null (NONE)

        Arguments:
            Num {float, int} -- value to replace if inf, null, or nan

        Keyword Arguments:
            replace_value {int} -- what to replace that value with (default: {0})
            check_zero {bool} -- additional chekc to make sure it isn't zero
            **kwargs -- additional parameters to pass in

        Returns:
            [float, int] -- return value

        Details:
            takes an input number, Num, and checks to see if it's a inf, Nan, or none.
            If it's any of them, it replaces that number with the replace_value. If it's
            none of them, just returns the value
        """
        pass

    def clip_edges(self, Data, pct_lower: float, pct_upper: float, **kwargs):
        """WIll clipp the upper % and lower % off of the index

        Arguments:
            Data {np.Array} -- the input array to clip

        Keyword Arguments:
            pct_lower {float} -- the lower bound to clip (IE anything below lower_bound % is removed)
            pct_upper {float} -- the upper bound to clip (anything above the value is removed)

        Returns:
            np.array


        """
        pass

    def standardize(self, Data):
        """Will subtract the mean and divided by the std

        Arguments:
            Data -- input numerical feature to standardize

        returns:
            Data, but standardized
        """
        pass

    def data_frequency(self, Data, Buckets: int = 10, **kwargs):
        """will order the data into bins and provide a frequency of the data

        Arguments:
            Data -- an input data

        Keyword Arguments:
            Buckets {int} -- Number of buckets to sort the data into
                            {default: 1}
            kwargs -- additional elements to pass

        returns:
        """
        pass

    def expectation_of_distro(self, Data, distribution, **kwargs) -> tuple:
        """will take an input and fit a distribution and take the expecation

        Arguments:
                Data {np.array} -- data source to fit distirbution expectation_of_distro
                distribution {str} -- the name of the distribtion to fit to. This must
                                    be a distribution name contained in scipy

        Returns:
                tuple of two numpy arrays:
                    0: The expected frequency of the distribution from the Data
                    1: the CDF of the fitted distribution from the data
                    2: A tuple of params from the fitting process

        Details:
            For the Data set given, a distribution is "trained" on it, in that it will find
            build a distribution with the same parameters as the Data set has.
            It will then take the expectation of that distribution - IE what should the values
            be per cutoff bucket IF the data was truly of that distribution
        """
        pass

    def find_distribution(
        self,
        Data: np.array,
        distributions: list,
        Buckets: int = 11,
        lower_bound: float = 0.001,
        upper_bound: float = 0.999,
    ):
        """Will calculate the chi square statistic for a list of distribution"""
        # Prepping Data
        sdz_data = self.standardize(Data)
        sdz_data = self.clip_edges(sdz_data, lower_bound, upper_bound)

        # Getting frequencys
        obs_frequency, bins = self.data_frequency(sdz_data, Buckets)

        if U.typeof(distributions) == "Str":
            distributions = [distributions]

        assert U.typeof(distributions) == "List", "distribution must be a list"
        # Fitting distro and getting chi square value
        Chi_square = []
        P_value = []
        RMSE = []
        for distro in distributions:
            exp_frequency, cdf_fitted, _params = self.expectation_of_distro(
                sdz_data, bins, distro
            )
            res = stats.chisquare(obs_frequency, exp_frequency)
            Chi_square.append(res.statistic)
            P_value.append(res.pvalue)
            RMSE.append(np.sum((obs_frequency - exp_frequency) ** 2))

        # Sorting chi_square
        Disto_results = (
            pd.DataFrame(
                {
                    "Distribution": distributions,
                    "RMSE": RMSE,
                    "Chi_Square": np.round(Chi_square, 0),
                    "P_Value": np.round(P_value, 4),
                }
            )
            .sort_values(by="Chi_Square", ascending=True)
            .reset_index()
        )
        return Disto_results[["Distribution", "RMSE", "Chi_Square", "P_Value"]]


def int_corr():
    """Find correlation for integers"""
    pass


def cramers_v(x, y):
    """calculating crammer v

    https://www.kaggle.com/akshay22071995/alone-in-the-woods-using-theil-s-u-for-survival
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = np.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def compute_confusion_matrix(x: np.array, y: np.array) -> np.array:
    """Computes a confusion matrix using numpy for two np.arrays
    true and pred.

    Results are identical (and similar in computation time) to:
    "from sklearn.metrics import confusion_matrix"

    However, this function avoids the dependency on sklearn."""

    K = len(np.unique(x))  # Number of classes
    result = np.zeros((K, K))

    for i in range(len(x)):
        result[x[i]][y[i]] += 1

    return result
