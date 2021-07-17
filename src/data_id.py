import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
from typing import Union
import util as U


def replace_nonnumeric(
    Num: Union[float, int], replace_value=0, check_zero: bool = False
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
    assert U.typeof(pct_lower) == "Float", "pct_lower needs to be of type float"
    assert U.typeof(pct_upper) == "Float", "pct_upper needs to be of type float"
    assert (
        pct_lower >= 0.0 and pct_lower < 1.00
    ), "pct_lower needs to be at least 0.0 and less than 1.0"
    assert (
        pct_upper <= 1.0 and pct_upper > 0
    ), "pct_upper needs to be at most 1.0 and greater than 0.0"

    if pct_upper < pct_lower:
        pct_lower, pct_upper = pct_upper, pct_lower

    data_length = len(Data)
    Data_Clip = Data[int(pct_lower * data_length) : int(data_length * pct_upper)]
    return Data_Clip


def standardize(Data: Union[pd.Series, np.array]) -> np.array:
    """Will subtract the mean and divided by the std

    Arguments:
        f {pd.Series, np.array} -- input numerical feature to standardize

    returns:
        np.array
    """

    assert U.typeof(Data, False) in [
        "Array",
        "Series",
    ], "Data for standarize isn't a numpy array or a pandas Series"

    assert U.dtype_simple(Data) in [
        "float",
        "int",
    ], "Column provided to standardize isn't numerical"

    # it's more efficent with pandas to specifically invoke numpy (IE dataframe.values)
    if U.typeof(Data) == "Series":
        v_sd = Data.values.std()
        v_sd = replace_nonnumeric(v_sd, 1, True)

        v_mean = Data.values.mean()
        v_mean = replace_nonnumeric(v_mean, 0)

        sdz_data = (Data.values - v_mean) / v_sd
    else:
        v_sd = Data.std()
        v_sd = replace_nonnumeric(v_sd, 1, True)

        v_mean = Data.mean()
        v_mean = replace_nonnumeric(v_mean, 0)
        sdz_data = (Data - v_mean) / v_sd

    return sdz_data


def data_frequency(Data: np.array, Buckets: int = 10) -> tuple:
    """will order the data into bins and provide a frequency of the data

    Arguments:
        Data {np.array} -- an input numpy array

    Keyword Arguments:
        Buckets {int} -- Number of buckets to sort the data into
                         {default: 1}

    returns:
        tuple of two numpy arrays
            0: the frequency count of each bins
            1: the right limit of the bins

    Details:
        It will go through and will sort the data into approperate number of bins,
        similar to what a histogram does, it will then do a culmative frequency on
        the bins for an added bonus.
    """
    assert U.typeof(Buckets) == "Int", "Buckets needs to be an int"
    assert U.typeof(Data) in "Array", "Data need to be an array"

    percentile_bins = np.percentile(Data, np.linspace(0, 100, Buckets))
    # Truth be told, it's actually faster to us np.histogram than write your own
    observed_frequency, bins = np.histogram(Data, bins=percentile_bins)
    return (observed_frequency, bins)


def expectation_of_distro(Data: np.array, buckets: np.array, distribution) -> tuple:
    """will take an input and fit a distribution and take the expecation

    Arguments:
            Data {np.array} -- data source to fit distirbution expectation_of_distro
            cutoffs {np.array} -- the percentile cutoff points to measure against
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

    assert U.typeof(distribution) == "Str", "Distribution must be a string value"
    assert U.typeof(buckets) in ["Array", "List"], "buckets need to be a list"
    assert len(buckets) > 0, "buckets needs to be a list with at least one element"

    # below is equivilant to scipy.stats.distribution
    dist = getattr(stats, distribution)

    # fitting data
    param = dist.fit(Data)

    # Get expected counts in percentile bins
    # cdf of fitted sistrinution across bins
    cdf_fitted = dist.cdf(buckets, *param)
    expected_frequency = []
    for i in range(len(buckets) - 1):
        expected_cdf_area = cdf_fitted[i + 1] - cdf_fitted[i]
        expected_frequency.append(expected_cdf_area)

    expected_frequency = np.array(expected_frequency) * len(Data)
    return expected_frequency, cdf_fitted, (param)


def find_distribution(
    Data: np.array,
    distributions: list,
    Buckets: int = 11,
    lower_bound: float = 0.001,
    upper_bound: float = 0.999,
):
    """Will calculate the chi square statistic for a list of distribution"""
    # Prepping Data
    sdz_data = standardize(Data)
    sdz_data = clip_edges(sdz_data, lower_bound, upper_bound)

    # Getting frequencys
    obs_frequency, bins = data_frequency(sdz_data, Buckets)

    if U.typeof(distributions) == "Str":
        distributions = [distributions]

    assert U.typeof(distributions) == "List", "distribution must be a list"
    # Fitting distro and getting chi square value
    Chi_square = []
    P_value = []
    RMSE = []
    for distro in distributions:
        exp_frequency, cdf_fitted, _params = expectation_of_distro(
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


def boxcox_transform(Data: np.ndarray, Lambda: float = None) -> tuple:
    """Converts a Numpy 1-D array to a normal distribution using a box-cox transform

    Arguments:
        Data {numpy 1d array} -- the data to convert

    Keyword Arguments:
        Lambda {float} -- the lambda value to transform
                          {Default: None}

    Returns:
        returns a tuple of :
            0: Numpy 1-d array of the converted data
            1: lambda value used

    Details:
        This will convert a given 1-D numpy array into a normal distirbutions. If lambda
        is None, it will go through and try to find the optimal lambda value (note, can be slow
        on large dataset ( ~20sec for 10M records)

        However, if you "kinda of" already know the distribution, you can provided the lambda
        and that will speed it up by 10x to 100x. Some common lambda values for transformations are:
            * -NL will run a 1/x^(n) transform
            * -1: will run 1/x transform
            * -0.5: will run 1/sqrt(x) transfrom (use with a 1/x^2 data set)
            * 0: will run a log(x) transfrom (use with a e^x dataset)
            * 0.5: will run a sqrt(x) trasnfrom (use with a x**2)
            * 1: x (IE, no transformation)
            * 2: will run a x*2 transform (use with a log_2(x) dataset)
            * N+: will wun a x*n transfrom
    """

    assert U.typeof(Data) == "Array", "BoxCox transform needs Data to be an array"

    if Lambda is not None:
        assert U.typeof(Lambda) in [
            "Float",
            "Int",
        ], "BoxCox transfrom - Lambda needs to be a float or int"

    Res, lmbda = stats.boxcox(Data, lmbda=Lambda)
    return (Res, lmbda)


def revert_boxcox_transform(Data: np.ndarray, Lambda: float = 0) -> np.ndarray:
    """Reverse a boxcox transformations

    Arguments:
        Data {np.ndarray} -- box-cox transformed 1-d numpy Array

    Keyword Arguments:
        Lambda {float} -- Lambda value to use to revert
                          (default: {0})

    Returns:
        np.ndarray -- reverted boxcox transform data

    Details:
        This will reverse a former box cox transfrom. However, you will need to provide a lambda
        in order for this to work. If no lambda is provided, it will just raise it to the power
        of e
    """
    assert (
        U.typeof(Data) == "Array"
    ), "revert_boxcox_transform needs Data to be an array"
    assert U.typeof(Lambda) in [
        "Float",
        "Int",
    ], "revert_boxcox _transfrom - Lambda needs to be a float or int"

    Res = inv_boxcox(Data, Lambda)
    return Res
