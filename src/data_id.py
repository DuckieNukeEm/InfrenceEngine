import pandas as pd
import numpy as np
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
    if Num is None or np.nan(Num) or np.inf(Num):
        return replace_value
    elif check_zero is True and Num == 0:
        return replace_value
    else:
        return Num


def one_col_standardize(Data: Union[pd.Series, np.array]) -> np.array:
    """Will subtract the mean and divided by the std

    Arguments:
        f {pd.Series, np.array} -- input numerical feature to standardize

    returns:
        np.array
    """

    assert U.typeof(Data) in [
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


def clip(Data: np.Array, pct_lower: float = 0.001, pct_upper: float = 0.99):
    """WIll clipp the upper % and lower % off of the index

    Arguments:
        Data {np.Array} -- the input array to clip

    Keyword Arguments:
        pct_lower {float} -- the lower bound to clip (IE anything below lower_bound % is removed)
        pct_upper {float} -- the upper bound to clip (anything above the value is removed)

    Returns:
        np.array


    """
    return ()
    data_length = len(Data)
    Data_Clip = Data[int(pct_lower * data_length) : int(data_length * pct_upper)]
    return Data_Clip


def standardize():
    """will standardize all columns selected"""
    pass


def bucketize(Data: np.Array, Bins: int = 10) -> np.array:
    """will sort and bucket columns"""
    pass


def fit_distro():
    """will take an input and fit a distribution to it"""
    pass


def chi_square_of_distro():
    """Will calculate the chi square statistic for a given distribution"""
    pass


def find_distribution():
    """will itteratie through numerical data and find the distribution style for each one"""
    pass
    """
    https://github.com/samread81/Distribution-Fitting-Used_Car_Dataset/blob/master/Workbook.ipynb
    def fit_distribution(column,pct,pct_lower):
    py  # Set up list of candidate distributions to use
        # See https://docs.scipy.org/doc/scipy/reference/stats.html for more
        y_std,size,y_org = standarise(column,pct,pct_lower)
        dist_names = ['weibull_min','norm','weibull_max','beta',
                    'invgauss','uniform','gamma','expon', 'lognorm','pearson3','triang']

        chi_square_statistics = []
        # 11 bins
        percentile_bins = np.linspace(0,100,11)
        percentile_cutoffs = np.percentile(y_std, percentile_bins)
        observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
        cum_observed_frequency = np.cumsum(observed_frequency)

        # Loop through candidate distributions

        for distribution in dist_names:
            # Set up distribution and get fitted distribution parameters
            dist = getattr(scipy.stats, distribution)
            param = dist.fit(y_std)
            print("{}\n{}\n".format(dist, param))


            # Get expected counts in percentile bins
            # cdf of fitted sistrinution across bins
            cdf_fitted = dist.cdf(percentile_cutoffs, *param)
            expected_frequency = []
            for bin in range(len(percentile_bins)-1):
                expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
                expected_frequency.append(expected_cdf_area)

            # Chi-square Statistics
            expected_frequency = np.array(expected_frequency) * size
            cum_expected_frequency = np.cumsum(expected_frequency)
            ss = round(sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency),0)
            chi_square_statistics.append(ss)


        #Sort by minimum ch-square statistics
        results = pd.DataFrame()
        results['Distribution'] = dist_names
        results['chi_square'] = chi_square_statistics
        results.sort_values(['chi_square'], inplace=True)


        print ('\nDistributions listed by Betterment of fit:')
        print ('............................................')
        print (results)
    """
