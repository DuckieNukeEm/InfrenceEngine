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
    if Num is None or np.isnan(Num) or np.isinf(Num):
        return replace_value
    elif check_zero is True and Num == 0:
        return replace_value
    else:
        return Num


def standardize(Data: Union[pd.Series, np.array]) -> np.array:
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


def clip_edges(Data: np.Array, pct_lower: float = 0.001, pct_upper: float = 0.99):
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


def data_frequency(Data: np.Array, Buckets: int = 10) -> tuple:
    """will order the data into bins and provide a frequency of the data

    Arguments:
        Data {np.array} -- an input numpy array

    Keyword Arguments:
        Buckets {int} -- Number of buckets to sort the data into
                         {default: 1}

    returns:
        tuple of numpy arrays
            0: the frequency count of each bins
            1: the percentile of eahc bin
            2: the right limit of the bins

    Details:
        It will go through and will sort the data into approperate number of bins,
        similar to what a histogram does, it will then do a culmative frequency on
        the bins for an added bonus.
    """
    
    percentile_bins = np.percentile(Data, np.linspace(0,100, Buckets))
    # Truth be told, it's actually faster to us np.histogram than write your own 
    observed_frequency, bins = (np.histogram(y_std, bins=percentile_bins))
    cum_observed_frequency = np.cumsum(observed_frequency)
    return(observed_frequency, percentile_bins, bins)
    

def expectation_of_distro(Data: np.array, cutoffs: np.array, distribution) -> tuple:
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
    
    Details:
        For the Data set given, a distribution is "trained" on it, in that it will find
        build a distribution with the same parameters as the Data set has. 
        
        It will then take the expectation of that distribution - IE what should the values
        be per cutoff bucket IF the data was truly of that distribution
    """
    
    # below is equivilant to scipy.stats.distribution
    dist = getattr(scipy.stats, distribution)
    
    # fitting data 
    param = dist.fit(Data)

    # Get expected counts in percentile bins
    # cdf of fitted sistrinution across bins
    cdf_fitted = dist.cdf(cutoffs, *param)
    expected_frequency = []
    for i in range(len(cutoffs)-1):
        expected_cdf_area = cdf_fitted[i+1] - cdf_fitted[i]
        expected_frequency.append(expected_cdf_area)
    
    expected_frequency = np.array(expected_frequency) * len(Data)
    return expected_frequency, cdf_fitter


def chi_square(observed_frequency: np.array, expected_frequency: np.array, cum_observed_frequency: np.array = np.array()) -> float:
    """will calculate the chi-square between the Data_origin and Data_expected
    
    Arguments:
        observed_frequency {np.array} -- the observed frequence from data 
        expected_frequency {np.array} -- the expected frqeuency 

    Keyword Arguments:
        cum_observed_frequency {np.array} -- the cumultaive observed frequency. If an emtpy array
                                             it will be calculated from the observed_frequency
                                             {default: []}
    Returns
        float -- the chi-square statistic between the observed and expected frequency 
        
    Details 
        This caculates the chi square frequency between the observed and expected frequency
        by the following formula:
            insert chi_square formula 
    
    """
    cum_exp_frequency = np.cumsum(expected_frequency)
    
    if len(cum_obs_frequency) == 0:
        cum_obs_frequency = np.cumsum(observed_frequency)
        
    chi_stat = sum(((cum_exp_frequency - cum_obs_frequency) ** 2) / cum_obs_frequency)
    return(chi_stat)



def chi_square_of_distro(Data: np.array, distributions: list(str), Buckets: int = 11):
    """Will calculate the chi square statistic for a list of distribution"""
    # Prepping Data
    sdz_data = standardize(Data) 
    sdz_data = clip(sdz, lower_bound, upper_bound)
    
    # Getting frequencys
    obs_frequency, percentile_bins, bins = data_frequency(sdz_data, Buckets)
    cum_obs_frequency = np.cumsum(obs_frequency)
    
    #Fitting distro and getting chi square value
    Chi_square = []
    for distro in distributions:
        exp_frequency, cdf_fitted = expectation_of_distro(sdz_data, percentile_bins, distro )
        sum_square_error = chi_square_of_distro(obs_frequency, exp_frequency, cum_obs_frequency)
        Chi_square.append(sum_square_error)
        
    # Sorting chi_square
    Disto_results = pd.DataFrame([distro, Chi_square], columns = ['Distribution','Error']).sort(['Error'], True)
    return(Disto_results)


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
