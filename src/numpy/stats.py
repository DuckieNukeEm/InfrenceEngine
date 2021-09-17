import numpy as np
import utils as U
from scipy import stats
from typing import Union
from .stats import Stats as base_stats
from .numpy.augmentation import Augmentation
from .numpy.data_types import data_types as np_data_types
from data_types import data_types


class Stats(base_stats):
    def __init__(self):
        self.aug = Augmentation()
        self.data_types = data_types()
        self.np_types = np_data_types()

    def sample(
        self, Data: np.array, size: Union[int, float], with_replacement: bool = True
    ) -> np.array:
        """Generates a np.array of index that are the samples sampled from the Data

        Arguments:
            Data {np.array} -- data to draw the samples from
            size {Union[int, float]} -- either the number of records to pull (if int)
                                        or the % to sample (if float)

        Keyword Arguments:
            with_replacement {bool} -- sample with replacement (default: {True})

        Returns:
            np.array -- [description]
        """
        if self.data_types.is_numeric(size) is False:
            raise TypeError("size needs to be an int or a float, not a %s" % type(size))

        if self.np_types.is_array(Data) is False:
            raise TypeError("Data needs to be an numpy ndarry")

        if with_replacement is not True:
            with_replacement = False

        if self.data_types.is_float(size) is True:
            size = int(round(Data.size[0] * size, 0))
        Sample = np.random.choice(Data.size[0], size=size, replace=with_replacement)
        return Sample

    def data_frequency(self, Data: np.array, Buckets: int = 10) -> tuple:
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
        if self.data_types.is_int(Buckets) is False:
            raise TypeError("Buckets shouldbe an int, %s was provided" % type(Buckets))

        if self.np_type.is_array(Data) is False:
            raise TypeError("Data needs to be a numpy array")

        percentile_bins = np.percentile(Data, np.linspace(0, 100, Buckets))
        # Truth be told, it's actually faster to us np.histogram than write your own
        observed_frequency, bins = np.histogram(Data, bins=percentile_bins)
        return (observed_frequency, bins)

    def expectation_of_distro(
        self, Data: np.array, buckets: np.array, distribution
    ) -> tuple:
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
        self,
        Data: np.array,
        distributions: list,
        Buckets: int = 11,
        lower_bound: float = 0.001,
        upper_bound: float = 0.999,
    ):
        """Will calculate the chi square statistic for a list of distribution"""
        # Prepping Data
        sdz_data = self.augmentation.standardize(Data)
        sdz_data = self.augmentation.clip_edges(sdz_data, lower_bound, upper_bound)

        # Getting frequencys
        obs_frequency, bins = self.data_frequency(sdz_data, Buckets)

        if self.data_Types.is_str(distributions) is True:
            distributions = [distributions]

        assert U.typeof(distributions) == "List", "distribution must be a list"
        # Fitting distro and getting chi square value
        # Chi_square = []
        # P_value = []
        # RMSE = []
        Array = []
        for distro in distributions:
            exp_frequency, cdf_fitted, _params = self.expectation_of_distro(
                sdz_data, bins, distro
            )
            res = stats.chisquare(obs_frequency, exp_frequency)
            # Chi_square.append(res.statistic)
            # P_value.append(res.pvalue)
            # RMSE.append(np.sum((obs_frequency - exp_frequency) ** 2))

            Chi_square = res.statistic
            P_value = res.pvalue
            RMSE = np.sum((obs_frequency - exp_frequency) ** 2)
            Array.append((distro, RMSE, Chi_square, P_value))

        NdArray = np.array(Array, dtype="|S4, f32, f32, f32")

        # Sorting chi_square
        NdArray = NdArray[NdArray[:, 2].argsort()]

        return NdArray
        # Sorting chi_square
        """   Disto_results = (
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
        return Disto_results[["Distribution", "RMSE", "Chi_Square", "P_Value"]] """

    def crammers_v(self, x: np.array, y: np.array()):
        """calculating crammer v of two categorical arrays

        https://www.kaggle.com/akshay22071995/alone-in-the-woods-using-theil-s-u-for-survival
        """
        confusion_matrix = np.crosstab(x, y)
        chi2 = np.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def int_corr():
    """Find correlation for integers"""
    pass


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
