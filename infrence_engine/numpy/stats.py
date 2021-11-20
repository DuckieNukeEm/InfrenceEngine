import numpy as np
from numpy.core.records import array
import util as U
from scipy import stats
from typing import Union, List
from infrence_engine.numpy.augmentation import Augmentation
from infrence_engine.numpy.data_types import data_types as np_data_types
from data_types import data_types
from infrence_engine.error import raise_type_error


class Stats:
    def __init__(self):
        self.dt = data_types()
        self.n_dt = np_data_types()

    def sample(
        self, Data: np.array, size: Union[int, float], with_replacement: bool = True
    ) -> np.array:
        """Generates a np.array of index that are the row index of sampled records

        Arguments:
            Data {np.array} -- data to draw the samples from
            size {Union[int, float]} -- either the number of records to pull (if int)
                                        or the % to sample (if float)

        Keyword Arguments:
            with_replacement {bool} -- sample with replacement (default: {True})

        Returns:
            np.array -- [description]
        """
        if self.dt.is_numeric(size) is False:
            raise_type_error(size, "size", self.dt.numeric)

        if self.n_dt.is_array(Data) is False:
            raise_type_error(Data, "Data", ["ndarry"])

        if with_replacement is not True:
            with_replacement = False

        if self.dt.is_float(size) is True:
            size = int(round(Data.size[0] * size, 0))

        Sample = np.random.choice(Data.size[0], size=size, replace=with_replacement)

        return Sample

    def crammers_v(self, x: np.array, y: np.array()) -> float:
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

    def corr(self, X: np.ndarray) -> array:
        """Find correlation matrix amongst variables of numeric type

        Args:
            X (array): numpy numeric array

        Raises:
            TypeError: if X isn't a numpy numeric type

        Returns:
            array: Correlation matrix of dtype.64.
        """

        if self.n_dt.is_numeric(X) is False:
            raise_type_error(X, "X", self.n_dt.numeric)

        Cor_Mat = np.cov(X, rowvar=False)
        return Cor_Mat


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
