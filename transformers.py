from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
import numpy as np

# Standard scaler data preparation class


class StandardScalerTransform(BaseEstimator, TransformerMixin):
    def __init__(self, feature_is_included_list=[]):
        self.standard_scalers = {}

    def fit(self, X, y=None):
        for index in range(len(self.feature_is_included_list)):
            if self.feature_is_included_list[index] > 0.5:
                self.standard_scalers[index] = StandardScaler()
                self.standard_scalers[index].fit(X[:, index:index + 1])
        return self

    def transform(self, X, y=None):
        for index in range(len(self.feature_is_included_list)):
            if self.feature_is_included_list[index] > 0.5:
                X[:, index:index + 1] = self.standard_scalers[index].transform(
                    X[:, index:index + 1])
        return np.c_[X]


# Min-max scaler data preparation class


class MinMaxScalerTransform(BaseEstimator, TransformerMixin):
    def __init__(self, feature_is_included_list=[]):
        self.min_max_scalers = {}

    def fit(self, X, y=None):
        for index in range(len(self.feature_is_included_list)):
            if self.feature_is_included_list[index] > 0.5:
                self.min_max_scalers[index] = MinMaxScaler(feature_range=(0, 10))
                self.min_max_scalers[index].fit(X[:, index:index + 1])
        return self

    def transform(self, X, y=None):
        for index in range(len(self.feature_is_included_list)):
            if self.feature_is_included_list[index] > 0.5:
                X[:, index:index + 1] = self.min_max_scalers[index].transform(
                    X[:, index:index + 1])
        return np.c_[X]



# KBinsDiscretizer data preparation class


class KBinsDiscretizerTransform(BaseEstimator, TransformerMixin):
    def __init__(self, feature_is_included_list={}):
        self.k_bins_discretizers = {}

    def fit(self, X, y=None):
        for index in range(len(self.feature_is_included_list)):
            if self.feature_is_included_list[index] == True:
                self.k_bins_discretizers[index] = KBinsDiscretizer(encode='ordinal', strategy='quantile')
                self.k_bins_discretizers[index].fit(X[:, index:index + 1])
        return self

    def transform(self, X, y=None):
        for index in range(len(self.feature_is_included_list)):
            if self.feature_is_included_list[index] == True:
                try:
                    X[:, index:index +
                      1] = self.k_bins_discretizers[index].transform(X[:, index:index + 1])
                except:
                    pass
        return np.c_[X]


# KBinsDiscretizer2 data preparation class


class KBinsDiscretizerTransform2(BaseEstimator, TransformerMixin):
    def __init__(self, feature_is_included_list={}):
        self.k_bins_discretizers = {}

    def fit(self, X, y=None):
        for index in range(len(self.feature_is_included_list)):
            if self.feature_is_included_list[index] == True:
                self.k_bins_discretizers[index] = KBinsDiscretizer(encode='ordinal', strategy='kmeans')
                self.k_bins_discretizers[index].fit(X[:, index:index + 1])
        return self

    def transform(self, X, y=None):
        for index in range(len(self.feature_is_included_list)):
            if self.feature_is_included_list[index] == True:
                try:
                    X[:, index:index +
                      1] = self.k_bins_discretizers[index].transform(X[:, index:index + 1])
                except:
                    pass
        return np.c_[X]


# KBinsDiscretizer3 data preparation class


class KBinsDiscretizerTransform3(BaseEstimator, TransformerMixin):
    def __init__(self, feature_is_included_list={}):
        self.k_bins_discretizers = {}

    def fit(self, X, y=None):
        for index in range(len(self.feature_is_included_list)):
            if self.feature_is_included_list[index] == True:
                self.k_bins_discretizers[index] = KBinsDiscretizer(encode='ordinal', strategy='uniform')
                self.k_bins_discretizers[index].fit(X[:, index:index + 1])
        return self

    def transform(self, X, y=None):
        for index in range(len(self.feature_is_included_list)):
            if self.feature_is_included_list[index] == True:
                try:
                    X[:, index:index +
                      1] = self.k_bins_discretizers[index].transform(X[:, index:index + 1])
                except:
                    pass
        return np.c_[X]


# Binarizer data preparation class


class BinarizerTransform(BaseEstimator, TransformerMixin):
    def __init__(self, feature_is_included_list={}):
        self.thresholds = {}

    def fit(self, X, y=None):
        for index in range(len(self.feature_is_included_list)):
            if self.feature_is_included_list[index] == True:
                self.thresholds[index] = np.quantile(X[:, index:index + 1],
                                                 0.50)
        return self

    def transform(self, X, y=None):
        for index in range(len(self.feature_is_included_list)):
            if self.feature_is_included_list[index] == True:
                X[:, index:index +
                  1] = X[:, index:index + 1] > self.thresholds[index]
        return np.c_[X]
