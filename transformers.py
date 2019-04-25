from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
import numpy as np

# The modeling class


class MyCustomModel:
    '''
    - Includes algorithms as ensembles

    '''

    def __init__(self, uses_xgboost=True): # in the future add many booleans not just this one
        self.model = None

    def fit(self, train_X, train_y):
        self.model = get_xgboost_model(train_X, train_y, quickly=True)
        self.model.fit(train_X, train_y)

    def predict(self, test_X):
        return self.model.predict(test_X)



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
