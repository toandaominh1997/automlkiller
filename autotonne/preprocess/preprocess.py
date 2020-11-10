import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
import sklearn
import sklearn.impute
import sklearn.feature_selection
import pyod
import pyod.models.knn
import pyod.models.iforest
import pyod.models.pca
import category_encoders as ce

from autotonne.preprocess.preprocee_factory import PreprocessFactory
from autotonne.utils.logger import LOGGER


@PreprocessFactory.register('preprocess-cleancolumnname')
class CleanColumnName(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        LOGGER.info('FIT CleanColumnName')
        return self
    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM CleanColumnName')
        X = X.copy()
        if y is not None:
            y = y.copy()
        X.columns = X.columns.str.replace(r"[\,\}\{\]\[\:\"\']", "")

        return X, y


@PreprocessFactory.register('preprocess-datatype')
class DataTypes(BaseEstimator, TransformerMixin):
    def __init__(self,
                 categorical_columns = [],
                 numeric_columns = [],
                 time_columns = [],
                 verbose = True):
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.time_columns = time_columns
        self.verbose = verbose
    @staticmethod
    def str_if_not_null(x):
        if pd.isnull(x) or (x is None) or pd.isna(x) or (x is not x):
            return x
        return str(x)
    def fit(self, X, y = None):
        LOGGER.info('FIT DATATYPE')
        X = X.copy()
        dataset = X
        if y is not None:
            y = y.copy()

        X.columns = [str(col) for col in X.columns]
        X.replace([np.inf, -np.inf], np.NaN, inplace=True)
        # remove columns with duplicate name
        # X = X.loc[:, ~X.columns.duplicated()]
        # remove NAs
        # X.dropna(axis = 0, how = 'all', inplace = True)
        # X.dropna(axis = 1, how = 'all', inplace = True)


        for col in X.select_dtypes(include=['object']).columns:
            try:
                X[col] = X[col].astype("int64")
            except:
                continue
        for col in X.select_dtypes(include=['object']).columns:
            try:
                X[col] = pd.to_datatime(X[col], infer_datetime_format = True, utc = False, errors = 'raise')
            except:
                continue
        # if data type is bool or pandas Categorical, convert to categorical
        for col in X.select_dtypes(include=["bool", "category"]).columns:
            X[col] = X[col].astype("object")

        # with csv, if we have any null in a column that was int, pandas will read it as float
        # so first we need to convert any such floats that hae NaN and unique values are lower than 20
        for col in X.select_dtypes(include = ["float64"]).columns:
            X[col] = X[col].astype("float32")

            nan_count = sum(X[col].isnull())
            count_float = np.nansum([False if r.is_integer() else True for r in X[col]])
            count_float = (count_float - nan_count)
            if (count_float == 0) and (X[col].nunique() <= 20) and (nan_count > 0):
                X[col] = X[col].astype("object")

        # if column is int and unique counts are more than two
        for col in X.select_dtypes(include = ["int64"]).columns:
            if X[col].nunique() <= 5:
                X[col] = X[col].apply(self.str_if_not_null)
            else:
                X[col] = dataset[col].astype("float32")
        for col in X.select_dtypes(include=['float32']).columns:
            if X[col].nunique() == 2:
                X[col] = X[col].apply(self.str_if_not_null)

        for col in self.numeric_columns:
            try:
                X[col] = dataset[col].astype("float32")
            except:
                X[col] = dataset[col].astype(self.str_if_not_null)

        for col in self.categorical_columns:
            try:
                X[col] = dataset[col].apply(self.str_if_not_null)
            except:
                X[col] = dataset[col].astype(self.str_if_not_null)

        for col in self.time_columns:
            try:
                X[col] = pd.to_datetime(X[col], infer_datetime_format=True, utc = False, errors = "raise")
            except:
                X[col] = pd.to_datetime(dataset[col], infer_datetime_format=True, utc = False, errors = "raise")
        for col in X.select_dtypes(include=["datetime64"]).columns:
            X[col] = X[col].astype("datetime64[ns]")
        self.learned_dtypes = X.dtypes

        X = X.replace([np.inf, -np.inf], np.NaN).astype(self.learned_dtypes)

        self.final_columns = X.columns.tolist()
        return self

    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM DATATYPE')
        X = X.copy()
        if y is not None:
            y = y.copy()

        X.columns = [str(col) for col in X.columns]
        X = X.loc[:, self.final_columns]
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.astype(self.learned_dtypes)


        return X, y
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)



@PreprocessFactory.register('preprocess-simpleimputer')
class SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_strategy,
                 categorical_strategy,
                 fill_value_numerical = 0,
                 fill_value_categorical = 'not_available'):
        self.numeric_imputer = sklearn.impute.SimpleImputer(strategy=numeric_strategy, fill_value=fill_value_numerical)
        self.categorical_imputer = sklearn.impute.SimpleImputer(strategy=categorical_strategy, fill_value=fill_value_categorical)

    def fit(self, X, y = None):
        LOGGER.info('FIT SIMPLE IMPUTER')
        X = X.copy()
        if y is not None:
            y = y.copy()
        self.numeric_columns = [str(col) for col in X.select_dtypes(include=[np.number]).columns.tolist()]
        self.categorical_columns = [str(col) for col in X.select_dtypes(include=['object', 'category']).columns.tolist()]
        self.time_columns = [str(col) for col in X.select_dtypes(include=['datetime64[ns]']).columns.tolist()]
        if len(self.numeric_columns) > 0:
            self.numeric_imputer.fit(X.loc[:, self.numeric_columns])

        if len(self.categorical_columns) > 0:
            self.categorical_imputer.fit(X[self.categorical_columns])

        if len(self.time_columns) > 0:
            self.most_frequent_time = []
            for col in self.time_columns:
                self.most_frequent_time.append(X[col].mode()[0])

        return self
    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM SIMPLE IMPUTER')
        X = X.copy()
        if y is not None:
            y = y.copy()
        if len(self.numeric_columns) > 0:
            X.loc[:, self.numeric_columns] = self.numeric_imputer.transform(X.loc[:, self.numeric_columns])
        if len(self.categorical_columns) > 0:
            X.loc[:, self.categorical_columns] = self.categorical_imputer.transform(X.loc[:, self.categorical_columns])
        if len(self.time_columns) > 0:
            for idx, col in enumerate(self.time_columns):
                X[col].fillna(self.most_frequent_time[idx])
        return X, y
    def fit_transform(self, X, y = None):
        X = X.copy()
        self.fit(X, y)
        return self.transform(X, y)


# Zero and Near Zero Variance
@PreprocessFactory.register('preprocess-zeronearzerovariance')
class ZeroNearZeroVariance(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_first = 0.1, threshold_second = 20):
        self.threshold_first = threshold_first
        self.threshold_second = threshold_second
        self.to_drop = []
    def fit(self, X, y = None):
        LOGGER.info('FIT ZERO AND NEARZERO Variance')
        X = X.copy()
        if y is not None:
            y = y.copy()

        for col in X.columns:
            u = pd.DataFrame(X[col].value_counts()).sort_values(by=col, ascending = False, inplace = False)
            first = len(u) / len(X)
            if len(u[col]) == 1:
                second = 100
            else:
                second = u.iloc[0, 0] / u.iloc[1, 0]

            if first <= self.threshold_first and second >= self.threshold_second:
                self.to_drop.append(col)
            if second == 100:
                self.to_drop.append(col)
    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM ZERO and NEARZERO Variance')
        X = X.copy()
        if y is not None:
            y = y.copy()
        X = X.drop(columns = self.to_drop)
        return X, y
    def fit_transform(self, X, y = None):
        X = X.copy()
        if y is not None:
            y = y.copy()
        self.fit(X, y)
        return self.transform(X, y)


@PreprocessFactory.register('preprocess-categoryencoder')
class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols = [], method='onehotencoder'):
        self.cols = cols
        self.method = method
    def fit(self, X, y = None):
        LOGGER.info('FIT Category-Encoder')
        X = X.copy()
        if y is not None:
            y = y.copy()
        if len(self.cols) == 0:
            self.cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        self.encoder = self.choose_method(cols = self.cols, method = self.method)
        self.encoder.fit(X, y)
    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM Category-Encoder')
        X = X.copy()
        if y is not None:
            y = y.copy()
        X = self.encoder.transform(X)
        return X, y
    def fit_transform(self, X, y = None):
        X = X.copy()
        self.fit(X, y)
        return self.transform(X, y)
    def choose_method(self, cols, method):
        if method == 'backwarddifferenceencoder':
            encoder = ce.BackwardDifferenceEncoder(cols=cols, return_df=True)
        elif method == 'baseencoder':
            encoder = ce.BaseNEncoder(cols=cols, return_df=True)
        elif method == 'binaryencoder':
            encoder = ce.BinaryEncoder(cols=cols, return_df=True)
        elif method == 'catboostencoder':
            encoder = ce.CatBoostEncoder(cols=cols, return_df=True)
        elif method == 'countencoder':
            encoder = ce.CountEncoder(cols=cols, return_df=True)
        elif method == 'glmmeencoder':
            encoder = ce.GLMMEncoder(cols=cols, return_df=True)
        elif method == 'hashingencoder':
            encoder = ce.HashingEncoder(cols=cols, return_df=True)
        elif method == 'helmerencoder':
            encoder = ce.HelmertEncoder(cols=cols, return_df=True)
        elif method == 'jamessteinencoder':
            encoder = ce.JamesSteinEncoder(cols=cols, return_df=True)
        elif method == 'leaveoneoutencoder':
            encoder = ce.LeaveOneOutEncoder(cols=cols, return_df=True)
        elif method == 'mestimateencoder':
            encoder = ce.MEstimateEncoder(cols=cols, return_df=True)
        elif method == 'onehotencoder':
            encoder = ce.OneHotEncoder(cols=cols, use_cat_names=True, return_df=True)
        elif method == 'ordinalencoder':
            encoder = ce.OrdinalEncoder(cols=cols, return_df=True)
        elif method == 'sumencoder':
            encoder = ce.SumEncoder(cols=cols, return_df=True)
        elif method == 'polynomialencoder':
            encoder = ce.PolynomialEncoder(cols=cols, return_df=True)
        elif method == 'targetencoder':
            encoder = ce.TargetEncoder(cols=cols, return_df=True)
        elif method == 'woeeencoder':
            encoder = ce.WOEEncoder(cols=cols, return_df=True)
        return encoder

@PreprocessFactory.register('preprocess-groupsimilarfeature')
class GroupSimilarFeature(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            group_name =[],
            list_of_group_feature = []):
        self.list_of_group_feature = list_of_group_feature
        self.group_name = group_name
    def fit(self, X, y = None):
        LOGGER.info('FIT GroupSimilarFeature')
        X = X.copy()
        if y is not None:
            y = y.copy()
        return self
    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM GroupSimilarFeature')
        X = X.copy()
        if y is not None:
            y = y.copy()
        if len(self.list_of_group_feature) > 0:
            for f, g in zip(self.list_of_group_feature, self.group_name):
                X[g+"_min"] = X[f].apply(np.min, 1)
                X[g+"_max"] = X[f].apply(np.max, 1)
                X[g+"_mean"] = X[f].apply(np.mean, 1)
                X[g+"_median"] = X[f].apply(np.median, 1)
                X[g+"_mode"] = stats.mode(X[f], 1)[0]
                X[g+"_std"] = X[f].apply(np.std, 1)
        return X, y
    def fit_transform(self, X, y = None):
        X = X.copy()
        if y is not None:
            y = y.copy()
        self.fit(X, y)
        return self.transform(X, y)



@PreprocessFactory.register('preprocess-binning')
class Binning(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_discretize):
        self.features_to_discretize = features_to_discretize
        self.binns = []
    def fit(self, X, y = None):
        LOGGER.info('FIT BINNING')
        X = X.copy()
        if y is not None:
            y = y.copy()
        if len(self.features_to_discretize) > 0:
            for col in self.features_to_discretize:
                hist, bin_edg = np.histogram(X[col], bins = 'sturges')
                self.binns.append(len(hist))
            self.disc = sklearn.preprocessing.KBinsDiscretizer(n_bins=self.binns,
                                                               encode = 'ordinal',
                                                               strategy='kmeans').fit(X[self.features_to_discretize])
        return self
    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM BINNING')
        X = X.copy()
        if y is not None:
            y = y.copy()
        if len(self.features_to_discretize) > 0:
            X[self.features_to_discretize] = self.disc.transform(X[self.features_to_discretize])

        return X, y

@PreprocessFactory.register('preprocess-scaling')
class Scaling(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_columns = 'not_available', method = 'zscore'):
        self.method = method
        self.numeric_columns = numeric_columns
    def fit(self, X, y = None):
        LOGGER.info('FIT SCALING')
        X = X.copy()
        if y is not None:
            y = y.copy()
        if isinstance(self.numeric_columns, str) and self.numeric_columns == 'not_available':
            return self
        if len(self.numeric_columns) == 0:
            self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.method == 'zscore':
            self.scale = sklearn.preprocessing.StandardScaler()
        elif self.method == 'minmax':
            self.scale = sklearn.preprocessing.MinMaxScaler()
        elif self.method == 'yj':
            self.scale = sklearn.preprocessing.PowerTransformer(method = 'yeo-johnson', standardize=True)
        elif self.method == 'quantile':
            self.scale = sklearn.preprocessing.QuantileTransformer(output_distribution= 'normal')
        elif self.method == 'robust':
            self.scale = sklearn.preprocessing.RobustScaler()
        elif self.method =='maxabs':
            self.scale = sklearn.preprocessing.MaxAbsScaler()
        self.scale.fit(X.loc[:, self.numeric_columns])
        return self
    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM SCALING')
        X = X.copy()
        if y is not None:
            y = y.copy()
        if isinstance(self.numeric_columns, str) and self.numeric_columns == 'not_available':
            return X
        X.loc[:, self.numeric_columns] = self.scale.transform(X.loc[:, self.numeric_columns])
        return X, y
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)

@PreprocessFactory.register('preprocess-maketimefeature')
class MakeTimeFeature(BaseEstimator, TransformerMixin):
    def __init__(self,
                 time_columns = [],
                 list_of_feature = ['month',  'dayofweek', 'weekday', 'is_month_end', 'is_month_start', 'hour']
                 ):
        self.time_columns =time_columns
        self.list_of_feature = set(list_of_feature)
    def fit(self, X, y = None):
        LOGGER.info('FIT MakeTimeFeature')
        X = X.copy()
        if y is not None:
            y = y.copy()
        if not self.time_columns:
            self.time_columns = X.select_dtypes(include = ['datetime64[ns]']).columns.tolist()
        return self
    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM MakeTimeFeature')
        X = X.copy()
        if y is not None:
            y = y.copy()
        for col in self.time_columns:
            if 'hour' in self.list_of_feature:
                X[col+"_hour"] = X[col].dt.hour
            if 'dayofweek' in self.list_of_feature:
                X[col+"_dayofweek"] = X[col].dt.dayofweek
            if 'weekday' in self.list_of_feature:
                X[col + "_weekday"] = X[col].dt.weekday
            if 'is_month_start' in self.list_of_feature:
                X[col + "_is_month_start"] = X[col].dt.is_month_start
            if 'is_month_end' in self.list_of_feature:
                X[col + "_is_month_end"] = X[col].dt.is_month_end
        X.drop(columns = self.time_columns, errors='ignore', inplace = True)
        return X, y
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)


@PreprocessFactory.register('preprocess-outlier')
class Outlier(BaseEstimator, TransformerMixin):
    def __init__(self, methods = ['knn', 'iforest', 'pca'], contamination = 0.2, random_state = 42,  verbose = True):
        self.contamination = contamination
        self.random_state = random_state
        self.methods = methods

    def fit(self, X, y = None):
        LOGGER.info('FIT OUTLIER')
        X = X.copy()
        if y is not None:
            y = y.copy()
        self.outlier = []
        if "knn" in self.methods:
            knn = pyod.models.knn.KNN(contamination = self.contamination)
            self.outlier.append(knn)
        if 'iforest' in self.methods:
            iforest = pyod.models.iforest.IForest(contamination= self.contamination)
            self.outlier.append(iforest)

        if 'pca' in self.methods:
            pca = pyod.models.pca.PCA(contamination=self.contamination)
            self.outlier.append(pca)

        return self

    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM OUTLIER')
        X = X.copy()
        if y is not None:
            y = y.copy()
        X['vote_outlier'] = 0
        for out in self.outlier:
            X['vote_outlier'] += out.fit_predict(X.drop(columns=['vote_outlier']))
        LOGGER.warning('Remove outlier: {} rows'.format((X['vote_outlier']==len(self.methods)).sum()))
        if y is not None:
            y = y.loc[X['vote_outlier']!=len(self.methods)]
        X = X.loc[X['vote_outlier']!=len(self.methods)].drop(columns=['vote_outlier'])
        return X, y
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)


@PreprocessFactory.register('preprocess-makenonlinearfeature')
class MakeNonLinearFeature(BaseEstimator, TransformerMixin):
    def __init__(self,
                polynomial_columns = [],
                degree = 2,
                interaction_only = False,
                include_bias = False,
                other_nonlinear_feature = ["sin", "cos", "tan"]
                 ):
        self.polynomial_columns = polynomial_columns
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.other_nonlinear_feature = other_nonlinear_feature
        self.poly = sklearn.preprocessing.PolynomialFeatures(degree=degree,
                                                             interaction_only=interaction_only,
                                                             include_bias=include_bias)
    def fit(self, X, y):
        X = X.copy()
        if y is not None:
            y = y.copy()
        LOGGER.info('FIT MakeNonLinearFeature')
        if len(self.polynomial_columns) == 0:
            self.polynomial_columns = X.select_dtypes(include=[np.number]).columns
        if self.polynomial_columns is None:
            self.polynomial_columns = []
        return self
    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM MakeNonLinearFeature')
        X = X.copy()
        if y is not None:
            y = y.copy()
        if len(self.polynomial_columns) > 0:
            self.poly.fit(X[self.polynomial_columns])
            poly_feature = self.poly.get_feature_names(input_features=self.polynomial_columns)

            poly_feature = [col.replace('^', 'power') for col in poly_feature]
            data = self.poly.transform(X[self.polynomial_columns])

            for idx, col in enumerate(poly_feature):
                X[col] = data[:, idx]

            if 'sin' in self.other_nonlinear_feature:
                for col in self.polynomial_columns:
                    X[col + "_sin"] = np.sin(X[col])
            if 'cos' in self.other_nonlinear_feature:
                for col in self.polynomial_columns:
                    X[col + "_cos"] = np.cos(X[col])
            if 'tan' in self.other_nonlinear_feature:
                for col in self.polynomial_columns:
                    X[col + "_tan"] = np.tan(X[col])


        return X, y
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)

@PreprocessFactory.register('preprocess-removeperfectmulticollinearity')
class RemovePerfectMulticollinearity(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y = None):
        LOGGER.info('FIT Remove Perfect Multicollinearity')
        X = X.copy()
        if y is not None:
            y = y.copy()
        corr = pd.DataFrame(np.corrcoef(X.T))
        corr.columns = X.columns
        corr.index = X.columns
        corr_matrix = abs(corr)

        corr_matrix["column"] = corr_matrix.index
        corr_matrix.reset_index(drop = True, inplace = True)
        cols = corr_matrix.column

        melt = corr_matrix.melt(id_vars = ['column'], value_vars = cols).sort_values(by = "value", ascending = False)
        melt['value'] = round(melt['value'], 2)
        c1 = melt['value'] == 1.00
        c2 = melt['column'] != melt['variable']
        melt = melt[((c1 ==True) & (c2 == True))]

        melt['all_columns'] = melt['column'] + melt['variable']
        melt['all_columns'] = [sorted(i) for i in melt['all_columns']]
        melt = melt.sort_values(by = 'all_columns')
        melt = melt.iloc[::2, :]
        self.columns_to_drop = melt['variable']
        if len(self.columns_to_drop) > 0:
            LOGGER.info('[Remove100] columns to drop: {}'.format(self.columns_to_drop))
        return self
    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM Remove Perfect Multicollinearity')
        X = X.copy()
        if y is not None:
            y = y.copy()
        X = X.drop(columns = self.columns_to_drop)
        return X, y
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)


@PreprocessFactory.register('preprocess-reducecategorywithcount')
class ReduceCategoricalWithCount(BaseEstimator, TransformerMixin):
    def __init__(self,
                 categorical_columns=[]):
        self.categorical_columns = categorical_columns
    def fit(self, X, y = None):
        LOGGER.info('FIT REDUCE CATEGORICAL WITH COUNT')
        X = X.copy()
        if y is not None:
            y = y.copy()
        self.data_count = {}
        for col in self.categorical_columns:
            if col not in self.data_count.keys():
                self.data_count[col] = {}
            self.data_count[col] = X[col].value_counts().to_dict()
    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM REDUCE CATEGORICAL WITH COUNT')
        X = X.copy()
        if y is not None:
            y = y.copy()
        for col in self.categorical_columns:
            X[col].replace(self.data_count[col].keys(), self.data_count[col].values(), inplace=True)
        return X, y
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)

@PreprocessFactory.register('preprocess-rfe')
class RecursiveFeatureElimination(BaseEstimator, TransformerMixin):
    def __init__(self,
                 estimator = None,
                 step = 1,
                 min_features_to_select = 2,
                 cv = 2):
        if cv == 0:
            self.selector = sklearn.feature_selection.RFE(estimator=estimator, step = step, n_features_to_select=min_features_to_select)
        else:
            self.selector = sklearn.feature_selection.RFECV(estimator=estimator, step = step, min_features_to_select=min_features_to_select, cv = cv, n_jobs=-1)
    def fit(self, X, y = None):
        LOGGER.info('FIT RFE')
        X = X.copy()
        if y is not None:
            y = y.copy()
        self.selector.fit(X, y)
        return self
    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM RFE')
        X = X.copy()
        if y is not None:
            y = y.copy()
        X = X.loc[:, self.selector.support_]
        return X, y
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X)
    def get_support(self, indices = False):
        return self.selector.get_support(indices=False)
    def get_ranking(self):
        return self.selector.ranking_

@PreprocessFactory.register('preprocess-reducedimension')
class ReduceDimension(BaseEstimator, TransformerMixin):
    def __init__(self,
                 method = 'pca_linear',
                 n_components = 0.99,
                 random_state = 42):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
    def fit(self, X, y = None):
        LOGGER.info('FIT REDUCE DIMENSION')
        X = X.copy()
        if y is not None:
            y = y.copy()
        if self.method == 'pca_linear':
            self.model = sklearn.decomposition.PCA(self.n_components, random_state=self.random_state).fit(X)
        if self.method == 'kernal_pca':
            self.model = sklearn.decomposition.KernelPCA(self.n_components, random_state=self.random_state).fit(X)
        if self.method == 'tsne':
            self.model = sklearn.manifold.TSNE(self.n_components, random_state=self.random_state).fit(X)
        if self.method == 'incremental':
            self.model = sklearn.decomposition.IncrementalPCA(self.n_components, random_state = random_state)
        return self
    def transform(self, X, y = None):
        LOGGER.info('TRANSFORM REDUCE DIMENSION')
        X = X.copy()
        if y is not None:
            y = y.copy()
        X = self.model.transform(X)
        return X, y
    def fit_transform(self, X, y = None):
        X = X.copy()
        self.fit(X, y)
        return self.transform(X, y)

