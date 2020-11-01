from enum import Enum, auto
import numpy as np
import pandas as pd
import sklearn
from imblearn.pipeline import Pipeline
from autotonne.preprocess.preprocess import *
from autotonne.preprocess.preprocee_factory import PreprocessFactory
from autotonne.utils import LOGGER

class Preprocess(object):
    def __init__(self,
                datatype = True,
                datatype_categorical_columns = [],
                datatype_numeric_columns = [],
                datatype_time_columns = [],
                imputer = True,
                imputer_numeric_strategy = "mean",
                imputer_categorical_strategy = "most_frequent",

                zeronearzerovariance = True,
                zeronearzerovariance_threshold_first = 0.1,
                zeronearzerovariance_threshold_second = 20,

                categoryencoder = True,
                categoryencoder_cols = [],
                categoryencoder_method = 'targetencoder',

                groupsimilarfeature = True,
                groupsimilarfeature_group_name = ['OMG', 'ONEID'],
                groupsimilarfeature_list_of_group_feature = [['GKDiving', 'GKHandling'], ['GKKicking', 'GKReflexes']],

                binning = False,
                binning_features_to_discretize= [],

                maketimefeature =True,
                makefeature_time_columns = [],
                maketimefeature_list_of_feature = ['month',  'dayofweek', 'weekday', 'is_month_end', 'is_month_start', 'hour'],

                scaling = True,
                scaling_method = 'zscore',
                scaling_numeric_columns = [],

                outlier = True,
                outlier_method = ['pca', 'iforest', 'knn'],
                outlier_contamination = 0.2,

                makenonlinearfeature = True,
                makenonlinearfeature_polynomial_columns = [],
                makenonlinearfeature_degree = 2,
                makenonlinearfeature_intergration_only = False,
                makenonlinearfeature_include_bias = False,
                makenonlinearfeature_other_nonlinear_feature = ["sin", "cos", "tan"],

                rfe = False,
                rfe_estimator = None,
                rfe_step = 1,
                rfe_min_features_to_select = 3,
                rfe_cv = 3,

                reducedimension = False,
                reducedimension_method = "pca_linear",
                reducedimension_n_components = 0.99,
                random_state = 42,
                n_jobs = -1,
                 ):
        super(Preprocess, self).__init__()
        self.params = {
            'datatype': {
                'flag': datatype,
                'numeric_columns': datatype_numeric_columns,
                'categorical_columns': datatype_categorical_columns,
                'time_columns': datatype_time_columns
            },
            'simpleimputer': {
                'flag': imputer,
                'numeric_strategy': imputer_numeric_strategy,
                'categorical_strategy': imputer_categorical_strategy
            },

            'zeronearzerovariance': {
                'flag': zeronearzerovariance,
                'threshold_first': zeronearzerovariance_threshold_first,
                'threshold_second': zeronearzerovariance_threshold_second
            },

            'categoryencoder': {
                'flag': categoryencoder,
                'cols': categoryencoder_cols,
                'method': categoryencoder_method
            },

            'groupsimilarfeature': {
                'flag': groupsimilarfeature,
                'group_name': groupsimilarfeature_group_name,
                'list_of_group_feature': groupsimilarfeature_list_of_group_feature
            },

            'binning': {
                'flag': binning,
                'features_to_discretize': binning_features_to_discretize
            },
            'scaling': {
                'flag': scaling,
                'method': scaling_method,
                'numeric_columns': scaling_numeric_columns
            },

            'maketimefeature': {
                'flag': maketimefeature,
                'time_columns': makefeature_time_columns,
                'list_of_feature': maketimefeature_list_of_feature
            },
            'outlier': {
                'flag': outlier,
                'methods': outlier_method,
                'contamination': outlier_contamination
            },

            'makenonlinearfeature': {
                'flag': makenonlinearfeature,
                "polynomial_columns": makenonlinearfeature_polynomial_columns,
                "degree": makenonlinearfeature_degree,
                "interaction_only": makenonlinearfeature_intergration_only,
                "include_bias": makenonlinearfeature_include_bias,
                "other_nonlinear_feature": makenonlinearfeature_other_nonlinear_feature
            },
            'rfe': {
                'flag': rfe,
                'estimator': rfe_estimator,
                'step': rfe_step,
                'min_features_to_select': rfe_min_features_to_select,
                'cv': rfe_cv
            },
            'reducedimension': {
                'flag': reducedimension,
                'method': reducedimension_method,
                'n_components': reducedimension_n_components
            }
        }
        self.preprocess_objs = []
        for name_registry, preprocess_param in self.params.items():
            if preprocess_param['flag'] == False or preprocess_param['flag'] is None:
                continue
            preprocess_param.pop('flag')
            name_registry = '-'.join(['preprocess', name_registry])
            LOGGER.info('preprocess registry {}'.format(name_registry))
            preprocess_object = PreprocessFactory.create_executor(name = name_registry,
                                                                  **preprocess_param)
            self.preprocess_objs.append(preprocess_object)

    def fit(self, X, y = None, **fit_params):
        X = X.copy()
        if y is not None:
            y = y.copy()
            if isinstance(y, pd.Series):
                y = pd.DataFrame(y)
            y.columns = [str(col) for col in y.columns.tolist()]
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        X.columns = [str(col) for col in X.columns.tolist()]
        X = X.loc[:, ~X.columns.duplicated()]
        X.dropna(axis=1, how='all', inplace=True)
        for obj in self.preprocess_objs:
            obj.fit(X, y)

    def transform(self, X, y = None, **fit_params):
        X = X.copy()
        if y is not None:
            y = y.copy()
            if isinstance(y, pd.Series):
                y = pd.DataFrame(y)
            y.columns = [str(col) for col in y.columns.tolist()]

        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        X.columns = [str(col) for col in X.columns.tolist()]
        X = X.loc[:, ~X.columns.duplicated()]
        X.dropna(axis=1, how='all', inplace=True)
        data = X, y
        for obj in self.preprocess_objs:
            data = obj.transform(*data)
        if data[1] is None:
            return data[0]
        return data
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)

if __name__=='__main__':
    X, y = sklearn.datasets.load_linnerud(return_X_y=True, as_frame=True)
    data = Preprocess().fit_transform(X)
    print(data)



