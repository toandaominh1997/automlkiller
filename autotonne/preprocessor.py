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
                imputer = True,
                imputer_numeric_strategy = "mean",
                imputer_categorical_strategy = "most_frequent",

                binning = False,
                binning_features_to_discretize= [],

                scaling = True,
                scaling_method = 'zscore',
                scaling_numeric_columns = [],

                outlier = False,
                outlier_method = ['pca', 'iforest', 'knn'],
                outlier_contamination = 0.2,

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
            'simpleimputer': {
                'flag': imputer,
                'numeric_strategy': imputer_numeric_strategy,
                'categorical_strategy': imputer_categorical_strategy
            },
            'categoryencoder': {
                'flag': False
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
            'outlier': {
                'flag': outlier,
                'methods': outlier_method,
                'contamination': outlier_contamination
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
        for obj in self.preprocess_objs:
            obj.fit(X, y)

    def transform(self, X, y = None, **fit_params):
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



