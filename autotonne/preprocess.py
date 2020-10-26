from enum import Enum, auto
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
# from utils.logging import getLogger
import warnings
from preprocess.preprocess import *

warnings.filterwarnings('ignore')

class Preprocess(object):
    def __init__(self,
                 ):
        super(Preprocess, self).__init__()
    def fit(self, X, y = None):
        self.pipe.fit(X, y)
    def predict(X, y = None, **fit_params):
        self.pipe.predict(X, y, **fit_params)

    def process(self,
                imputer = 'simple',
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

                rfe = True,
                rfe_estimator = None,
                rfe_step = 1,
                rfe_min_features_to_select = 3,
                rfe_cv = 3,

                reduce = True,
                reduce_method = "pca_linear",
                reduce_n_components = 0.99,
                random_state = 42,
                n_jobs = -1,
                ):
        pipe = []

        imuter = None
        if  imputer== 'simple':
            imputer = SimpleImputer(
                numeric_strategy = imputer_numeric_strategy,
                categorical_strategy = imputer_numeric_strategy,
            )
        if imputer is not None:
            pipe.append(('imputer', imputer))

        bn = None
        if binning == True:
            bn = Binning(features_to_discretize=binning_features_to_discretize)
        if bn is not None:
            pipe.append(('binning', bn))

        scale = None
        if scaling_method is not None:
            scale = Scaling(method = scaling_method)

        if scale is not None:
            pipe.append(('scaling', scale))

        outlier_model = None
        if outlier==True:
            outlier_model = Outlier(methods = outlier_method,
                           contamination = outlier_contamination,
                           random_state = random_state,
                           )

        if outlier_model is not None:
            pipe.append(('outlier', outlier_model))

        rfe_model = None
        if rfe == True:
            rfe_model = RecursiveFeatureElimination(
                estimator = sklearn.ensemble.RandomForestClassifier(),
                step = rfe_step,
                min_features_to_select = rfe_min_features_to_select,
                cv = rfe_cv
            )
        if rfe_model is not None:
            pipe.append(('rfe', rfe_model))

        reduce_model = None
        if reduce == True:
            reduce_model = ReduceDimensionForSupervised(
                method = reduce_method,
                n_components = reduce_n_components,
                random_state = random_state
            )
        if reduce_model is not None:
            pipe.append(('reduce', reduce_model))
        self.pipe = Pipeline(pipe)
        return self.pipe

if __name__=='__main__':
    X, y = sklearn.datasets.load_linnerud(return_X_y=True, as_frame=True)
    Preprocess().process().fit_transform(X, y['Waist'])



