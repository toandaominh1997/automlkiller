
import os
import time
import numpy as np
import pandas as pd
import lightgbm
import warnings
from autotonne.classification import AUTOML
from autotonne.utils import save_model, load_model
from sklearn.datasets import make_classification, load_iris
import sklearn
warnings.filterwarnings('ignore')


if __name__ =='__main__':
    df = pd.read_csv('./data.csv').head(5000)
    print('columns: ', df.columns)
    df = df.drop(columns = ['calendar_dim_id', 'user_id', 'Unnamed: 0'])
    X = df.drop(columns = ['has_order'])

    y = df['has_order']
    obj = AUTOML(X, y,
                    cleancolumnname = {},
                    datatype = {"categorical_columns": [], "numeric_columns":[], "time_columns":[]},
                    simpleimputer =  {"numeric_strategy": "mean", "categorical_strategy": "most_frequent"},
                    zeronearzerovariance = {"threshold_first" : 0.1, "threshold_second": 20},
                    categoryencoder = {"cols": [], "method": "targetencoder"},
                    groupsimilarfeature = {"group_name": [], "list_of_group_feature": []},
                    binning = {"features_to_discretize": []},
                    maketimefeature = {"time_columns": [], "list_of_feature": ['month',  'dayofweek', 'weekday', 'is_month_end', 'is_month_start', 'hour']},
                    scaling = {"method": "zscore", "numeric_columns": []},
                    # outlier = {"methods": ["pca", "iforest", "knn"], "contamination": 0.2},
                    removeperfectmulticollinearity = {},
                    makenonlinearfeature = {"polynomial_columns": [], "degree": 2, "interaction_only": False, "include_bias": False, "other_nonlinear_feature": ["sin", "cos", "tan"]},
                    # rfe = {"estimator": None, "step": 1, "min_features_to_select": 3, "cv": 3},
                    # reducedimension = {"method": "pca_linear", "n_components": 0.99}
                 )
    obj.create_model(estimator=['classification-lgbmclassifier',
                                # 'classification-kneighborsclassifier',
                                # 'classification-logisticregression',
                                # 'classification-xgbclassifier',
                                # 'classification-catboostclassifier',
                                # 'classification-randomforestclassifier'
                                ],
                     verbose = True,
                     n_jobs = -1,
                     cv = 2,
                     estimator_params = {
                                        # 'classification-kneighborsclassifier': {'n_jobs': 8},
                                        #  'classification-logisticregression': {'n_jobs': 8},
                                        'classification-lgbmclassifier': {'n_jobs': 8},
                                        #  'classification-xgbclassifier': {'n_jobs': 8},
                                    #  'classification-randomforestclassifier': {'n_jobs': 8}
                                         },
                    scoring = ['accuracy', 'roc_auc', 'recall', 'precision', 'f1']
                     )
    # estimator_params = obj.tune_model(estimator=['classification-lgbmclassifier',
    #                             'classification-kneighborsclassifier',
    #                             'classification-logisticregression',
    #                             'classification-xgbclassifier',
    #                             'classification-catboostclassifier',
    #                             'classification-randomforestclassifier'
    #                             ],
    #                n_iter=10,
    #                  verbose = True,
    #                  n_jobs = -1
    #                )
    # print('PREPARE PREDICT MODEL')
    # y_pred = obj.predict_model(X)
    # print('y_pred: ', y_pred)
    # y_pred_proba = obj.predict_proba_model(X)
    # print('y_pred_proba: ', y_pred_proba)
    # result = obj.report_classification(sort_by = 'test_roc_auc_1fold')
    # print('result: ', result.T)
    # result.to_csv('./data/report.csv')
    obj.feature_visualizer()
    obj.target_visualizer()
    obj.evaluate_visualizer()
