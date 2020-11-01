
import os
import numpy as np
import pandas as pd
import warnings
from autotonne.classification import Classification
from autotonne.utils import save_model, load_model
from sklearn.datasets import make_classification, load_iris

warnings.filterwarnings('ignore')


if __name__ =='__main__':
    X, y = load_iris(return_X_y=True, as_frame=True)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    obj = Classification(X, y,
        groupsimilarfeature = False
    )
    # obj.create_model(estimator=['lgbmclassifier', 'kneighborsclassifier'])
    obj.compare_model(scoring=['balanced_accuracy', 'roc_auc_ovr', 'f1_micro'])
    print(obj.report_classification())
    estimator_params = obj.tune_model()
    obj.compare_model(scoring=['balanced_accuracy', 'roc_auc_ovr', 'f1_micro'],
                      estimator_params = estimator_params)
    obj.ensemble_model(scoring=['balanced_accuracy', 'roc_auc_ovr', 'f1_micro'])
    print(obj.report_classification())
    obj.voting_model(scoring=['balanced_accuracy', 'roc_auc_ovr', 'f1_micro'])
    print(obj.report_classification())
    obj.stacking_model(scoring=['balanced_accuracy', 'roc_auc_ovr', 'f1_micro'])
    print(obj.report_classification())
    obj.report_classification().to_csv('./data/report.csv')
