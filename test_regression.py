import os
import numpy as np
import pandas as pd
import warnings
from autotonne.regression import Regression
from autotonne.utils import save_model, load_model
from sklearn.datasets import make_regression


warnings.filterwarnings('ignore')


if __name__ =='__main__':
    X, y = make_regression(n_samples=1000, n_features=50)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    obj = Regression(
        groupsimilarfeature = False
    )
    obj.create_model(X, y, estimator='lasso')
    print('estimator', obj.estimator)
    obj.compare_model(X, y)
    # best_params = obj.tune_model(X, y, estimator=None)
    # print('BEST_PARAMS: ', best_params)
    best_params = obj.tune_model(X, y, estimator=['regression-lasso'])
    print('BEST_PARAMS: ', best_params)

    print('INFOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo')
    obj.create_model(X, y, estimator='regression-lasso', estimator_params=best_params)
    save_model(obj, os.path.join(os.getcwd(), 'data/modeling.pkl'))
    load_model(model_path = os.path.join(os.getcwd(), 'data/modeling.pkl'))