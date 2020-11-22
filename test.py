import os
import numpy as np
import pandas as pd
import warnings
from automlkiller.classification import Classification
from automlkiller.utils import save_model, load_model
from sklearn.datasets import make_classification


warnings.filterwarnings('ignore')


if __name__ =='__main__':
    X, y = make_classification(n_samples=1000, n_features=50)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    obj = Classification(X, y,
        groupsimilarfeature = False
    )
    obj.create_model(estimator='lgbm')
    print('estimator', obj.estimator)
    obj.compare_model()
    # best_params = obj.tune_model(X, y, estimator=None)
    # print('BEST_PARAMS: ', best_params)
    best_params = obj.tune_model(estimator=['classification-lgbmclassifier'])
    print('BEST_PARAMS: ', best_params)

    print('INFOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo')
    obj.create_model(estimator='classification-lgbmclassifier', estimator_params=best_params)
    save_model(obj, os.path.join(os.getcwd(), 'data/modeling.pkl'))
    load_model(model_path = os.path.join(os.getcwd(), 'data/modeling.pkl'))
