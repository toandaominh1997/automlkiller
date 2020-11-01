import os
import numpy as np
import pandas as pd
import warnings
from autotonne.classification import Classification
from autotonne.utils import save_model, load_model

warnings.filterwarnings('ignore')
columns = [
"CALENDAR_DIM_ID",
"Age",
"Nationality",
"GKDiving",
"GKHandling",
"GKKicking",
"GKPositioning",
"GKReflexes",
]
def main():
    df = pd.read_csv('./data/data.csv')
    df['CALENDAR_DIM_ID'] = pd.date_range(start = '1/1/2020', periods = len(df))
    df = df.loc[:, columns].head(1000)
    X = df.drop(columns = ['Age'])
    y = df['Age']
    obj = Classification(
        groupsimilarfeature = True
    )
    obj.create_model(X, y, estimator='lgbm')
    print('estimator', obj.estimator)
    obj.compare_model(X, y)
    best_params = obj.tune_model(X, y, estimator=None)
    print('BEST_PARAMS: ', best_params)
    best_params = obj.tune_model(X, y, estimator=['classification-lgbmclassifier'])
    print('BEST_PARAMS: ', best_params)

    print('INFOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo')
    obj.create_model(X, y, estimator='classification-lgbmclassifier', estimator_params=best_params)
    save_model(obj, os.path.join(os.getcwd(), 'data/modeling.pkl'))
    load_model(model_path = os.path.join(os.getcwd(), 'data/modeling.pkl'))
if __name__ =='__main__':
    main()
