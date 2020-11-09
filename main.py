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
    X = df
    df['target'] = [0]*500 + [1]*500
    y = df['target']
    print('value_counts: ', y.value_counts())
    obj = Classification(X, y,
        groupsimilarfeature = True
    )
    # obj.create_model(estimator=None)
    # print('estimator', obj.estimator)
    # obj.compare_model()
    # obj.report_classification()
    # obj.ensemble_model(estimator = None)
    # obj.ensemble_model(estimator=['classification-lgbmclassifier', 'classification-kneighborsclassifier'])
    # obj.voting_model()
    # obj.voting_model(estimator=['classification-lgbmclassifier', 'classification-kneighborsclassifier'])
    # obj.stacking_model(estimator=['classification-lgbmclassifier', 'classification-kneighborsclassifier'])
    # obj.report_classification()
    # best_params = obj.tune_model(estimator=['classification-lgbmclassifier'])
    # print('BEST_PARAMS: ', best_params)

    # obj.create_model(estimator=['classification-lgbmclassifier'], estimator_params=best_params)
    # save_model(obj, os.path.join(os.getcwd(), 'data/modeling.pkl'))
    # load_model(model_path = os.path.join(os.getcwd(), 'data/modeling.pkl'))
    # obj.plot_model()
if __name__ =='__main__':
    main()
