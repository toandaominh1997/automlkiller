import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from autotonne.classification import Classification

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
    print(df.head())
    obj = Classification(data = df, target='Age')
    obj.create_models(estimator='LGBM')
    # obj.compare_models()
    # obj.tune_models()
if __name__ =='__main__':
    main()
