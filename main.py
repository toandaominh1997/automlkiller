import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from autotonne.classification import Classification

columns = [
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
    df = df.loc[:, columns]
    print(df.head())
    obj = Classification(data = df, target='Age')
    obj.compare_models()
    obj.tune_models()
if __name__ =='__main__':
    main()
