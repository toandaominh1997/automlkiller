import os
import numpy as np
import pandas as pd
import warnings
from autotonne.classification import Classification
from autotonne.utils import save_model, load_model
from cloudservice import CloudService


import re
warnings.filterwarnings('ignore')
with open('./query.sql') as fp:
    query_string = fp.read()
df_path = './data.csv'
if os.path.isfile(df_path) == False:
    print('Read GBQ')
    df = CloudService(project = 'vinid-data-science-prod').read_gbq(query_string)
    df.to_csv(df_path)
else:
    df = pd.read_csv(df_path)
df = df.drop(columns = ['Unnamed: 0', 'order_datetime', 'calendar_dim_id', 'user_id', 'payamount1month'])
X = df.drop(columns = ['has_order'])

y = df['has_order']

print('data: ', df.columns)
print((df.isna().sum()/len(df)).sort_values())
model = Classification(X,
                       y,
                       groupsimilarfeature = False,
                       zeronearzerovariance = True,
                       categoryencoder=  True,
                       categoryencoder_cols = [],
                       categoryencoder_method = 'targetencoder',
                       makenonlinearfeature = False,
                       outlier = False,
                       removeperfectmulticollinearity = True,
                       )

for col in model.X.columns:
    print(col)
# model.create_model(estimator = ['classification-adaboostclassifier'],
#                    scoring = ['accuracy', 'f1', 'roc_auc'],
#                    estimator_params = {'classification-lgbmclassifier': {'num_leaves': 100}})

model.create_model(estimator = None,
                   scoring = ['accuracy', 'f1', 'roc_auc',  'recall', 'precision'],
                   estimator_params = {'classification-lgbmclassifier': {'num_leaves': 100}})

best_params = model.tune_model(estimator=None)

model.voting_model(scoring = ['accuracy', 'roc_auc', 'recall', 'precision', 'f1'])
model.stacking_model(scoring = ['accuracy', 'roc_auc', 'recall','precision', 'f1'])
model.ensemble_model(scoring = ['accuracy', 'roc_auc', 'recall', 'precision', 'f1'])
model.voting_model(scoring = ['accuracy', 'roc_auc', 'recall', 'precision', 'f1'])
model.stacking_model(scoring = ['accuracy', 'roc_auc', 'recall', 'precision', 'f1'])
df = model.report_classification(sort_by = 'test_roc_auc_1fold')
print(df)
df.to_csv('report.csv')
model.plot_model()
