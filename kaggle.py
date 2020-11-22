import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from pandarallel import pandarallel
pandarallel.initialize()

df = pd.read_csv('/home/tonne/code/automlkiller/data/riiid/train.csv', nrows= 1000000)
print('shape: ', df.shape)
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, stratify = df['answered_correctly'])
# features_df = train_df.iloc[:int(5 /10 * len(train_df))]
features_df = train.copy()
train_df = train.copy()
test_df = test.copy()
# train_df = train_df.iloc[int(5 /10 * len(train_df)):]

def maketimefeature(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    df['microsecond'] = df['timestamp'].dt.microsecond
    return df
train = maketimefeature(train)
test = maketimefeature(test)
print('answered_correctly: ', train_df['answered_correctly'].value_counts())

def user_apply(x, group = 'user_id', column = 'answered_correctly'):
    df = pd.DataFrame()
    df[f'{group}_mean_{column}'] = x.mean()
    df[f'{group}_count_{column}'] = x.count()
    df[f'{group}_std_{column}'] = x.std()
    df[f'{group}_median_{column}'] = x.median()
    # df[f'{group}_mode_{column}'] = x.mode()
    df[f'{group}_min_{column}'] = x.min()
    df[f'{group}_max_{column}'] = x.max()
    df[f'{group}_sum_{column}'] = x.sum()
    # df[f'{group}_cov_{column}'] = x.cov()
    # df[f'{group}_corr_{column}'] = x.corr()
    return df
user_answers_df = features_df[features_df['answered_correctly']!=-1].groupby('user_id').parallel_apply(user_apply, group = 'user', column='answered_correctly')
user_answers_df = user_answers_df.reset_index()
user_answers_df.drop(columns = ['level_1'], inplace=True)
print('done group user', user_answers_df.reset_index().columns)
content_answers_df = features_df[features_df['answered_correctly']!=-1].groupby('content_id').parallel_apply(user_apply, group = 'content', column='answered_correctly')
content_answers_df = content_answers_df.reset_index()
content_answers_df.drop(columns = ['level_1'], inplace=True)
print('done group content', content_answers_df.columns)

train_df = train_df[train_df['answered_correctly'] != -1]
test_df = test_df[test_df['answered_correctly']!=-1]

train_df = train_df.merge(user_answers_df, how='left', on='user_id')
train_df = train_df.merge(content_answers_df, how='left', on='content_id')

test_df = test_df.merge(user_answers_df, how='left', on='user_id')
test_df = test_df.merge(content_answers_df, how='left', on='content_id')


train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].fillna(value=False).astype(bool)
test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value=False).astype(bool)

numeric_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
train_df[numeric_columns] = SimpleImputer(strategy = 'mean').fit_transform(train_df[numeric_columns])
test_df[numeric_columns] = SimpleImputer(strategy = 'mean').fit_transform(test_df[numeric_columns])
features = [
    # "row_id",
# "timestamp",
# "user_id",
# "content_id",
# "content_type_id",
# "task_container_id",
'minute',
'second',
'microsecond'

"user_answer",
"answered_correctly",
"prior_question_elapsed_time",
"prior_question_had_explanation",
"mean_user_accuracy",
"questions_answered",
"std_user_accuracy",
"median_user_accuracy",
"skew_user_accuracy",
"mean_accuracy",
"question_asked",
"std_accuracy",
"median_accuracy",
"skew_accuracy",
]


params = {
    'bagging_fraction': 0.5817242323514327,
    'feature_fraction': 0.6884588361650144,
    'learning_rate': 0.42887924851375825, 
    'max_depth': 6,
    'min_child_samples': 946, 
    'min_data_in_leaf': 47, 
    'objective': 'cross_entropy',
    'num_leaves': 29,
    'random_state': 666,
    'num_boost_round': 500,
    'metric': 'auc'
}
X_train = train_df.drop(columns = ['answered_correctly', 'row_id', 'timestamp', 'user_id', 'content_id', 'content_type_id', 'task_container_id'])
y_train = train_df['answered_correctly']
X_test = test_df.drop(columns =  ['answered_correctly', 'row_id', 'timestamp', 'user_id', 'content_id', 'content_type_id', 'task_container_id'])
y_test = test_df['answered_correctly']
data_train = lgb.Dataset(X_train, y_train)
data_test = lgb.Dataset(X_test, y_test)
model = lgb.train(params = params, train_set = data_train, valid_sets = data_test, early_stopping_rounds = 300, verbose_eval = 50)

y_pred_proba = model.predict(X_test)
y_pred = np.where(y_pred_proba>0.3, 1, 0)
print(classification_report(y_test, y_pred))
print('AUC: ', roc_auc_score(y_test, y_pred_proba))
