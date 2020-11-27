# automlkiller

Automated Machine Learning

## Usage

1. Step 1: Load data and Preprocessing

```python
model = AUTOML(X, y,
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
```

2. Step 2: Training Model


```python
model.create_model(estimator=['classification-lgbmclassifier',
                            # 'classification-kneighborsclassifier',
                            'classification-logisticregression',
                            # 'classification-xgbclassifier',
                            # 'classification-catboostclassifier',
                            # 'classification-randomforestclassifier'
                            ],
                verbose = True,
                n_jobs = 2,
                cv = 2,
                estimator_params = {
                            'classification-lgbmclassifier': {'n_jobs': 8},
                },
                scoring = ['accuracy', 'roc_auc', 'recall', 'precision', 'f1']
            )
model.ensemble_model(scoring = ['accuracy'])
model.voting_model(scoring = ['accuracy'])
model.stacking_model(scoring = ['accuracy'])
```
3. Step 3: Model Performance
```python
model.report_tensorboard()
```
