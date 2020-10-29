from enum import Enum, auto
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from models.model_factory import ModelFactory
from models.classification import *
class Classification(object):
    def __init__(self,
                 data: pd.DataFrame,
                 target: str,
                 test_size: float = 0.2,
                 preprocess: bool = True,
                 ):
        super(Classification, self).__init__()
        self.data = data
        self.target = target

    def compare_models(self,
                       cross_validation: bool = True,
                       n_splits: int = 2,
                       sort: str = 'Accuracy',
                       verbose: bool = True):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        X = self.data.drop(columns=[self.target])
        y = self.data[self.target]
        models = []
        scores = {}
        for name_model in ModelFactory.name_registry:
            if 'classification' in name_model:
                model = ModelFactory.create_executor(name_model)
                score_model = {
                    "accuracy_score": []
                }
                if name_model not in scores.keys():
                    scores[name_model] = score_model
                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X.loc[train_index], X.loc[test_index]
                    y_train, y_test = y.loc[train_index], y.loc[test_index]
                    model.fit(X = X_train, y = y_train)
                    scores[name_model]['accuracy_score'].append(accuracy_score(y_test, model.predict(X_test)))
                for key, value in scores[name_model].items():
                    scores[name_model][key] = np.mean(value)
        print('scores: ', scores)
                


if __name__=='__main__':
    X, y = make_classification(n_samples=10000, n_features=50)
    data = pd.DataFrame(X)
    data['target'] = y
    print(data)
    Classification(data = data, target='target').compare_models(n_splits=5)
