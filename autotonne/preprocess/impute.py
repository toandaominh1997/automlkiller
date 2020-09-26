
from sklearn.base import BaseEstimator, TransformerMixin

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_strategy, category_strategy, target_variable):
        self.numeric_strategy = numeric_strategy
        self.category_strategy = category_strategy
        self.target = target_variable
    def fit(self, X, y = None):
        X = X.copy()

        return X
    def transform(self, X, y = None):
        X = X.copy()
        return X
    def fit_transform(self, X, y = None):
        X = X.copy()
        return X
