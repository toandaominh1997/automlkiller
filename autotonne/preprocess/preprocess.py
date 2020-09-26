from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import categoru_encoders
class CategoryEncoder(BaseEstimator, TransformerMixin):
    def _init__(self, dict_cols = None, strategy):
        pipes = []
        if dict_cols is None:
            for method, cols in dict_cols:
                if method =='BackwardDifferenceEncoder':
                    pipes.append(categoru_encoders.BackwardDifferenceEncoder(cols = cols))
                elif method == 'BaseNEncoder':
                    pipes.append(category_encoders.BaseNEncoder(cols=cols))
                elif method =='BinaryEncoder':
                    pipes.append(category_encoders.BinaryEncoder(cols = cols))
                elif method == 'CatBoostEncoder':
                    pipes.append(category_encoders.CatBoostEncoder(cols = cols))
                elif method == 'CountEncoder':
                    pipes.append(category_encoder.CountEncoder(cols = cols))
                elif method == 'GLMMEncoder':
                    pipes.append(GLMMEncoder(cols = cols))
























class Preprocess(Pipeline):
    def __init__(self, **kwargs):
        super(Preprocess, self).__init__(**kwargs)
