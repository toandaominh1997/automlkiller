from sklearn.base_estimator import BaseEstimator, TransformerMixin
from autotonne.models.model_factory import ModelFactory
from lightfm import LightFM

@ModelFactory.register('recommendation-lightfm')
class LightFMRecommendationContainer():
    def __init__(self, **kwargs):

        self.estimator = LightFMRecommendation(**kwargs)




class LightFMRecommendation(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.lightfm = LightFM(**kwargs)
    def fit(self, X, y = None):
        pass
    def predict(self, X, k = 10):
        pass

