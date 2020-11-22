from sklearn.base_estimator import BaseEstimator, TransformerMixin
from automlkiller.models.model_factory import ModelFactory
from annoy import AnnoyIndex

@ModelFactory.register('recommendation-annoy')
class AnnoyRecommendationContainer():
    def __init__(self, **kwargs):

        self.estimator = AnnoyRecommendation(**kwargs)




class AnnoyRecommendation(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.annoy = AnnoyIndex(**kwargs)
    def fit(self, X, y = None):
        for i, v in enumerate(X.values):
            self.annoy.add_item(i, v)

        self.annoy.build(1000)
        return self
    def predict(self, X, k = 10):
        results = []
        for v in X.values:
            results.append(self.annoy.get_nns_by_vector(v, k))
        return results
