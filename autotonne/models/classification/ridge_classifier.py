from sklearn.linear_model import RidgeClassifier
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution

@ModelFactory.register('classification-ridgeclassifier')
class RidgeClassifierContainer(RidgeClassifier):
    def __init__(self, **kwargs):
        # super(RidgeClassifierContainer, self).__init__()
        super().__init__()
        tune_grid = {}
        tune_distributions = {}
        tune_grid = {
            "normalize": [True, False],
        }

        tune_grid["alpha"] = np_list_arange(0.01, 10, 0.01, inclusive=False)
        tune_grid["fit_intercept"] = [True, False]
        tune_distributions["alpha"] = UniformDistribution(0.001, 10)

        self.tune_grid = tune_grid
        self.tune_distributions = tune_distributions
        self.estimator = RidgeClassification(**kwargs)

class RidgeClassification(RidgeClassifier):
    def predict_proba(self, X):
        return self.predict(X)
