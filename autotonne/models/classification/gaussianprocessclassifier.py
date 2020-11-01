from sklearn.gaussian_process import GaussianProcessClassifier
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution
@ModelFactory.register('classification-logisticregression')
class LogisticRegressionContainer():
    def __init__(self, **kwargs):
        tune_grid = {}
        tune_distributions = {}
        tune_grid = {
            "max_iter_predict": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,]
        }
        tune_distributions = {"max_iter_predict": IntUniformDistribution(100, 1000)}
        self.tune_grid = tune_grid

        self.tune_distributions = tune_distributions
        self.estimator = GaussianProcessClassifier(**kwargs)

