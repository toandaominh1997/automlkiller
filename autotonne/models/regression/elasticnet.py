from sklearn.linear_model import ElasticNet
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution

@ModelFactory.register('regression-elasticnet')
class ElasticNetRegressionContainer(ElasticNet):
    def __init__(self, **kwargs):
        super().__init__()
        tune_grid = {}
        tune_distributions = {}

        tune_grid = {
            "alpha": np_list_arange(0.01, 10, 0.01, inclusive=True),
            "l1_ratio": np_list_arange(0.01, 1, 0.001, inclusive=False),
            "fit_intercept": [True, False],
            "normalize": [True, False],
        }
        tune_distributions = {
            "alpha": UniformDistribution(0, 1),
            "l1_ratio": UniformDistribution(0.01, 0.9999999999),
        }

        self.tune_grid = tune_grid
        self.tune_distributions = tune_distributions
        self.estimator = ElasticNet(**kwargs)
