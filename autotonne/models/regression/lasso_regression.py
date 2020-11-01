from sklearn.linear_model import Lasso
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution

@ModelFactory.register('regression-lasso')
class LassoRegressionContainer(Lasso):
    def __init__(self, **kwargs):
        super().__init__()
        tune_grid = {}
        tune_distributions = {}

        tune_grid = {
            "alpha": np_list_arange(0.01, 10, 0.01, inclusive=True),
            "fit_intercept": [True, False],
            "normalize": [True, False],
        }
        tune_distributions = {"alpha": UniformDistribution(0.001, 10)}

        self.tune_grid = tune_grid
        self.tune_distributions = tune_distributions
        self.estimator = Lasso(**kwargs)
