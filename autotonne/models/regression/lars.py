from sklearn.linear_model import Lars
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution

@ModelFactory.register('regression-lars')
class LarsRegressionContainer(Lars):
    def __init__(self, **kwargs):
        super().__init__()
        tune_grid = {}
        tune_distributions = {}

        tune_grid = {
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "eps": [
                0.00001,
                0.0001,
                0.001,
                0.01,
                0.05,
                0.0005,
                0.005,
                0.00005,
                0.02,
                0.007,
                0.1,
            ],
        }
        tune_distributions = {
            "eps": UniformDistribution(0.00001, 0.1),
        }

        self.tune_grid = tune_grid
        self.tune_distributions = tune_distributions
        self.estimator = Lars(**kwargs)
