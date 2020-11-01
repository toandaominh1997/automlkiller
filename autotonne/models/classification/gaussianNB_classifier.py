from sklearn.naive_bayes import GaussianNB
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution
@ModelFactory.register('classification-gaussiannb')
class GaussianNBContainer():
    def __init__(self, **kwargs):
        tune_grid = {}
        tune_distributions = {}
        tune_grid = {
            "var_smoothing": [
                0.000000001,
                0.000000002,
                0.000000005,
                0.000000008,
                0.000000009,
                0.0000001,
                0.0000002,
                0.0000003,
                0.0000005,
                0.0000007,
                0.0000009,
                0.00001,
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.007,
                0.009,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.1,
                1,
            ]
        }
        tune_distributions = {
            "var_smoothing": UniformDistribution(0.000000001, 1, log=True)
        }
        self.tune_grid = tune_grid

        self.tune_distributions = tune_distributions
        self.estimator = GaussianNB(**kwargs)

