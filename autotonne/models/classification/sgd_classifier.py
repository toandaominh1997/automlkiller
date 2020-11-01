from sklearn.linear_model import SGDClassifier
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution
@ModelFactory.register('classification-sgdclassifier')
class SGDClassifierContainer():
    def __init__(self, tol = 0.001, loss = 'hinge', penalty = 'l2', eta0=0.001, **kwargs):
        tune_grid = {}
        tune_distributions = {}
        tune_grid = {
            "penalty": ["elasticnet", "l2", "l1"],
            "l1_ratio": np_list_arange(0.0000000001, 1, 0.01, inclusive=False),
            "alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "fit_intercept": [True, False],
            "learning_rate": ["constant", "invscaling", "adaptive", "optimal"],
            "eta0": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
        tune_distributions = {
            "l1_ratio": UniformDistribution(0.0000000001, 0.9999999999),
            "alpha": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "eta0": UniformDistribution(0.001, 0.5, log=True),
        }
        self.tune_grid = tune_grid

        self.tune_distributions = tune_distributions
        self.estimator = SGDClassifier(tol = tol, loss = loss, penalty=penalty, eta0=eta0, **kwargs)

