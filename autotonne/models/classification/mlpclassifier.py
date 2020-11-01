
from sklearn.neural_network import MLPClassifier
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution, CategoricalDistribution
@ModelFactory.register('classification-mlpclassifier')
class MLPClassifierContainer():
    def __init__(self, **kwargs):
        tune_grid = {}
        tune_distributions = {}
        tune_grid = {
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                0.9,
            ],
            "activation": ["tanh", "identity", "logistic", "relu"],
        }
        tune_distributions = {
            "alpha": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "activation": CategoricalDistribution(["tanh", "identity", "logistic", "relu"]),
            "learning_rate": CategoricalDistribution(["constant", "invscaling", "adaptive"])
        }
        self.tune_grid = tune_grid

        self.tune_distributions = tune_distributions
        self.estimator = MLPClassifier(**kwargs)

