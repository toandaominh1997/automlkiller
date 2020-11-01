from sklearn.neural_network import MLPRegressor
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution

@ModelFactory.register('regression-mlpregressor')
class MLPRegressorContainer(MLPRegressor):
    def __init__(self, **kwargs):
        super().__init__()
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
            "hidden_layer_size_0": [50, 100],
            "hidden_layer_size_1": [0, 50, 100],
            "hidden_layer_size_2": [0, 50, 100],
            "activation": ["tanh", "identity", "logistic", "relu"],
        }
        tune_distributions = {
            "alpha": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "hidden_layer_size_0": IntUniformDistribution(50, 100),
            "hidden_layer_size_1": IntUniformDistribution(0, 100),
            "hidden_layer_size_2": IntUniformDistribution(0, 100),
        }

        self.tune_grid = tune_grid
        self.tune_distributions = tune_distributions
        self.estimator = MLPRegressor(**kwargs)
