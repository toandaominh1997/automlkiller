from sklearn.ensemble import RandomForestClassifier
from autotonne.models.model_factory import ModelFactory
from autotonne.utils import IntUniformDistribution, UniformDistribution, np_list_arange
@ModelFactory.register('classification-randomforestclassifier')
class RandomForestClassifierContainer(RandomForestClassifier):
    def __init__(self, **kwargs):
        super(RandomForestClassifierContainer, self).__init__(**kwargs)
        tune_grid = {}
        tune_distributions = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [
                0,
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
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_features": [1.0, "sqrt", "log2"],
            "bootstrap": [True, False],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "max_depth": IntUniformDistribution(1, 11),
            "min_impurity_decrease": UniformDistribution(0.000000001, 0.5, log=True),
            "max_features": UniformDistribution(0.4, 1),
        }
        self.tune_grid = tune_grid
        self.tune_distributions = tune_distributions
        self.estimator = RandomForestClassifier(**kwargs)
