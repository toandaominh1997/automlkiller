from sklearn.tree import DecisionTreeClassifier
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution
@ModelFactory.register('classification-decisiontreeclassifier')
class DecisionTreeClassifierContainer():
    def __init__(self, **kwargs):
        tune_grid = {}
        tune_distributions = {}
        tune_grid = {
            "max_depth": np_list_arange(1, 16, 1, inclusive=True),
            "max_features": [1.0, "sqrt", "log2"],
            "min_samples_leaf": [2, 3, 4, 5, 6],
            "min_samples_split": [2, 5, 7, 9, 10],
            "criterion": ["gini", "entropy"],
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
        }
        tune_distributions = {
            "max_depth": IntUniformDistribution(1, 16),
            "max_features": UniformDistribution(0.4, 1),
            "min_samples_leaf": IntUniformDistribution(2, 6),
            "min_samples_split": IntUniformDistribution(2, 10),
            "min_impurity_decrease": UniformDistribution(0.000000001, 0.5, log=True),
        }
        self.tune_grid = tune_grid

        self.tune_distributions = tune_distributions
        self.estimator = DecisionTreeClassifier(**kwargs)

