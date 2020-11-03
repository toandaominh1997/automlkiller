from sklearn.ensemble import AdaBoostClassifier
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution
@ModelFactory.register('classification-adaboostclassifier')
class AdaBoostClassifierContainer():
    def __init__(self, **kwargs):
        tune_grid = {}
        tune_distributions = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "learning_rate": np_list_arange(0.001, 0.5, 0.001, inclusive=True),
            "algorithm": ["SAMME", "SAMME.R"],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
        }

        self.tune_grid = tune_grid
        self.tune_distributions = tune_distributions
        self.estimator = AdaBoostClassifier(**kwargs)

