from catboost import CatBoostClassifier
from autotonne.models.model_factory import ModelFactory
from autotonne.utils import np_list_arange, IntUniformDistribution, UniformDistribution
@ModelFactory.register('classification-catboostclassifier')
class CatBoostClassifierContainer(CatBoostClassifier):
    def __init__(self, logging_level = 'Silent', **kwargs):
        super(CatBoostClassifierContainer, self).__init__(**kwargs)
        tune_grid = {
            "depth": list(range(1, 12)),
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "random_strength": np_list_arange(0, 0.8, 0.1, inclusive=True),
            "l2_leaf_reg": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100, 200],
        }
        tune_distributions = {
            "depth": IntUniformDistribution(1, 11),
            "n_estimators": IntUniformDistribution(10, 300),
            "random_strength": UniformDistribution(0, 0.8),
            "l2_leaf_reg": IntUniformDistribution(1, 200, log=True),
        }
        self.tune_grid = tune_grid
        self.tune_distributions = tune_distributions
        self.estimator = CatBoostClassifier(logging_level=logging_level, **kwargs)
