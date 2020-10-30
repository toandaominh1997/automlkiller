from xgboost import XGBClassifier
from autotonne.models.model_factory import ModelFactory
from autotonne.utils import np_list_arange, IntUniformDistribution, UniformDistribution
@ModelFactory.register('classification-xgbclassifier')
class XGBClassifierContainer(XGBClassifier):
    def __init__(self, **kwargs):
        super(XGBClassifierContainer, self).__init__(**kwargs)
        tune_grid = {
            "learning_rate": np_list_arange(0.001, 0.5, 0.001, inclusive=True),
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "subsample": [0.2, 0.3, 0.5, 0.7, 0.9, 1],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "colsample_bytree": [0.5, 0.7, 0.9, 1],
            "min_child_weight": [1, 2, 3, 4],
            "reg_alpha": [
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
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "reg_lambda": [
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
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "scale_pos_weight": np_list_arange(0, 50, 0.1, inclusive=True),
        }
        tune_distributions = {
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "n_estimators": IntUniformDistribution(10, 300),
            "subsample": UniformDistribution(0.2, 1),
            "max_depth": IntUniformDistribution(1, 11),
            "colsample_bytree": UniformDistribution(0.5, 1),
            "min_child_weight": IntUniformDistribution(1, 4),
            "reg_alpha": UniformDistribution(0.0000000001, 10, log=True),
            "reg_lambda": UniformDistribution(0.0000000001, 10, log=True),
            "scale_pos_weight": UniformDistribution(1, 50),
        }
        self.tune_grid = tune_grid
        self.tune_distributions = tune_distributions
        self.estimator = XGBClassifier(**kwargs)
