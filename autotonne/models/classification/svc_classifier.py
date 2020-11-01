from sklearn.svm import SVC
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution
@ModelFactory.register('classification-svc')
class SVCClassifierContainer():
    def __init__(self, **kwargs):
        tune_grid = {}
        tune_distributions = {}
        tune_grid = {
            "C": np_list_arange(0, 50, 0.01, inclusive=True),
            "class_weight": ["balanced", {}],
        }
        tune_distributions = {
            "C": UniformDistribution(0, 50),
        }
        self.tune_grid = tune_grid

        self.tune_distributions = tune_distributions
        self.estimator = SVC(**kwargs)

