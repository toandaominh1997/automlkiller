from sklearn.linear_model import LogisticRegression
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution
@ModelFactory.register('classification-logisticregression')
class LogisticRegressionContainer():
    def __init__(self, **kwargs):
        tune_grid = {}
        tune_distributions = {}
        tune_grid['penalty'] = ["l2", "none"]
        tune_grid['C'] = np_list_arange(0, 10, 1.0, inclusive=True)

        tune_distributions['C'] = UniformDistribution(0, 10)
        self.tune_grid = tune_grid

        self.tune_distributions = tune_distributions
        self.estimator = LogisticRegression(**kwargs)

