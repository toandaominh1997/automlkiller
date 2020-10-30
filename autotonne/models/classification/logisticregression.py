from sklearn.linear_model import LogisticRegression
from models.model_factory import ModelFactory
from utils.distributions import np_list_arange, UniformDistribution
@ModelFactory.register('classification-logisticregression')
class LogisticRegressionContainer(LogisticRegression):
    def __init__(self, **kwargs):
        super(LogisticRegressionContainer, self).__init__(**kwargs)

        tune_grid = {}
        tune_distributions = {}
        tune_grid['penalty'] = ["l2", "none"]
        tune_grid['C'] = np_list_arange(0, 10, 1.0, inclusive=True)

        tune_distributions['C'] = UniformDistribution(0, 10)
        self.tune_grid = tune_grid

        self.tune_distributions = tune_distributions

