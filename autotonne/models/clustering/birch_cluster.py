from sklearn.cluster import Birch
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution

@ModelFactory.register('clustering-birch')
class BirchContainer(Birch):
    def __init__(self, **kwargs):
        super().__init__()
        tune_grid = {}
        tune_distributions = {}

        self.tune_grid = tune_grid
        self.tune_distributions = tune_distributions
        self.estimator = Birch(**kwargs)
