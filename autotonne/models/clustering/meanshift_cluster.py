from sklearn.cluster import MeanShift
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution

@ModelFactory.register('clustering-meanshift')
class MeanShiftContainer(MeanShift):
    def __init__(self, **kwargs):
        super().__init__()
        tune_grid = {}
        tune_distributions = {}

        self.tune_grid = tune_grid
        self.tune_distributions = tune_distributions
        self.estimator = MeanShift(**kwargs)
