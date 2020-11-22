from sklearn.cluster import KMeans
from automlkiller.models.model_factory import ModelFactory
from automlkiller.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution

@ModelFactory.register('clustering-kmeans')
class KMeansContainer(KMeans):
    def __init__(self, **kwargs):
        super().__init__()
        tune_grid = {}
        tune_distributions = {}

        self.tune_grid = tune_grid
        self.tune_distributions = tune_distributions
        self.estimator = KMeans(**kwargs)
