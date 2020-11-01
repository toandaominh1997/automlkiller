
from sklearn.neighbors import KNeighborsClassifier
from autotonne.models.model_factory import ModelFactory
from autotonne.utils import np_list_arange, UniformDistribution, IntUniformDistribution
@ModelFactory.register('classification-kneighborsclassifier')
class KNeighborsClassifierContainer(KNeighborsClassifier):
    def __init__(self, **kwargs):
        super(KNeighborsClassifierContainer, self).__init__(**kwargs)
        tune_grid = {}
        tune_distributions = {}

        # common
        tune_grid["n_neighbors"] = range(1, 30)
        tune_grid["weights"] = ["uniform"]
        tune_grid["metric"] = ["minkowski", "euclidean", "manhattan"]


        tune_distributions["n_neighbors"] = IntUniformDistribution(1, 30)

        self.tune_grid = tune_grid
        self.tune_distributions = tune_distributions
        self.estimator = KNeighborsClassifier(**kwargs)
