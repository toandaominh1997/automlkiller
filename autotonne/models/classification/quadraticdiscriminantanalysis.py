from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution
@ModelFactory.register('classification-quadraticdiscriminantanalysis')
class QuadraticDiscriminantAnalysisContainer():
    def __init__(self, **kwargs):
        tune_grid = {}
        tune_distributions = {}
        tune_grid = {"reg_param": np_list_arange(0, 1, 0.01, inclusive=True)}
        tune_distributions = {"reg_param": UniformDistribution(0, 1)}
        self.tune_grid = tune_grid

        self.tune_distributions = tune_distributions
        self.estimator = QuadraticDiscriminantAnalysis(**kwargs)

