from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from autotonne.models.model_factory import ModelFactory
from autotonne.utils.distributions import np_list_arange, UniformDistribution, IntUniformDistribution, CategoricalDistribution
@ModelFactory.register('classification-lineardiscriminantanalysis')
class LinearDiscriminantAnalysisContainer():
    def __init__(self, **kwargs):
        tune_grid = {}
        tune_distributions = {}
        tune_grid = {
            "solver": ["lsqr", "eigen"],
            "shrinkage": [
                None,
                "auto",
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
            ],
        }
        tune_distributions = {
            "solver": CategoricalDistribution(['lsqr', 'eigen']),
            "shrinkage": UniformDistribution(0.0001, 1, log=True),
        }
        self.tune_grid = tune_grid

        self.tune_distributions = tune_distributions
        self.estimator = LinearDiscriminantAnalysis(**kwargs)

