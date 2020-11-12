
from enum import Enum, auto
import numpy as np
import pandas as pd
import json
from copy import deepcopy
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from autotonne.preprocessor import Preprocess
from autotonne.models.model_factory import ModelFactory
from autotonne.models.clustering import *

#Tuning
import optuna
from ray.tune.sklearn import TuneGridSearchCV
from ray.tune.sklearn import TuneSearchCV
from autotonne.utils.distributions import get_optuna_distributions

from autotonne.utils import LOGGER, can_early_stop
class Clustering(object):
    def __init__(self,
                X, 
                preprocess: bool = True,
                **kwargs
                ):
        super(Clustering, self).__init__()
        self.preprocess = preprocess
        if self.preprocess == True:
            self.preprocessor = Preprocess(**kwargs)
            X, _ = self.preprocessor.fit_transform(X, None)
        self.X = X
        self.estimator = {}
        self.model = {}
        self.metrics = {}
        self.estimator_params = {}

    def create_model(self,
                      X,
                      num_clusters,
                      estimator,
                      fit_params = {},
                      estimator_params = {},
                      n_jobs = -1,
                      verbose = False,
                      ):
        """
        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.
        **kwargs:
            Additional keyword arguments to pass to the estimator.
        """
        if self.preprocess == True:
            self.preprocessor.fit(X, None)
            X = self.preprocessor.transform(X, None)
        X = pd.DataFrame(X).reset_index(drop=True)

        score_models = {}
        for name_model in ModelFactory.name_registry:
            if str(estimator) in name_model:
                if name_model in estimator_params.keys():
                    estimator_param = estimator_params[name_model]
                else:
                    estimator_param = estimator_params
                model = ModelFactory.create_executor(name_model, **estimator_param)
                estimator = model.estimator
                estimator.fit(X)
                self.estimator[name_model] = estimator
    def predict(self, X, sample_weight = None):
        if self.preprocess == True:
            X = self.preprocessor.transform(X)
        labels = {}
        for name_model, estimator in self.estimator.items():
            labels[name_model] = estimator.predict(X)
        return labels




