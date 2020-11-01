from enum import Enum, auto
import numpy as np
import pandas as pd
import json

import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from autotonne.preprocessor import Preprocess
from autotonne.models.model_factory import ModelFactory
from autotonne.models.classification import *
from autotonne.metrics.classification import ClassificationMetricContainer

#Tuning
import optuna
from ray.tune.sklearn import TuneGridSearchCV
from ray.tune.sklearn import TuneSearchCV
from autotonne.utils.distributions import get_optuna_distributions

from autotonne.utils import LOGGER, can_early_stop
class Classification(object):
    def __init__(self,
                 X,
                 y,
                 test_size: float = 0.2,
                 preprocess: bool = True,
                 **kwargs
                 ):
        super(Classification, self).__init__()
        X, y = Preprocess(**kwargs).fit_transform(X, y)
        self.X = pd.DataFrame(X).reset_index(drop=True)
        self.y = pd.DataFrame(y).reset_index(drop=True)
        del X, y

    def create_models(self,
                      estimator,
                      cv: int  = 2,
                      scoring = ['roc_auc_ovr'],
                      fit_params = {},
                      n_jobs = -1,
                      verbose = False,
                      **kwargs):
        """
        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.
        **kwargs:
            Additional keyword arguments to pass to the estimator.
        """
        X = self.X
        y = self.y
        score_models = {}
        for name_model in ModelFactory.name_registry:
            if str(estimator) in name_model:
                model = ModelFactory.create_executor(name_model, **kwargs)
                estimator = model.estimator
                scores = sklearn.model_selection.cross_validate(estimator = estimator,
                                                                X = X,
                                                                y = y,
                                                                scoring=scoring,
                                                                cv = cv,
                                                                n_jobs = n_jobs,
                                                                verbose = verbose,
                                                                fit_params = fit_params,
                                                                return_train_score = True,
                                                                return_estimator = False)


                name_model = ''.join(name_model.split('-')[1:])
                for key, values in scores.items():
                    for i, value in enumerate(values):
                        if name_model not in score_models.keys():
                            score_models[name_model] = {}
                        score_models[name_model][key + "_{}fold".format(i + 1)] = value

        score_models = pd.read_json(json.dumps(score_models))
        print('score_models: ', score_models)
        return score_models


    def compare_models(self,
                      cv: int  = 2,
                      scoring = ['roc_auc_ovr'],
                      fit_params = {},
                      n_jobs = -1,
                      verbose = False,
                      **kwargs):
        X = self.X
        y = self.y
        score_models = {}
        for name_model in ModelFactory.name_registry:
            if 'classification' in name_model:
                model = ModelFactory.create_executor(name_model, **kwargs)
                estimator = model.estimator
                scores = sklearn.model_selection.cross_validate(estimator = estimator,
                                                                X = X,
                                                                y = y,
                                                                scoring=scoring,
                                                                cv = cv,
                                                                n_jobs = n_jobs,
                                                                verbose = verbose,
                                                                fit_params = fit_params,
                                                                return_train_score = True,
                                                                return_estimator = False)
                name_model = ''.join(name_model.split('-')[1:])
                for key, values in scores.items():
                    for i, value in enumerate(values):
                        if name_model not in score_models.keys():
                            score_models[name_model] = {}
                        score_models[name_model][key + "_{}fold".format(i + 1)] = value

        score_models = pd.read_json(json.dumps(score_models))
        print('score_models: ', score_models)
        return score_models



    def tune_models(self,
                    estimator = None,
                    fold = None,
                    n_iter = 10,
                    optimize = 'accuracy',
                    search_library: str = 'optuna',
                    search_algorithm = 'random',
                    early_stopping = 'asha',
                    early_stopping_max_iters = 10,
                    verbose = True,
                    n_jobs = -1
                    ):
        LOGGER.info('tune models')
        best_params_model = {}
        model_grid = None
        print(ModelFactory.name_registry)
        for name_model in ModelFactory.name_registry:
            LOGGER.info('tunning model_name: {}'.format(name_model))
            model = ModelFactory.create_executor(name_model)
            estimator = model.estimator

            parameter_grid = model.tune_grid
            parameter_distributions = model.tune_distributions
            if (search_library == 'scikit-learn' or search_library == 'tune-sklearn') and (search_algorithm == 'grid' or search_algorithm == 'random'):
                parameter_grid = model.tune_grid
            else:
                parameter_grid = model.tune_distributions
            model_grid = None
            if search_library == 'optuna':
                pruner_translator = {
                    "asha": optuna.pruners.SuccessiveHalvingPruner(),
                    "hyperband": optuna.pruners.HyperbandPruner(),
                    "median": optuna.pruners.MedianPruner(),
                    False: optuna.pruners.NopPruner(),
                    None: optuna.pruners.NopPruner(),
                }
                pruner = early_stopping
                if pruner in pruner_translator:
                    pruner = pruner_translator[early_stopping]

                sampler_translator = {
                    "tpe": optuna.samplers.TPESampler(seed=24),
                    "random": optuna.samplers.RandomSampler(seed=24),
                }
                sampler = sampler_translator[search_algorithm]

                try:
                    param_grid = get_optuna_distributions(parameter_distributions)
                except:
                    logger.warning(
                        "Couldn't convert param_grid to specific library distributions. Exception:"
                    )
                    logger.warning(traceback.format_exc())
                study = optuna.create_study(
                    direction = 'maximize', sampler = sampler, pruner = pruner
                )
                LOGGER.info('Initializing optuna.intergration.OptnaSearchCV')
                model_grid = optuna.integration.OptunaSearchCV(
                    estimator = estimator,
                    param_distributions = param_grid,
                    max_iter = early_stopping_max_iters,
                    n_jobs = n_jobs,
                    n_trials = n_iter,
                    random_state = 24,
                    scoring = optimize,
                    study = study,
                    refit = False,
                    verbose = verbose,
                    error_score = 'raise'
                )
            elif search_library == 'tune-sklearn':
                early_stopping_translator = {
                                            "asha": "ASHAScheduler",
                                            "hyperband": "HyperBandScheduler",
                                            "median": "MedianStoppingRule",
                                        }
                if early_stopping in early_stopping_translator:
                    early_stopping = early_stopping_translator[early_stopping]
                do_early_stop = early_stopping and can_early_stop(estimator, True, True, True, parameter_grid)

                if search_algorithm == 'grid':

                    LOGGER.info('Initializing tune_sklearn.TuneGridSearchCV')
                    model_grid = TuneGridSearchCV(
                        estimator = estimator,
                        param_grid = parameter_grid,
                        early_stopping = do_early_stop,
                        scoring = optimize,
                        cv = fold,
                        max_iters=early_stopping_max_iters,
                        refit = True,
                        n_jobs = n_jobs
                    )



            model_grid.fit(self.X, self.y)
            best_params = model_grid.best_params_
            LOGGER.info('best_params: {}'.format(best_params))
            best_params_model[name_model] = best_params
        return best_params_model


if __name__=='__main__':
    X, y = make_classification(n_samples=100000, n_features=50)
    data = pd.DataFrame(X)
    data['target'] = y
    obj = Classification(data = data, target='target')
    obj.create_models(estimator = 'LGBM')
    # obj.tune_models()
