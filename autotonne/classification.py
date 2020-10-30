from enum import Enum, auto
import numpy as np
import pandas as pd
import json

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
                 data: pd.DataFrame,
                 target: str,
                 test_size: float = 0.2,
                 preprocess: bool = True,
                 ):
        super(Classification, self).__init__()
        X = data.drop(columns=[target])
        y = data[target]
        X, y = Preprocess().fit_transform(X, y)
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.DataFrame(y).reset_index(drop=True)
        self.X = X
        self.y = y

    def compare_models(self,
                       cross_validation: bool = True,
                       n_splits: int = 2,
                       sort: str = 'Accuracy',
                       verbose: bool = True):
        X = self.X
        y = self.y
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        models = []
        scores = {}
        for name_model in ModelFactory.name_registry:
            if 'classification' in name_model:
                model = ModelFactory.create_executor(name_model)
                estimator = model.estimator
                score_model = ClassificationMetricContainer()
                if name_model not in scores.keys():
                    scores[name_model] = score_model
                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X.loc[train_index], X.loc[test_index]
                    y_train, y_test = y.loc[train_index], y.loc[test_index]
                    estimator.fit(X = X_train, y = y_train)
                    score_model.classification_report(y_test, estimator.predict(X_test))
                    try:
                        score_model.classification_report_proba(y_test, estimator.predict_proba(X_test)[:, 1])
                    except:
                        LOGGER.warn('{} has no attribute predict proba'.format(name_model))
                scores[name_model] = score_model.score_mean()
        scores = pd.read_json(json.dumps(scores))
        print('scores: ', scores)
        return scores




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
    obj = Classification(data = data, target='target').compare_models()
    obj.tune_models()
