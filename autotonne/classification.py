import os
from enum import Enum, auto
import scipy
import numpy as np
import pandas as pd
import json
from copy import deepcopy
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier

from autotonne.preprocessor import Preprocess
from autotonne.models.model_factory import ModelFactory
from autotonne.models.classification import *

#Tuning
import optuna
from ray.tune.sklearn import TuneGridSearchCV
from ray.tune.sklearn import TuneSearchCV
# visualization
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ROCAUC, PrecisionRecallCurve, ClassPredictionError, DiscriminationThreshold
import matplotlib.pyplot as plt

from autotonne.utils.distributions import get_optuna_distributions

from autotonne.utils import LOGGER, can_early_stop
class Classification(object):
    def __init__(self,
                 X,
                 y,
                 preprocess: bool = True,
                 **kwargs
                 ):
        super(Classification, self).__init__()
        self.preprocess = preprocess
        if self.preprocess == True:
            self.preprocessor = Preprocess(**kwargs)
        if self.preprocess == True:
            X, y = self.preprocessor.fit_transform(X, y)
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.DataFrame(y).reset_index(drop=True)
        self.X = X
        self.y = y
        self.estimator = {}
        self.model = {}
        self.metrics = {}
        self.estimator_params = {}

    def create_model(self,
                      estimator,
                      cv: int  = 2,
                      scoring = ['roc_auc_ovr'],
                      sort = None,
                      fit_params = {},
                      estimator_params = {},
                      n_jobs = -1,
                      verbose = False
                     ):
        """
        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.
        **kwargs:
            Additional keyword arguments to pass to the estimator.
        """
        X = self.X
        y = self.y
        if sort is None:
            sort = scoring[0]
        score_models = {}

        estimator_model = {}
        if estimator is None:
            if len(self.estimator.keys()) > 0:
                for name_model, estimator in self.estimator.items():
                    if name_model in estimator_params.keys():
                        estimator_model[name_model] = ModelFactory.create_executor(name_model, **estimator_params[name_model])
                    else:
                        estimator_model[name_model] = estimator

            else:
                for name_model in ModelFactory.name_registry:
                    if name_model in estimator_params.keys():
                        estimator_model[name_model] = ModelFactory.create_executor(name_model, **estimator_params[name_model])
                    else:
                        estimator_model[name_model] = ModelFactory.create_executor(name_model)
        else:
            for name_model in estimator:
                if name_model in estimator_params.keys():
                    estimator_model[name_model] = ModelFactory.create_executor(name_model, **estimator_params[name_model])
                else:
                    estimator_model[name_model] = ModelFactory.create_executor(name_model)

        # update estimator_params
        for name_model, params in estimator_params.items():
            self.estimator_params[name_model] = params

        for name_model, model in estimator_model.items():
            try:
                estimator = model.estimator
            except:
                estimator = model

            scores = sklearn.model_selection.cross_validate(estimator = estimator,
                                                            X = X,
                                                            y = y,
                                                            scoring=scoring,
                                                            cv = cv,
                                                            n_jobs = n_jobs,
                                                            verbose = verbose,
                                                            fit_params = fit_params,
                                                            return_train_score = True,
                                                            return_estimator = True,
                                                            error_score=-1)
            self.estimator[name_model] = scores['estimator'][np.argmax(scores['test_'+sort])]
            scores.pop('estimator')
            name_model = ''.join(name_model.split('-')[1:])
            for key, values in scores.items():
                for i, value in enumerate(values):
                    if name_model not in self.metrics.keys():
                        self.metrics[name_model] = {}
                    self.metrics[name_model][key + "_{}fold".format(i + 1)] = value
        return self

    def tune_model(self,
                    estimator = None,
                    n_iter = 2,
                    optimize = 'accuracy',
                    search_library: str = 'optuna',
                    search_algorithm = 'random',
                    early_stopping = 'asha',
                    early_stopping_max_iters = 10,
                    estimator_params = {},
                    n_jobs = -1,
                    verbose = True,
                    ):
        X = self.X
        y = self.y
        LOGGER.info('TUNNING MODEL')
        best_params_model = {}
        model_grid = None

        estimator_model = {}
        if estimator is None:
            if len(self.estimator.keys()) > 0:
                for name_model, estimator in self.estimator.items():
                    estimator_model[name_model] = ModelFactory.create_executor(name_model)

            else:
                for name_model in ModelFactory.name_registry:
                    estimator_model[name_model] = ModelFactory.create_executor(name_model)
        else:
            for name_model in estimator:
                estimator_model[name_model] = ModelFactory.create_executor(name_model)

        # update estimator_params
        for name_model, params in estimator_params.items():
            self.estimator_params[name_model] = params

        for name_model, model in estimator_model.items():
            LOGGER.info('tunning model_name: {}'.format(name_model))
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
                    LOGGER.warn(
                        "Couldn't convert param_grid to specific library distributions. Exception:"
                    )
                    LOGGER.warn(traceback.format_exc())
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



            model_grid.fit(X, y)
            best_params = model_grid.best_params_
            best_params_model[name_model] = best_params

        # update estimator_params
        for name_model, params in best_params_model.items():
            self.estimator_params[name_model] = params
        LOGGER.info('best_params_model: {}'.format(best_params_model))
        return best_params_model
    def ensemble_model(self,
                       estimator = None,
                       method = 'bagging',
                       n_estimators = 2,
                       cv = 2,
                       scoring = ['roc_auc_ovr'],
                       sort = None,
                       fit_params = {},
                       verbose = True,
                       estimator_params = {},
                       n_jobs = -1):
        X = self.X
        y = self.y
        LOGGER.info('ensemble models')

        if sort is None:
            sort = scoring[0]
        score_models = {}

        estimator_model = {}
        if estimator is None:
            if len(self.estimator.keys()) > 0:
                for name_model, estimator in self.estimator.items():
                    if name_model in estimator_params.keys():
                        estimator_model[name_model] = ModelFactory.create_executor(name_model, **estimator_params[name_model])
                    else:
                        estimator_model[name_model] = estimator

            else:
                for name_model in ModelFactory.name_registry:
                    if name_model in estimator_params.keys():
                        estimator_model[name_model] = ModelFactory.create_executor(name_model, **estimator_params[name_model])
                    else:
                        estimator_model[name_model] = ModelFactory.create_executor(name_model)
        else:
            for name_model in estimator:
                if name_model in estimator_params.keys():
                    estimator_model[name_model] = ModelFactory.create_executor(name_model, **estimator_params[name_model])
                else:
                    estimator_model[name_model] = ModelFactory.create_executor(name_model)

        # update estimator_params
        for name_model, params in estimator_params.items():
            self.estimator_params[name_model] = params

        for name_model, model in estimator_model.items():
            try:
                estimator = model.estimator
            except:
                estimator = model

            if method == 'bagging':
                LOGGER.info('Initializing BaggingClassifer')
                estimator = BaggingClassifier(base_estimator=estimator,
                                                               n_estimators=n_estimators,
                                                               random_state=0)
            elif method == 'adaboost':
                LOGGER.info('Initializing AdaBoostClassifier')
                estimator = AdaBoostClassifier(base_estimator=estimator,
                                                                n_estimators=n_estimators,
                                                                random_state=0)

            scores = sklearn.model_selection.cross_validate(estimator = estimator,
                                                            X = X,
                                                            y = y,
                                                            scoring=scoring,
                                                            cv = cv,
                                                            n_jobs = n_jobs,
                                                            verbose = verbose,
                                                            fit_params = fit_params,
                                                            return_train_score = True,
                                                            return_estimator = True,
                                                            error_score=-1)
            self.estimator[name_model] = scores['estimator'][np.argmax(scores['test_'+sort])]
            scores.pop('estimator')
            name_model = ''.join(name_model.split('-')[1:])
            for key, values in scores.items():
                for i, value in enumerate(values):
                    if name_model not in self.metrics.keys():
                        self.metrics[name_model] = {}
                    self.metrics[name_model][key + "_{}fold".format(i + 1)] = value
        return self
    def voting_model(self,
                     estimator = None,
                       cv = 2,
                       scoring = ['roc_auc_ovr'],
                       sort = None,
                       fit_params = {},
                       verbose = True,
                       estimator_params = {},
                       n_jobs = -1):
        X = self.X
        y = self.y
        LOGGER.info('VOTING MODELs')

        if sort is None:
            sort = scoring[0]
        score_models = {}

        estimator_model = {}
        if estimator is None:
            if len(self.estimator.keys()) > 0:
                for name_model, estimator in self.estimator.items():
                    if name_model in estimator_params.keys():
                        estimator_model[name_model] = ModelFactory.create_executor(name_model, **estimator_params[name_model])
                    else:
                        estimator_model[name_model] = estimator

            else:
                for name_model in ModelFactory.name_registry:
                    if name_model in estimator_params.keys():
                        estimator_model[name_model] = ModelFactory.create_executor(name_model, **estimator_params[name_model])
                    else:
                        estimator_model[name_model] = ModelFactory.create_executor(name_model)
        else:
            for name_model in estimator:
                if name_model in estimator_params.keys():
                    estimator_model[name_model] = ModelFactory.create_executor(name_model, **estimator_params[name_model])
                else:
                    estimator_model[name_model] = ModelFactory.create_executor(name_model)

        # update estimator_params
        for name_model, params in estimator_params.items():
            self.estimator_params[name_model] = params

        model_voting = []
        for name_model, model in estimator_model.items():
            try:
                estimator = model.estimator
            except:
                estimator = model
            model_voting.append((name_model, estimator))
        name_model = 'classification-votingclassifer'
        try:
            LOGGER.info('TRY soft voting')
            estimator = VotingClassifier(estimators= model_voting,
                                     voting='soft')
            scores = sklearn.model_selection.cross_validate(estimator = estimator,
                                                X = X,
                                                y = y,
                                                scoring=scoring,
                                                cv = cv,
                                                n_jobs = n_jobs,
                                                verbose = verbose,
                                                fit_params = fit_params,
                                                return_train_score = True,
                                                return_estimator = True,
                                                error_score=-1)
        except:
            LOGGER.warn('TRY hard voting')
            estimator = VotingClassifier(estimators= model_voting,
                                     voting='hard')
            scores = sklearn.model_selection.cross_validate(estimator = estimator,
                                                X = X,
                                                y = y,
                                                scoring=scoring,
                                                cv = cv,
                                                n_jobs = n_jobs,
                                                verbose = verbose,
                                                fit_params = fit_params,
                                                return_train_score = True,
                                                return_estimator = True,
                                                error_score=-1)
        self.estimator['classification-votingclassifer'] = scores['estimator'][np.argmax(scores['test_'+sort])]
        scores.pop('estimator')
        name_model = ''.join(name_model.split('-')[1:])
        for key, values in scores.items():
            for i, value in enumerate(values):
                if name_model not in self.metrics.keys():
                    self.metrics[name_model] = {}
                self.metrics[name_model][key + "_{}fold".format(i + 1)] = value
        return self

    def stacking_model(self,
                     estimator = None,
                    final_estimator = sklearn.linear_model.LogisticRegression(),
                       cv = 2,
                       scoring = ['roc_auc_ovr'],
                       sort = None,
                       fit_params = {},
                       verbose = True,
                       estimator_params = {},
                       n_jobs = -1):
        X = self.X
        y = self.y
        LOGGER.info('VOTING MODELs')

        if sort is None:
            sort = scoring[0]
        score_models = {}

        estimator_model = {}
        if estimator is None:
            if len(self.estimator.keys()) > 0:
                for name_model, estimator in self.estimator.items():
                    if name_model in estimator_params.keys():
                        estimator_model[name_model] = ModelFactory.create_executor(name_model, **estimator_params[name_model])
                    else:
                        estimator_model[name_model] = estimator

            else:
                for name_model in ModelFactory.name_registry:
                    if name_model in estimator_params.keys():
                        estimator_model[name_model] = ModelFactory.create_executor(name_model, **estimator_params[name_model])
                    else:
                        estimator_model[name_model] = ModelFactory.create_executor(name_model)
        else:
            for name_model in estimator:
                if name_model in estimator_params.keys():
                    estimator_model[name_model] = ModelFactory.create_executor(name_model, **estimator_params[name_model])
                else:
                    estimator_model[name_model] = ModelFactory.create_executor(name_model)

        # update estimator_params
        for name_model, params in estimator_params.items():
            self.estimator_params[name_model] = params

        model_stacking = []
        for name_model, model in estimator_model.items():
            try:
                estimator = model.estimator
            except:
                estimator = model
            model_stacking.append((name_model, estimator))
        name_model = 'classification-stacking_model'
        LOGGER.info('TRY STACKING MODEL')
        estimator = StackingClassifier(estimators= model_stacking,
                                       final_estimator=final_estimator,
                                       cv = cv,
                                       n_jobs= n_jobs,
                                       verbose=verbose)
        scores = sklearn.model_selection.cross_validate(estimator = estimator,
                                            X = X,
                                            y = y,
                                            scoring=scoring,
                                            cv = cv,
                                            n_jobs = n_jobs,
                                            verbose = verbose,
                                            fit_params = fit_params,
                                            return_train_score = True,
                                            return_estimator = True,
                                            error_score=-1)
        self.estimator['classification-stackingclassifer'] = scores['estimator'][np.argmax(scores['test_'+sort])]
        scores.pop('estimator')
        name_model = ''.join(name_model.split('-')[1:])
        for key, values in scores.items():
            for i, value in enumerate(values):
                if name_model not in self.metrics.keys():
                    self.metrics[name_model] = {}
                self.metrics[name_model][key + "_{}fold".format(i + 1)] = value
        return self
    def predict_proba_model(self,
                            X,
                            estimator = None,
                            probability_threshold = None,
                            round = 4,
                            verbose = False
                            ):
        X = X.copy()
        X = self.preprocessor.transform(X)
        estimator_model = {}
        if estimator is None:
            for name_model, estimator in self.estimator.items():
                estimator_model[name_model] = estimator
        else:
            for name_model in estimator:
                if name_model in self.estimators.keys():
                    estimator_model[name_model] = self.estimators[name_model]
        preds = []
        for name_model, estimator in estimator_model.items():
            try:
                y_pred = estimator.predict_proba(X)
                preds.append(y_pred)
            except:
                LOGGER.warn(f'{estimator.__class__.__name__} not function predict_proba')
        y_pred_proba = np.mean(np.vstack(preds), axis = 1)
        return y_pred_proba


    def predict_model(self,
                    X,
                    estimator = None,
                      probability_threshold = None,
                      rountd = 4,
                      verbose = False
                      ):
        X = X.copy()
        X = self.preprocessor.transform(X, None)
        estimator_model = {}
        if estimator is None:
            for name_model, estimator in self.estimator.items():
                estimator_model[name_model] = estimator
        else:
            for name_model in estimator:
                if name_model in self.estimators.keys():
                    estimator_model[name_model] = self.estimator[name_model]
        preds = []
        for name_model, estimator in estimator_model.items():
            try:
                y_pred = estimator.predict(X)
                preds.append(y_pred)
            except:
                LOGGER.warn(f'{estimator.__class__.__name__} not function predict')
        y_pred = scipy.stats.mode(np.vstack(preds), axis = 0)[0][0].tolist()
        return y_pred


    def plot_model(self):
        LOGGER.info('Initializing plot model')
        size_plot = len(self.estimator.items())
        # fig, axes = plt.subplots(size_plot*6, figsize=(20, 20*size_plot*6))
        if os.path.isdir(os.path.join(os.getcwd(), 'viz')) == False:
            os.makedirs(os.path.join(os.getcwd(), 'viz/'))
        classes = pd.value_counts(self.y.values.flatten()).index.tolist()
        print('classes: ', classes)
        index = 0
        for idx, (name_model, estimator) in enumerate(self.estimator.items()):
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(self.X, self.y, test_size = 0.2, stratify=self.y, random_state = 24)
            try:
                viz = ClassificationReport(model = estimator, classes=classes)
                index +=1
                viz.fit(X_train, y_train)
                viz.score(X_test, y_test)
                viz.show(outpath=os.path.join(os.getcwd(), f'viz/{viz.__class__.__name__}_{estimator.__class__.__name__}_{index}.png'), clear_figure=True)
                index = index + 1
            except:
                pass
            try:
                viz = ConfusionMatrix(model = estimator, classes=classes)
                viz.fit(X_train, y_train)
                viz.score(X_test, y_test)
                viz.show(outpath=os.path.join(os.getcwd(), f'viz/{viz.__class__.__name__}_{estimator.__class__.__name__}_{index}.png'), clear_figure=True)
                index = index + 1
            except:
                pass
            try:
                viz = ROCAUC(model = estimator,classes=classes)
                viz.fit(X_train, y_train)
                viz.score(X_test, y_test)
                viz.show(outpath=os.path.join(os.getcwd(), f'viz/{viz.__class__.__name__}_{estimator.__class__.__name__}_{index}.png'), clear_figure=True)
                index = index + 1
            except:
                pass
            try:
                viz = PrecisionRecallCurve(model = estimator, per_class=True, classes=classes)
                viz.fit(X_train, y_train)
                viz.score(X_test, y_test)
                viz.show(outpath=os.path.join(os.getcwd(), f'viz/{viz.__class__.__name__}_{estimator.__class__.__name__}_{index}.png'), clear_figure=True)
                index = index + 1
            except:
                pass
            try:
                viz = ClassPredictionError(model = estimator, classes = classes)
                viz.fit(X_train, y_train)
                viz.score(X_test, y_test)
                viz.show(outpath=os.path.join(os.getcwd(), f'viz/{viz.__class__.__name__}_{estimator.__class__.__name__}_{index}.png'), clear_figure=True)
                index = index + 1
            except:
                LOGGER.warn(f'{viz.__class__.__name__} ERROR')
            try:
                viz = DiscriminationThreshold(model = estimator, classes = classes)
                viz.fit(X_train, y_train)
                viz.score(X_test, y_test)
                viz.show(outpath=os.path.join(os.getcwd(), f'viz/{viz.__class__.__name__}_{estimator.__class__.__name__}_{index}.png'), clear_figure=True)
                index = index + 1
            except:
                LOGGER.warn(f'{viz.__class__.__name__} ERROR')

    def report_classification(self, sort_by=None):
        scores = pd.DataFrame.from_dict(self.metrics, orient = 'index')
        if sort_by is not None:
            scores = scores.sort_values(by = sort_by, ascending=False)
        return scores





