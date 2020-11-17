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
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from yellowbrick.model_selection import *
from yellowbrick.features import *
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ROCAUC, PrecisionRecallCurve, ClassPredictionError, DiscriminationThreshold
from yellowbrick.target import BalancedBinningReference, ClassBalance, FeatureCorrelation
import matplotlib.pyplot as plt

from autotonne.utils.distributions import get_optuna_distributions

from autotonne.utils import LOGGER, can_early_stop

from datetime import datetime
# Writer will output to ./runs/ directory by default
log_dir = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
writer = SummaryWriter(log_dir = os.path.join(os.getcwd(), 'runs', log_dir))

class AUTOML(object):
    X = None
    y = None
    def __init__(self,
                 X,
                 y,
                 preprocess: bool = True,
                 **kwargs
                 ):
        super(AUTOML, self).__init__()
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

            hparam_dict = {}
            for key, values in scores.items():
                for i, value in enumerate(values):
                    if i not in hparam_dict.keys():
                        hparam_dict[i] = {}
                    hparam_dict[i][f'hparam/{key}'] = value
            for fold in hparam_dict.keys():
                writer.add_hparams({'name_model': name_model, 'KFold': str(fold + 1)}, hparam_dict[fold])
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
        LOGGER.info('TUNNING MODEL ...')
        best_params_model = {}
        model_grid = None

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



            model_grid.fit(self.X, self.y)
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
                                                            X = self.X,
                                                            y = self.y,
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
                                                X = self.X,
                                                y = self.y,
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
                                            X = self.X,
                                            y = self.y,
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
                y_pred = estimator.predict_proba(X)[:, 1]
                preds.append(y_pred)
            except:
                LOGGER.warn(f'{estimator.__class__.__name__} not function predict_proba')
        y_pred_proba = np.mean(np.vstack(preds), axis = 0)
        return y_pred_proba


    def predict_model(self,
                    X,
                    estimator = None,
                      probability_threshold = None,
                      rountd = 4,
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
    def feature_visualizer(self, classes = None, params = {}):
        if os.path.isdir(os.path.join(os.getcwd(), 'visualizer/')) == False:
            os.makedirs(os.path.join(os.getcwd(), 'visualizer/'))
        if classes is None:
            classes = pd.value_counts(self.y.values.flatten()).index.tolist()
        try:
            LOGGER.info('Visualizer RadViz')
            visualizer = RadViz(classes = classes, features = self.X.columns.tolist())
            if visualizer.__class__.__name__ in params.keys():
                visualizer = RadViz(**params[visualizer.__class__.__name__])
            visualizer.fit(self.X, self.y)
            visualizer.transform(self.X)
            visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
            plt.cla()
        except:
            LOGGER.warn('ERROR RadViz')
        try:
            LOGGER.info('Visualizer Rank1D')
            visualizer = Rank1D()
            if visualizer.__class__.__name__ in params.keys():
                visualizer = Rank1D(**params[visualizer.__class__.__name__])
            visualizer.fit(self.X, self.y)
            visualizer.transform(self.X)
            visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
            plt.cla()
        except:
            LOGGER.warn('ERROR Rank1D')
        try:
            LOGGER.info('Visualizer Rank2D')
            visualizer = Rank2D()
            if visualizer.__class__.__name__ in params.keys():
                visualizer = Rank2D(**params[visualizer.__class__.__name__])
            visualizer.fit(self.X, self.y)
            visualizer.transform(self.X)
            visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
            plt.cla()
        except:
            LOGGER.warn('ERROR Rank2D')
        try:
            LOGGER.info('Visualizer ParallelCoordinates')
            visualizer = ParallelCoordinates(classes = classes, features = self.X.columns.tolist(), shuffle=True)
            if visualizer.__class__.__name__ in params.keys():
                visualizer = ParallelCoordinates(**params[visualizer.__class__.__name__])
            visualizer.fit_transform(self.X, self.y)
            visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
            plt.cla()
        except:
            LOGGER.warn('ERROR ParallelCoordinates')
        try:
            LOGGER.info('Visualizer PCA 3D')
            visualizer = PCA(classes = classes, scale = True, projection = 3)
            if visualizer.__class__.__name__ in params.keys():
                visualizer = PCA(**params[visualizer.__class__.__name__])
            visualizer.fit_transform(self.X, self.y)
            visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
            plt.cla()
        except:
            LOGGER.warn('ERROR PCA 3D')
        try:
            LOGGER.info('Visualizer PCA Biplot')
            visualizer = PCA(classes = classes, scale = True, proj_features = True)
            if visualizer.__class__.__name__ in params.keys():
                visualizer = PCA(**params[visualizer.__class__.__name__])
            visualizer.fit_transform(self.X, self.y)
            visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
            plt.cla()
        except:
            LOGGER.warn('ERROR PCA Biplot')
        try:
            LOGGER.info('Visualizer Manifold')
            visualizer = Manifold(classes = classes)
            if visualizer.__class__.__name__ in params.keys():
                visualizer = Manifold(**params[visualizer.__class__.__name__])
            visualizer.fit_transform(self.X, self.y)
            visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
            plt.cla()
        except:
            LOGGER.warn('ERROR Manifold')

    def target_visualizer(self, classes = None, params = {'BalancedBinningReference': {'bins': 5}}):
        LOGGER.info('Initializing target visualizer')
        if os.path.isdir(os.path.join(os.getcwd(), 'visualizer/')) == False:
            os.makedirs(os.path.join(os.getcwd(), 'visualizer/'))
        visualizers = []
        y = self.y.squeeze()
        try:
            LOGGER.info('Visualizer BalancedBinningReference')
            visualizer = BalancedBinningReference()
            if visualizer.__class__.__name__ in params.keys():
                visualizer = BalancedBinningReference(**params[visualizer.__class__.__name__])
            visualizer.fit(y)
            visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
            plt.cla()
        except:
            LOGGER.warn('ERROR BalancedBinning')
        try:
            LOGGER.info('Visualizer CLassBalance')
            visualizer = ClassBalance()
            if visualizer.__class__.__name__ in params.keys():
                visualizer = ClassBalance(**params[visualizer.__class__.__name__])
            visualizer.fit(y)
            visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
            plt.cla()
        except:
            LOGGER.warn('ERROR ClassBalance')
        try:
            LOGGER.info('Visualizer Feature Correlation')
            visualizer = FeatureCorrelation(method = 'mutual_info-classification', feature_names = self.X.columns.tolist(), sort = True)
            if visualizer.__class__.__name__ in params.keys():
                visualizer = FeatureCorrelation(**params[visualizer.__class__.__name__])
            visualizer.fit(self.X, y)
            visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
            plt.cla()
        except:
            LOGGER.warn('ERROR FeatureCorrelation')
    def evaluate_visualizer(self, classes = None, params = {}):
        LOGGER.info('Initializing plot model')
        if os.path.isdir(os.path.join(os.getcwd(), 'visualizer/')) == False:
            os.makedirs(os.path.join(os.getcwd(), 'visualizer/'))
        if classes is None:
            classes = pd.value_counts(self.y.values.flatten()).index.tolist()
        visualizers = []
        for idx, (name_model, estimator) in enumerate(self.estimator.items()):
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(self.X, self.y, test_size = 0.2, stratify=self.y, random_state = 24)
            try:
                LOGGER.info('Visualizer ClassificationReport')
                visualizer = ClassificationReport(model = estimator, classes=classes)
                if visualizer.__class__.__name__ in params.keys():
                    visualizer = ClassificationReport(**params[visualizer.__class__.__name__])
                visualizer.fit(X_train, y_train)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath = os.path.join(os.getcwd(), f'visualizer/{visualizer.__class__.__name__}_{estimator.__class__.__name__}.png'))
                plt.cla()
            except:
                LOGGER.warn('ERROR ClassificationReport')
            try:
                LOGGER.info('Visualizer ConfusionMatrix')
                visualizer = ConfusionMatrix(model = estimator, classes=classes)
                if visualizer.__class__.__name__ in params.keys():
                    visualizer = ConfusionMatrix(**params[visualizer.__class__.__name__])
                visualizer.fit(X_train, y_train)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath = os.path.join(os.getcwd(), f'visualizer/{visualizer.__class__.__name__}_{estimator.__class__.__name__}.png'))
                plt.cla()
            except:
                LOGGER.warn('ERROR ConfusionMatrix')
            try:
                LOGGER.info('Visualizer ROCAUC')
                visualizer = ROCAUC(model = estimator,classes=classes)
                if visualizer.__class__.__name__ in params.keys():
                    visualizer = ROCAUC(**params[visualizer.__class__.__name__])
                visualizer.fit(X_train, y_train)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath = os.path.join(os.getcwd(), f'visualizer/{visualizer.__class__.__name__}_{estimator.__class__.__name__}.png'))
                plt.cla()
            except:
                LOGGER.warn('ERROR ROCAUC')
            try:
                LOGGER.info('Visualizer PrecisionRecallCurve')
                visualizer = PrecisionRecallCurve(model = estimator, per_class=True, classes=classes)
                if visualizer.__class__.__name__ in params.keys():
                    visualizer = PrecisionRecallCurve(**params[visualizer.__class__.__name__])
                visualizer.fit(X_train, y_train)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath = os.path.join(os.getcwd(), f'visualizer/{visualizer.__class__.__name__}_{estimator.__class__.__name__}.png'))
                plt.cla()
            except:
                LOGGER.warn('ERROR PrecisionRecallCurve')
            try:
                LOGGER.info('Visualizer ClassPredictionError')
                visualizer = ClassPredictionError(model = estimator, classes = classes)
                if visualizer.__class__.__name__ in params.keys():
                    visualizer = ClassPredictionError(**params[visualizer.__class__.__name__])
                visualizer.fit(X_train, y_train)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath = os.path.join(os.getcwd(), f'visualizer/{visualizer.__class__.__name__}_{estimator.__class__.__name__}.png'))
                plt.cla()
            except:
                LOGGER.warn('ERROR ClassPredictionError')
            try:
                LOGGER.info('Visualizer Discrimination Threshold')
                visualizer = DiscriminationThreshold(model = estimator, classes = classes)
                if visualizer.__class__.__name__ in params.keys():
                    visualizer = DiscriminationThreshold(**params[visualizer.__class__.__name__])
                visualizer.fit(X_train, y_train)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath = os.path.join(os.getcwd(), f'visualizer/{visualizer.__class__.__name__}_{estimator.__class__.__name__}.png'))
                plt.cla()
            except:
                LOGGER.warn('ERROR Discrimination Threshold')
    def model_selection_visualizer(self, classes = None, params = {}):

        for idx, (name_model, estimator) in enumerate(self.estimator.items()):
            cv = StratifiedKFold(n_splits=2, random_state=42)
            try:
                if visualizer.__class__.__name__ in params.keys():
                    LOGGER.info('Visualizer ValidationCurve')
                    visualizer = ValidationCurve(model = estimator, cv = cv, **params[visualizer.__class__.__name__])
                    visualizer.fit(self.X, self.y)
                    visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
                    plt.cla()
            except:
                LOGGER.warn('ERROR ValidationCurve')
            try:
                LOGGER.info('Visualizer LearningCurve')
                visualizer = CVScores(model = estimator, cv = cv)
                if visualizer.__class__.__name__ in params.keys():
                    visualizer = LearningCurve(**params[visualizer.__class__.__name__])
                visualizer.fit(self.X, self.y)
                visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
                plt.cla()
            except:
                LOGGER.warn('ERROR LearningCurve')
            try:
                LOGGER.info('Visualizer CVScores')
                visualizer = CVScores(model = estimator, cv = cv)
                if visualizer.__class__.__name__ in params.keys():
                    visualizer = CVScores(**params[visualizer.__class__.__name__])
                visualizer.fit(self.X, self.y)
                visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
                plt.cla()
            except:
                LOGGER.warn('ERROR CVScores')
            try:
                LOGGER.info('Visualizer FeatureImportances')
                visualizer = FeatureImportances(estimator)
                if visualizer.__class__.__name__ in params.keys():
                    visualizer = FeatureImportances(**params[visualizer.__class__.__name__])
                visualizer.fit(self.X, self.y)
                visualizer.show(outpath = os.path.join(os.getcwd(), f"visualizer/{visualizer.__class__.__name__}.png"))
                plt.cla()
            except:
                LOGGER.warn('ERROR FeatureImportances')
    def report_classification(self, sort_by=None):
        scores = pd.DataFrame.from_dict(self.metrics, orient = 'index')
        if sort_by is not None:
            scores = scores.sort_values(by = sort_by, ascending=False)
        return scores




