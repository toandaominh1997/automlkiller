import pickle

from sklearn.pipeline import Pipeline
from .logger import LOGGER



def save_model(model, model_path):
    LOGGER.info(f"Initializing save_model({model.__class__.__name__}, {model_path})")
    with open(model_path, 'wb') as fp:
        pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info('Preprocess and Model Succesfully Saved')
    LOGGER.info('"save_model() succesfully completed......................................"')

def load_model(model_path):
    LOGGER.info(f"Initializing load_model({model_path})")
    with open(model_path, 'rb') as fp:
        obj = pickle.load(fp)
        LOGGER.info("Transformation Pipeline and Model Successfully Loaded")
    return obj

def supports_partial_fit(estimator, params: dict = None) -> bool:
    # special case for MLP
    from sklearn.neural_network import MLPClassifier

    if isinstance(estimator, MLPClassifier):
        try:
            if (
                params and "solver" in params and "lbfgs" in list(params["solver"])
            ) or estimator.solver == "lbfgs":
                return False
        except:
            return False

    if isinstance(estimator, Pipeline):
        return hasattr(estimator.steps[-1][1], "partial_fit")

    return hasattr(estimator, "partial_fit")
def can_early_stop(
    estimator, consider_partial_fit, consider_warm_start, consider_xgboost, params,
):
    """
    From https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_basesearch.py.

    Helper method to determine if it is possible to do early stopping.
    Only sklearn estimators with ``partial_fit`` or ``warm_start`` can be early
    stopped. warm_start works by picking up training from the previous
    call to ``fit``.

    Returns
    -------
        bool
            if the estimator can early stop
    """


    from sklearn.tree import BaseDecisionTree
    from sklearn.ensemble import BaseEnsemble

    try:
        base_estimator = estimator.steps[-1][1]
    except:
        base_estimator = estimator

    if consider_partial_fit:
        can_partial_fit = supports_partial_fit(base_estimator, params=params)
    else:
        can_partial_fit = False

    if consider_warm_start:
        is_not_tree_subclass = not issubclass(type(base_estimator), BaseDecisionTree)
        is_ensemble_subclass = issubclass(type(base_estimator), BaseEnsemble)
        can_warm_start = hasattr(base_estimator, "warm_start") and (
            (
                hasattr(base_estimator, "max_iter")
                and is_not_tree_subclass
                and not is_ensemble_subclass
            )
            or (is_ensemble_subclass and hasattr(base_estimator, "n_estimators"))
        )
    else:
        can_warm_start = False

    if consider_xgboost:
        from xgboost.sklearn import XGBModel

        is_xgboost = isinstance(base_estimator, XGBModel)
    else:
        is_xgboost = False

    LOGGER.info(
        f"can_partial_fit: {can_partial_fit}, can_warm_start: {can_warm_start}, is_xgboost: {is_xgboost}"
    )

    return can_partial_fit or can_warm_start or is_xgboost
