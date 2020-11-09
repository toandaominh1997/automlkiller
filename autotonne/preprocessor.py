from enum import Enum, auto
import numpy as np
import pandas as pd
import sklearn
from imblearn.pipeline import Pipeline
from autotonne.preprocess.preprocess import *
from autotonne.preprocess.preprocee_factory import PreprocessFactory
from autotonne.utils import LOGGER

class Preprocess(object):
    def __init__(self, **kwargs):
        super(Preprocess, self).__init__()
        self.preprocess_objs = []
        for name_registry, preprocess_param in kwargs.items():
            name_registry = '-'.join(['preprocess', name_registry])
            if name_registry not in PreprocessFactory.registry.keys():
                LOGGER.info(f'{name_registry} not PreprocessFactory')
            LOGGER.info('preprocess registry {}'.format(name_registry))
            preprocess_object = PreprocessFactory.create_executor(name = name_registry,
                                                                  **preprocess_param)
            self.preprocess_objs.append(preprocess_object)

    def fit(self, X, y = None, **fit_params):
        X = X.copy()
        if y is not None:
            y = y.copy()
            if isinstance(y, pd.Series):
                y = pd.DataFrame(y)
            y.columns = [str(col) for col in y.columns.tolist()]
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        X.columns = [str(col) for col in X.columns.tolist()]
        X = X.loc[:, ~X.columns.duplicated()]
        X.dropna(axis=1, how='all', inplace=True)
        for obj in self.preprocess_objs:
            obj.fit(X, y)

    def transform(self, X, y = None, **fit_params):
        X = X.copy()
        if y is not None:
            y = y.copy()
            if isinstance(y, pd.Series):
                y = pd.DataFrame(y)
            y.columns = [str(col) for col in y.columns.tolist()]

        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        X.columns = [str(col) for col in X.columns.tolist()]
        X = X.loc[:, ~X.columns.duplicated()]
        X.dropna(axis=1, how='all', inplace=True)
        data = X, y
        for obj in self.preprocess_objs:
            data = obj.transform(*data)
        if isinstance(data, pd.DataFrame) ==False:
            return data[0]
        return data
    def fit_transform(self, X, y = None):
        for obj in self.preprocess_objs:
            obj.fit(X, y)
            X, y = obj.transform(X, y)

        if y is None:
            return X
        return X, y




