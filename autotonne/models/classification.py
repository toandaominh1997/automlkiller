import numpy as np
import pandas as  pd
from autotonne.base.model_base import ModelBase
from autotonne.base.object_factory import ObjectFactory
# models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from autotonne.utils.distributions import *


@ObjectFactory.register('logistic_regression_container')
class LogisticRegressionContainer(LogisticRegression):
    def __init__(self, **kwargs):
        super(LogisticRegressionContainer, self).__init__(**kwargs)
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        tune_grid['penalty'] = ['l2', 'none']
        tune_grid['C'] = np_list_arange(0, 10, 0.001, inclusive=True)
        tune_distributions['C'] = UniformDistribution(0, 10)



class KNeighborsClassifierContainer(KNeighborsClassifier):
    def __init__(self, **kwargs):
        super(KNeighborsClassifierContainer, self).__init__(**kwargs)
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        tune_grid['penalty'] = ['l2', 'none']
        tune_grid['C'] = np_list_arange(0, 10, 0.001, inclusive=True)
        tune_distributions['C'] = UniformDistribution(0, 10)

# class GaussianNBContainer(GaussianNB):
#     def __init__(self, **kwargs)
