from enum import Enum, auto
import pandas as pd
from autotonne.models.classification import *

class MLUSECASE(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()
    CLUSTERING = auto()
    ANAMALY = auto()

class Classification(object):
    def __init__(self,
                 data: pd.DataFrame,
                 target: str,
                 test_size: float = 0.2,
                 preprocess: bool = True,
                 ):
        super(Classification, self).__init__()


    def compare_models(self,
                       cross_validation: bool = True,
                       sort: str = 'Accuracy',
                       verbose: bool = True):

