from lightgbm import LGBMClassifier
from models.model_factory import ModelFactory

@ModelFactory.register('classification-lgbmclassifier')
class LGBMClassifierContainer(LGBMClassifier):
    def __init__(self, **kwargs):
        random_state = 24
        super(LGBMClassifierContainer, self).__init__(random_state=random_state, **kwargs)
