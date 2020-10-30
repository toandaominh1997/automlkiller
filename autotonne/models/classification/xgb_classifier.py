from xgboost import XGBClassifier
from models.model_factory import ModelFactory

@ModelFactory.register('classification-xgbclassifier')
class XGBClassifierContainer(XGBClassifier):
    def __init__(self, **kwargs):
        super(XGBClassifierContainer, self).__init__(**kwargs)
