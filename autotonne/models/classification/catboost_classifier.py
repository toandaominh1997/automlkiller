from catboost import CatBoostClassifier
from autotonne.models.model_factory import ModelFactory

@ModelFactory.register('classification-catboostclassifier')
class CatBoostClassifierContainer(CatBoostClassifier):
    def __init__(self, **kwargs):
        super(CatBoostClassifierContainer, self).__init__(**kwargs)

