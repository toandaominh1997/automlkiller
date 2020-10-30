from sklearn.ensemble import GradientBoostingClassifier
from models.model_factory import ModelFactory

@ModelFactory.register('classification-gradientboostingclassifier')
class GradientBoostingClassifierContainer(GradientBoostingClassifier):
    def __init__(self, **kwargs):
        super(GradientBoostingClassifierContainer, self).__init__(**kwargs)
