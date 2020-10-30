from sklearn.ensemble import RandomForestClassifier
from models.model_factory import ModelFactory

@ModelFactory.register('classification-randomforestclassifier')
class RandomForestClassifierContainer(RandomForestClassifier):
    def __init__(self, **kwargs):
        super(RandomForestClassifierContainer, self).__init__(**kwargs)
