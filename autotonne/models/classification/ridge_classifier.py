from sklearn.linear_model import RidgeClassifier
from models.model_factory import ModelFactory

@ModelFactory.register('classification-ridgeclassifier')
class RidgeClassifierContainer(RidgeClassifier):
    def __init__(self, **kwargs):
        super(RidgeClassifierContainer, self).__init__(**kwargs)