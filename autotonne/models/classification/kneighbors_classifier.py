
from sklearn.neighbors import KNeighborsClassifier
from models.model_factory import ModelFactory

@ModelFactory.register('classification-kneighborsclassifier')
class KNeighborsClassifierContainer(KNeighborsClassifier):
    def __init__(self, **kwargs):
        super(KNeighborsClassifierContainer, self).__init__(**kwargs)
