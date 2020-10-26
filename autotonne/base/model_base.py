from abc import ABC, abstractmethod


class ModelBase(ABC):
    def __init__(self):
        super(ModelBase, self).__init__()
    @abstractmethod
    def fit(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        raise NotImplementedError()

    @abstractmethod
    def predict_proba(self):
        raise NotImplementedError()
