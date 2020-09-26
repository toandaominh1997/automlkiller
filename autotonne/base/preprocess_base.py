from abc import ABC, abstractmethod

class PreprocessBase(ABC):
    def __init__(self):
        super(PreprocessBase, self).__init__()
    @abstractmethod
    def fit(self):
        raise NotImplementedError()
    @abstractmethod
    def transform(self):
        raise NotImplementedError()

