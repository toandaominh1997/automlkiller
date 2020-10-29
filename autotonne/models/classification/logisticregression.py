from sklearn.linear_model import LogisticRegression
from models.model_factory import ModelFactory

@ModelFactory.register('classification-logisticregression')
class LogisticRegressionContainer(LogisticRegression):
    def __init__(self, **kwargs):
        super(LogisticRegressionContainer, self).__init__(**kwargs)