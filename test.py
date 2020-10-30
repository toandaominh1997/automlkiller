import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import make_classification
from autotonne.classification import Classification
print('RUN CIRCLE CI')


if __name__ =='__main__':
    X, y = make_classification(n_samples=1000000, n_features=50) 
    data = pd.DataFrame(X) 
    data['target'] = y
    Classification(data = data, target='target').compare_models()
    Classification(data = data, target='target').tune_models()
