import os
import numpy as np
import pandas as pd
import warnings
from autotonne.clustering import Clustering
from autotonne.utils import save_model, load_model
from sklearn.datasets import make_biclusters, make_regression


warnings.filterwarnings('ignore')


if __name__ =='__main__':
    # X = make_biclusters(shape=(1000, 50), n_clusters=5)
    X, _ = make_regression(n_samples=1000, n_features=50)
    X = pd.DataFrame(X)
    obj = Clustering(
        categoryencoder = True,
        categoryencoder_method = 'onehotencoder',
        groupsimilarfeature = False
    )
    obj.create_model(X, num_clusters= 5, estimator='kmeans')
    labels = obj.predict(X)
    print('labels: ', labels)
    save_model(obj, os.path.join(os.getcwd(), 'data/modeling.pkl'))
    load_model(model_path = os.path.join(os.getcwd(), 'data/modeling.pkl'))
