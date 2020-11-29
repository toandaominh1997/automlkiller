import numpy as np
import pandas as pd
import sklearn



class Metric(object):
    def __init__():
        super().__init__()

    def merge_rating_true_pred(self,
                               rating_true,
                               rating_pred,
                               col_user = 'userID',
                               col_item = 'itemID',
                               col_rating = 'rating',
                               col_prediction = 'prediction'):
        suffixes = ["_true", "_pred"]
        rating_true_pred = pd.merge(rating_true, rating_pred, on = [col_user, col_item], suffixes = suffixes)
        if col_rating in rating_pred.columns:
            col_rating = col_rating + suffixes[0]
        if col_prediction in rating_true.columns:
            col_prediction = col_prediction + suffixes[1]
        return rating_true_pred[col_rating], rating_true_pred[col_prediction]
    def rmse(self,
               rating_true,
               rating_pred,
               col_user = 'userID',
               col_item = 'itemID',
               col_rating = 'rating',
               col_prediction = 'prediction'):
        y_true, y_pred = merge_rating_true_pred(rating_true = rating_true,
                                                rating_pred = rating_pred,
                                                col_user = col_user,
                                                col_item = col_item,
                                                col_rating = col_rating,
                                                col_prediction = col_prediction)
        return np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))

    def mae(self,
               rating_true,
               rating_pred,
               col_user = 'userID',
               col_item = 'itemID',
               col_rating = 'rating',
               col_prediction = 'prediction'):
        y_true, y_pred = merge_rating_true_pred(rating_true = rating_true,
                                                rating_pred = rating_pred,
                                                col_user = col_user,
                                                col_item = col_item,
                                                col_rating = col_rating,
                                                col_prediction = col_prediction)
        return np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
