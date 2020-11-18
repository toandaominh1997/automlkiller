import  torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl
from multiprocess import cpu_count

class RCDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(RCDataset, self).__init__()
        try:
            self.X = X.values
            self.y = y.values
        except:
            self.X = X
            self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(self.y[index])




class AUTORC(object):
    def __init__(self,
                 X,
                 y,
                 preprocess = True,
                 batch_size = 4,
                 num_workers = 4,
                 test_size = 0.2,
                 **kwargs):
        super(AUTORC, self).__init__()

        self.X = X
        self.y = y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
        train_dataset = RCDataset(X_train, y_train)
        test_dataset = RCDataset(X_test, y_test)


        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size = batch_size,
                                                       shuffle = True,
                                                       num_workers = num_workers,
                                                       pin_memory = True)

        self.train_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size = batch_size,
                                                       shuffle = True,
                                                       num_workers = num_workers,
                                                       pin_memory = True)
    def get_dataloader(self, X, y, batch_size):
        dataset = RCDataet(X, y)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size = self.batch_size,
                                                  shuffle = True,
                                                  num_workers = cpu_count(),
                                                  pin_memory = True
                                                  )
        return data_loader
    def create_model(self,
                     estimator,
                     cv = 2,
                     verbose = True,
                     n_jobs = -1
                     ):
        skf = StratifiedKFold(n_splits = 2)
        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]






