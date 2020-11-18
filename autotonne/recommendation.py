import  torch
import torch.nn as nn
import pytorch_lightning as pl


class RCDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(RCDataset, self).__init__()
    def __len__(self):
        return 100
    def __getitem__(self, index):




class AUTORC(object):
    def __init__(self,
                 X,
                 y,
                 preprocess = True,
                 **kwargs):
        super(AUTORC, self).__init__()


