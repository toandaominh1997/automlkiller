import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from sklearn.datasets import make_classification
from layers.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron

from torch.utils.data import Dataset, DataLoader



class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))



class FMDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.items = np.random.randint(10, size=(1000, 20))
        self.targets = np.random.randn(1000, 10)
        self.field_dims = np.max(self.items, axis = 0) + 1

    def __len__(self):
        return 1000
    def __getitem__(self, index):
        return torch.from_numpy(self.items[index]), torch.from_numpy(self.targets[index])
if __name__ == '__main__':
    dataset = FMDataset()
    field_dims = dataset.field_dims
    loader = DataLoader(dataset, batch_size = 4, num_workers=8)
    model = FactorizationMachineModel(field_dims = [10],
                                      embed_dim = 40)
    for field, target in loader:
        out = model(field)
        print('out: ', out)
        print('shape: ', out.shape)
        break
