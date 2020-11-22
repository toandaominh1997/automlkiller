import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from sklearn.datasets import make_classification
from layers.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron

class DeepFactorizationMachineModel(pl.LightningModule):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum = True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) *embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)


    def forward(self, x):
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
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
        return torch.from_numpy(self.items[index])
if __name__ == '__main__':
    dataset = FMDataset()
    field_dims = dataset.field_dims
    loader = DataLoader(dataset, batch_size = 4, num_workers=8)
    model = DeepFactorizationMachineModel(field_dims = [10],
                                      embed_dim = 40)
    for field in loader:
        out = model(field)
        print('out: ', out)
        print('shape: ', out.shape)
        break
