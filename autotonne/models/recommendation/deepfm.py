import torch
import torch.nn as nn
import pytorch_lightning as pl

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

if __name__ == '__main__':
    X, y = make_classification(1000, 40)
    inputs = torch.from_numpy(X)
    print('inputs: ', inputs.size())
    model = DeepFactorizationMachineModel(field_dims = [40],
                                          embed_dim = 40,
                                          mlp_dims = 40,
                                          dropout = 0.5)
    out = model(inputs)
