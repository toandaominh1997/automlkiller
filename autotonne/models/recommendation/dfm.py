import torch
import torch.nn as nn
import pytorch_lightning as pl
from .layers import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


@ModelFactory.register('recommendation-dfm')
class DeepFactorizationMachineModelContainer():
    def __init__(self, **kwargs):

        self.etimator = DeepFactorizationMachineModel(**kwargs)



class DeepFactorizationMachineModel(pl.LightningModule):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = nn.CrossEntropyLoss()(out, y)

        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer
