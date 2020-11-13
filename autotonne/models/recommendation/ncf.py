import torch
import torch.nn as nn
import pytorch_lightning as pl

from autotonne.models.model_factory import ModelFactory



class NCFContainer(object):
    def __init__(self, **kwargs):

        self.model = NCF()
        self.estimator = pl.Trainer()


class NCF(pl.LightningModule):
    def __init__(self):
        super().__init__()
        num_users = 10
        num_items = 10
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)



        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)


    def forward(self, user_features, item_features):
        user_embedding = self.user_embedding(user_features)
        item_embedding = self.item_embedding(item_features)

        output = torch.cat([user_embedding, item_embedding], dim = -1)

        output = nn.ReLU()(self.fc1(output))
        output = nn.ReLU()(self.fc2(output))

        out = nn.Sigmoid()(self.output(output))
        return out
    def training_step(self, batch):
        user_features, item_features, interaction = batch
        out = self.forward(user_features, item_features)

        loss = nn.CrossEntropyLoss()(interaction, out)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer
