import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import dataloader
from layers import MLP


class ConcatMLP(nn.Module):
    def __init__(self, args):
        super(ConcatMLP, self).__init__()

        # Argument Define
        self.dim_of_desc = args.desc_dim
        self.dim_of_Linear = args.hidden_dim

        self.N_predict_layer = args.N_MLP_layer
        self.N_predict_FC = args.N_predictor_layer

        self.N_properties = args.N_properties

        self.dropout = args.dropout

        self.embedding=nn.Linear(self.dim_of_desc,self.dim_of_Linear)

        self.MLPs= nn.ModuleList([
            MLP(self.dim_of_Linear,self.dim_of_Linear,self.dropout) for _ in range(self.N_predict_layer)])

        self.predict = \
            nn.ModuleList([
                nn.Sequential(nn.Linear(self.dim_of_Linear,self.dim_of_Linear),
                              nn.Dropout(p=self.dropout),
                              nn.ReLU())
                for _ in range(self.N_predict_FC-1)] +
                [nn.Linear(self.dim_of_Linear,self.N_properties)
            ])

    def forward(self, x):
        x = self.embedding(x)

        for layer in self.MLPs:
            x = layer(x)

        for layer in self.predict:
            x = layer(x)
        return x


if __name__ == "__main__":
    fn = "../data/test.txt"
    _, data_loader = dataloader.get_dataset_dataloader(fn, batch_size=2)
