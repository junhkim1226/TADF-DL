import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import dataloader
from layers import GCN_layer


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()

        # Argument Define
        self.dim_of_conv_layer = args.hidden_dim
        self.dim_of_FC = args.hidden_dim

        self.N_predict_layer = args.N_GCN_layer
        self.N_predict_FC_layer = args.N_predictor_layer

        self.N_properties = args.N_properties
        self.N_atom_features = utils.N_atom_features
        self.N_bond_features = utils.N_bond_features
        self.N_extra_atom_features = utils.N_extra_atom_features
        self.N_extra_bond_features = utils.N_extra_bond_features

        self.dropout = args.dropout

        # Layer Define
        self.embed_graph = nn.Linear(self.N_atom_features + self.N_extra_atom_features, self.dim_of_conv_layer)

        self.GCN_layers = nn.ModuleList(
                [GCN_layer(self.dim_of_conv_layer,self.dim_of_conv_layer,self.dropout) for _ in
                    range(self.N_predict_layer)])

        self.readout = nn.Linear(self.dim_of_conv_layer,self.dim_of_FC)

        self.predict = \
            nn.ModuleList([
                nn.Sequential(nn.Linear(self.dim_of_FC,self.dim_of_FC),
                              nn.Dropout(p=self.dropout),
                              nn.ReLU())
                for _ in range(self.N_predict_FC_layer-1)] +
                [nn.Linear(self.dim_of_FC,self.N_properties)
            ])

    def forward(self, x, A ,atom_mask):
        x = self.embed_graph(x)

        for layer in self.GCN_layers:
            x = layer(x, A)

        x = self.readout(x)
        x = x.mean(1)
        x = F.relu(x)

        for layer in self.predict:
            x = layer(x)

        return x

if __name__ == "__main__":
    fn = "../data/test.txt"
    _, data_loader = dataloader.get_dataset_dataloader(fn, batch_size=2)
