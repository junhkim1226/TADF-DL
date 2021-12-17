import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN_layer(torch.nn.Module):

    def __init__(self, in_feature, out_feature,dropout):
        super().__init__()
        self.in_feature = in_feature
        self.W = nn.Linear(in_feature, out_feature)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, adj):

        h = self.W(x)
        h = self.dropout(h)
        h = torch.matmul(adj,h)
        h = h + x
        h = self.activation(h)

        return h
