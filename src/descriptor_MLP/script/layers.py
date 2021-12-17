import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_feature, out_feature, dropout):
        super().__init__()
        self.in_feature = in_feature
        self.dropout = dropout
        self.Linear=nn.Linear(in_feature,out_feature)
        self.dropout = nn.Dropout(p=self.dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        skip_x = x
        x = self.Linear(x)
        x = self.dropout(x)
        x = x+skip_x
        x = self.activation(x)

        return x

if __name__ == "__main__":
    MLP = MLP(12,12,0.4)
