import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import dataloader


class RNN(nn.Module):
    def __init__(self, args, n_char):
        super(RNN, self).__init__()

        # Argument Define
        self.n_char = n_char
        self.hidden_dim = args.hidden_dim

        self.N_predict_layer = args.N_RNN_layer
        self.N_predict_FC = args.N_predictor_layer

        self.N_properties = args.N_properties

        self.dropout = args.dropout

        self.embedding = nn.Embedding(self.n_char,self.hidden_dim)

        self.GRUs = nn.GRU(self.hidden_dim,self.hidden_dim,self.N_predict_layer,dropout=self.dropout)

        self.predict = \
            nn.ModuleList([
                nn.Sequential(nn.Linear(self.hidden_dim,self.hidden_dim),
                              nn.Dropout(p=self.dropout),
                              nn.ReLU())
                for _ in range(self.N_predict_FC-1)] +
                [nn.Linear(self.hidden_dim,self.N_properties)
            ])

    def forward(self, x, l):
        x = self.embedding(x)
        x = x.permute((1,0,2))
        output, h_n = self.GRUs(x)
        selected = []
        for i in range(len(l)):
            selected.append(output[int(l[i]-1),int(i),:])
        retval = torch.stack(selected)
        for layer in self.predict:
            retval = layer(retval)

        return retval

if __name__ == "__main__":
    fn = "../data/test.txt"
    _, data_loader = dataloader.get_dataset_dataloader(fn, batch_size=2)
