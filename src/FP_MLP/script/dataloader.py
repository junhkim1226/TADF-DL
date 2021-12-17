import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import utils

def get_dataset_dataloader(fn, batch_size=8, shuffle=True,
                           num_workers=1, length=None):
    dataset = FingerprintDataset(fn, length)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=shuffle, drop_last=False,
                            pin_memory=True)
    return dataset, dataloader


class FingerprintDataset(Dataset):

    def __init__(self, fn, length=None):
        with open(fn, "r") as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]
        self.data = lines[:length]

        self.fp_list = []
        self.target = []
        self.key = []
        self.smiles = []
        for word in self.data:
            key, smiles,H,L,S,T = word

            m = Chem.MolFromSmiles(smiles)
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m,2,nBits=1024)
            fp = torch.from_numpy(np.array(fp))

            self.key.append(key)
            self.smiles.append(smiles)
            self.target.append((float(H),float(L),float(S),float(T)))
            self.fp_list.append(fp)

    def __getitem__(self, idx):

        feature_dict = dict()

        feature_dict["key"] = self.key[idx]
        feature_dict["target"] = self.target[idx]
        feature_dict["fp"] = self.fp_list[idx]
        return feature_dict

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    fn='../data/test.txt'

    dataset, data_loader = get_dataset_dataloader(
        fn, batch_size=4, num_workers=1)
