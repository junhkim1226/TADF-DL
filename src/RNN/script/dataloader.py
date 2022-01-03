import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from descriptastorus.descriptors import rdNormalizedDescriptors
import utils

def collate_fn(batch) :
    sample = dict()
    n_char = batch[0]['n_char']
    seq = torch.nn.utils.rnn.pad_sequence([b['seq'] for b in batch], batch_first=True, padding_value = n_char-1)
    length = torch.Tensor([b['length'] for b in batch])
    smiles = [b['smiles'] for b in batch]
    key = [b['key'] for b in batch]
    target = [b['target'] for b in batch]
    sample['key'] = key
    sample['target'] = target
    sample['seq'] = seq
    sample['length'] = length
    sample['smiles'] = smiles
    return sample

def get_dataset_dataloader(fn, maxlen,batch_size=8, shuffle=True,
                           num_workers=1, length=None):
    dataset = RNNDataset(fn, maxlen,length)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=shuffle, drop_last=False,
                            pin_memory=True)
    return dataset, dataloader

def get_c_to_i(fn):
    c_to_i = {}
    smiles_list = []

    with open(fn, "r") as f:
        lines = f.readlines()
        for line in lines:
            word= line.strip().split()
            key, w_smiles,w_H,w_L,w_S,w_T = word
            smiles_list.append(w_smiles)

    for smiles in smiles_list:
        for letter in smiles:
            if letter not in c_to_i:
                c_to_i[letter] = len(c_to_i)
    c_to_i['X'] = len(c_to_i)
    import pickle
    f = open('./c_to_i.pkl','wb')
    pickle.dump(c_to_i,f)
    f.close()

def adjust_smiles(smiles_list,maxlen):
    for i in range(len(smiles_list)):
        smiles_list[i] = smiles_list[i].ljust(maxlen,'X')

class RNNDataset(Dataset):

    def __init__(self, fn, maxlen, c_to_i, length=None):
        with open(fn, "r") as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]
        self.data = lines[:length]

        self.fp_list = []
        self.target = []
        self.key = []
        self.smiles = []
        for word in self.data:
            key, w_smiles,w_H,w_L,w_S,w_T = word

            if len(w_smiles) < maxlen:
                self.key.append(key)
                self.smiles.append(w_smiles)
                self.target.append((float(w_H),float(w_L),float(w_S),float(w_T)))

        self.c_to_i = c_to_i
        adjust_smiles(self.smiles,maxlen)
        self.seq_list = self.encode_smiles()
        self.length_list = []
        for seq in self.seq_list:
            self.length_list.append(len(seq))

    def encode_smiles(self):
        smiles_list = self.smiles
        c_to_i = self.c_to_i
        seq_list = []
        for smiles in smiles_list:
            seq=[]
            for s in smiles:
                seq.append(c_to_i[s])
            seq = torch.from_numpy(np.array(seq))
            seq_list.append(seq)
        return seq_list


    def __getitem__(self, idx):

        feature_dict = dict()

        feature_dict["key"] = self.key[idx]
        feature_dict["target"] = self.target[idx]
        feature_dict["smiles"] = self.smiles[idx]
        feature_dict["seq"] = self.seq_list[idx]
        feature_dict["length"] = self.length_list[idx]
        feature_dict["n_char"] = len(self.c_to_i)
        return feature_dict

    def __len__(self):
        return len(self.data)
